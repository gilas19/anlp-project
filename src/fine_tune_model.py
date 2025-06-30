#!/usr/bin/env python3
"""
Fine-tuning script for Flan-T5-large on Reddit ChangeMyView dataset
for improved argumentative reasoning in fact verification debates.
"""

import pandas as pd
import torch
import os
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CMVDataProcessor:
    """Processes ChangeMyView dataset for fine-tuning"""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load and preprocess CMV conversation tree dataset"""
        # Handle relative path from project root
        if not os.path.isabs(file_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            file_path = os.path.join(project_root, 'data', file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        logger.info(f"Loaded {len(conversations)} conversation trees")
        return conversations
    
    def create_training_prompts(self, conversations: List[Dict]) -> List[Dict[str, str]]:
        """Create training prompts for debate-style argumentative reasoning"""
        prompts = []
        
        for conversation in conversations:
            title = conversation['title']
            op_text = conversation['op_text']
            tree = conversation['tree']
            
            # Extract debate exchanges from conversation tree
            debate_exchanges = self._extract_debate_exchanges(tree, title, op_text)
            prompts.extend(debate_exchanges)
        
        logger.info(f"Created {len(prompts)} training examples from conversation trees")
        return prompts
    
    def _extract_debate_exchanges(self, tree: List[Dict], title: str, op_text: str) -> List[Dict[str, str]]:
        """Extract debate-style exchanges from conversation tree"""
        exchanges = []
        
        # Find meaningful exchanges where someone responds to the OP
        for node in tree:
            if node['speaker'] != tree[0]['speaker'] and node['text'] and len(node['text'].strip()) > 50:
                # Create debate-style prompt similar to the debate system
                input_text = f"""Claim: "{title}"
Original Position: "{op_text[:500]}..."

Respond to this position with a thoughtful counterargument or perspective. Reference specific points and build a logical argument (max 3 sentences)."""
                
                target_text = node['text'].strip()
                
                # Skip deleted or very short responses
                if target_text != "[deleted]" and len(target_text) > 30:
                    exchanges.append({
                        'input_text': input_text,
                        'target_text': target_text
                    })
                
                # Also extract child responses for multi-turn debate simulation
                if node['children']:
                    child_exchanges = self._extract_child_exchanges(node, title, op_text)
                    exchanges.extend(child_exchanges)
        
        return exchanges
    
    def _extract_child_exchanges(self, parent_node: Dict, title: str, op_text: str) -> List[Dict[str, str]]:
        """Extract exchanges from child nodes to simulate multi-turn debates"""
        exchanges = []
        
        for child in parent_node['children']:
            if child['text'] and child['text'] != "[deleted]" and len(child['text'].strip()) > 50:
                # Create context from parent argument
                parent_arg = parent_node['text'][:300] + "..." if len(parent_node['text']) > 300 else parent_node['text']
                
                input_text = f"""Claim: "{title}"
Previous Argument: "{parent_arg}"

Respond to the previous argument with a counter-response or elaboration. Build on the debate and provide specific reasoning (max 3 sentences)."""
                
                target_text = child['text'].strip()
                
                exchanges.append({
                    'input_text': input_text,
                    'target_text': target_text
                })
        
        return exchanges
    
    def create_fact_verification_prompts(self, conversations: List[Dict]) -> List[Dict[str, str]]:
        """Create fact verification debate training prompts similar to debate system"""
        prompts = []
        
        for conversation in conversations:
            title = conversation['title']
            op_text = conversation['op_text']
            tree = conversation['tree']
            
            # Extract structured debates for fact verification training
            fact_debates = self._extract_fact_verification_debates(tree, title, op_text)
            prompts.extend(fact_debates)
        
        logger.info(f"Created {len(prompts)} fact verification debate examples")
        return prompts
    
    def _extract_fact_verification_debates(self, tree: List[Dict], title: str, op_text: str) -> List[Dict[str, str]]:
        """Extract structured debates that simulate fact verification scenarios"""
        debates = []
        
        # Find sequences of arguments and counter-arguments
        for i, node in enumerate(tree):
            if node['speaker'] != tree[0]['speaker'] and node['text'] and len(node['text'].strip()) > 50:
                # Create Agent A (supporting) prompt
                support_prompt = f"""You are participating in a structured fact-verification debate.

Claim: "{title}"
Evidence: "{op_text[:400]}..."

Your Role: SUPPORT the claim using the evidence provided.

Guidelines:
- Reference specific details from the evidence
- Build logical arguments connecting evidence to your position
- Stay focused and persuasive (max 3 sentences)
- Base arguments on factual analysis

Agent A: """
                
                # Create Agent B (refuting) prompt with context
                if node['children']:
                    for child in node['children']:
                        if child['text'] and child['text'] != "[deleted]" and len(child['text'].strip()) > 50:
                            refute_prompt = f"""You are participating in a structured fact-verification debate.

Claim: "{title}"
Evidence: "{op_text[:400]}..."

Your role is to REFUTE the claim using the evidence and respond thoughtfully to your opponent's arguments.
Reference specific points made by Agent A where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences.

Debate so far:
Agent A: {node['text'][:200]}...

Agent B: """
                            
                            debates.append({
                                'input_text': support_prompt,
                                'target_text': node['text'].strip()
                            })
                            
                            debates.append({
                                'input_text': refute_prompt,
                                'target_text': child['text'].strip()
                            })
        
        return debates
    
    def tokenize_function(self, examples):
        """Tokenize input and target texts"""
        inputs = [ex for ex in examples['input_text']]
        targets = [ex for ex in examples['target_text']]
        
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_length, 
            truncation=True, 
            padding=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, 
                max_length=self.max_length, 
                truncation=True, 
                padding=True
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

def fine_tune_model(
    model_name: str = "google/flan-t5-large",
    data_path: str = "cmv_10_conversation_trees.json",
    output_dir: str = "./finetuned_model",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    training_format: str = "debate"  # "debate" for fact verification format, "general" for general argumentative
):
    """Fine-tune Flan-T5 on CMV dataset"""
    
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Process data
    processor = CMVDataProcessor(tokenizer)
    conversations = processor.load_data(data_path)
    
    # Create training data from conversation trees based on format
    if training_format == "debate":
        training_data = processor.create_fact_verification_prompts(conversations)
        logger.info("Using debate-style fact verification training format")
    else:
        training_data = processor.create_training_prompts(conversations)
        logger.info("Using general argumentative training format")
    
    # Use subset for faster training if needed
    if len(training_data) > 1000:
        training_data = training_data[:1000]
        logger.info(f"Using subset of {len(training_data)} examples for training")
    
    dataset = Dataset.from_list(training_data)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        processor.tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # Split into train/val
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    # Use debate format by default to match the actual debate system
    fine_tune_model(training_format="debate")