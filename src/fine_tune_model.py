#!/usr/bin/env python3
"""
Fine-tuning script for google/flan-t5-large on CMV conversation data
for generating better debate responses using LoRA (Low-Rank Adaptation)
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Tuple
import os
import re
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebateDataset(Dataset):
    """Dataset for CMV conversation data formatted for debate response generation"""
    
    def __init__(self, data: List[Dict], tokenizer, max_input_length=512, max_target_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the debate prompt as input and the response as target
        input_text = item['prompt']
        target_text = item['response']
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': target_encoding.input_ids.squeeze()
        }

def extract_debate_training_pairs(conversation_data: List[Dict]) -> List[Dict]:
    """
    Extract debate training pairs from CMV conversation trees.
    Creates prompt-response pairs for training debate agents.
    """
    training_pairs = []
    
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove reddit formatting
        text = re.sub(r'&gt;.*?\n', '', text)  # Remove quotes
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()
    
    def determine_stance(text: str, is_op_response: bool = False) -> str:
        """Determine if text supports or refutes the claim"""
        text_lower = text.lower()
        
        # Check for delta (view changed)
        if 'delta' in text_lower or '∆' in text:
            return 'refute' if is_op_response else 'support'
        
        # Look for agreement/disagreement indicators
        support_indicators = ['agree', 'correct', 'right', 'exactly', 'true', 'valid point']
        refute_indicators = ['disagree', 'wrong', 'incorrect', 'however', 'but', 'actually', 'not quite']
        
        support_count = sum(1 for indicator in support_indicators if indicator in text_lower)
        refute_count = sum(1 for indicator in refute_indicators if indicator in text_lower)
        
        if support_count > refute_count:
            return 'support'
        elif refute_count > support_count:
            return 'refute'
        else:
            return 'refute' if is_op_response else 'support'  # Default based on context
    
    for conversation in conversation_data:
        title = conversation.get('title', '')
        op_text = conversation.get('op_text', '')
        
        # Extract claim from title
        claim = title.replace('CMV:', '').strip()
        if not claim or len(claim) < 10:
            continue
            
        # Create context from OP text
        op_context = clean_text(op_text)[:300]  # Truncate for context
        
        def extract_from_tree(node, parent_text='', depth=0, history=[]):
            if depth > 4 or not node:  # Limit depth
                return
                
            node_text = clean_text(node.get('text', ''))
            speaker = node.get('speaker', '')
            
            if not node_text or node_text == '[deleted]' or len(node_text) < 20:
                return
                
            # Skip very short responses or system messages
            if len(node_text) < 30 or speaker in ['DeltaBot', '[deleted]']:
                return
            
            # Determine stance and role
            is_op_response = (speaker == conversation.get('tree', [{}])[0].get('speaker', ''))
            stance = determine_stance(node_text, is_op_response)
            
            # Create debate history context
            history_context = ''
            if history:
                recent_history = history[-2:]  # Last 2 exchanges
                for hist_speaker, hist_text in recent_history:
                    history_context += f"Agent {hist_speaker}: {hist_text[:100]}...\n"
            
            # Create training prompt similar to debate system
            prompt = f"""Claim: "{claim}"
Evidence: "{op_context}"

Your role is to {stance} the claim using the evidence and respond thoughtfully to your opponent's arguments.
Reference specific points where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences.

Debate so far:
{history_context}

Agent {stance[0].upper()}: """
            
            # The response is the actual text from the conversation
            response = node_text[:400]  # Truncate response
            
            training_pairs.append({
                'prompt': prompt,
                'response': response,
                'claim': claim,
                'stance': stance,
                'conversation_id': conversation.get('conversation_id', ''),
                'speaker': speaker
            })
            
            # Update history for children
            new_history = history + [(stance[0].upper(), node_text[:100])]
            
            # Process children
            for child in node.get('children', []):
                extract_from_tree(child, node_text, depth + 1, new_history)
        
        # Process the conversation tree
        tree = conversation.get('tree', [])
        for node in tree:
            extract_from_tree(node)
    
    return training_pairs

def preprocess_data(data_path: str, max_samples: int = None) -> Tuple[List[Dict], List[Dict]]:
    """Load and preprocess CMV data for debate response training"""
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    logger.info(f"Loaded {len(conversations)} conversations")
    
    # Limit conversations if max_samples specified
    if max_samples:
        conversations = conversations[:max_samples]
        logger.info(f"Limited to {len(conversations)} conversations")
    
    # Extract training pairs
    training_pairs = extract_debate_training_pairs(conversations)
    logger.info(f"Extracted {len(training_pairs)} debate prompt-response pairs")
    
    # Show examples
    if training_pairs:
        logger.info("\n" + "="*50)
        logger.info("TRAINING EXAMPLES:")
        logger.info("="*50)
        for i, pair in enumerate(training_pairs[:3]):
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"Claim: {pair['claim']}")
            logger.info(f"Stance: {pair['stance']}")
            logger.info(f"PROMPT:\n{pair['prompt']}")
            logger.info(f"TARGET:\n{pair['response']}")
            logger.info("-" * 30)
        logger.info("="*50)
    
    # Filter out very short responses
    training_pairs = [pair for pair in training_pairs if len(pair['response']) > 50]
    logger.info(f"After filtering: {len(training_pairs)} pairs")
    
    # Split into train/validation
    train_data, val_data = train_test_split(
        training_pairs, 
        test_size=0.2, 
        random_state=42
    )
    
    logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    return train_data, val_data

def fine_tune_model_with_lora(
    model_name: str = "google/flan-t5-large",
    data_path: str = "data/cmv_10_conversation_trees.json",
    output_dir: str = "models/flan-t5-large-cmv-debate-lora",
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    max_input_length: int = 512,
    max_target_length: int = 256,
    max_samples: int = None
):
    """Fine-tune FLAN-T5-Large on CMV debate data using LoRA"""
    
    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,  # Rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        target_modules=["q", "v", "k", "o", "wi", "wo"],  # T5 attention and FFN modules
        bias="none",
    )
    
    # Apply LoRA to model
    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess data
    train_data, val_data = preprocess_data(data_path, max_samples)
    
    # Create datasets
    train_dataset = DebateDataset(train_data, tokenizer, max_input_length, max_target_length)
    val_dataset = DebateDataset(val_data, tokenizer, max_input_length, max_target_length)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # Disable wandb
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Fine-tune
    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save LoRA adapter
    logger.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    return trainer, eval_results

def load_lora_model(base_model_name: str, lora_path: str):
    """Load base model with LoRA adapter"""
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer

def evaluate_debate_responses(model_path: str, test_data: List[Dict], base_model_name: str = "google/flan-t5-large"):
    """Evaluate fine-tuned LoRA model on debate response generation"""
    model, tokenizer = load_lora_model(base_model_name, model_path)
    model.eval()
    
    logger.info(f"Evaluating on {len(test_data)} test examples...")
    
    sample_results = []
    
    for i, item in enumerate(test_data[:10]):  # Sample first 10 for inspection
        prompt = item['prompt']
        true_response = item['response']
        
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=3,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from generated text
        if prompt in generated_response:
            generated_response = generated_response.replace(prompt, "").strip()
        
        sample_results.append({
            'prompt': prompt[:200] + "...",
            'generated': generated_response,
            'true': true_response[:200] + "...",
            'claim': item['claim'],
            'stance': item['stance']
        })
        
        if i < 3:  # Print first 3 examples
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"Claim: {item['claim']}")
            logger.info(f"Stance: {item['stance']}")
            logger.info(f"Generated: {generated_response}")
            logger.info(f"Original: {true_response}")
    
    return sample_results

def test_with_debate_system(lora_path: str, base_model_name: str = "google/flan-t5-large"):
    """Test the fine-tuned model with the debate system"""
    from debate_system import DebateSimulator
    
    # Load the fine-tuned model
    model, tokenizer = load_lora_model(base_model_name, lora_path)
    
    # Create a modified debate simulator that uses our fine-tuned model
    class FineTunedDebateSimulator(DebateSimulator):
        def __init__(self, model, tokenizer):
            self.tokenizer = tokenizer
            self.model = model
    
    simulator = FineTunedDebateSimulator(model, tokenizer)
    
    # Test with a sample claim and evidence
    test_claim = "Electric cars are better for the environment than gasoline cars"
    test_evidence = "Electric cars produce zero direct emissions, but electricity generation may still involve fossil fuels. However, the overall lifecycle emissions are typically lower."
    
    logger.info("Testing fine-tuned model with debate system...")
    logger.info(f"Claim: {test_claim}")
    logger.info(f"Evidence: {test_evidence}")
    
    debate_history = simulator.simulate_enhanced_debate(test_claim, test_evidence, rounds=2)
    
    logger.info("\n--- Debate Results ---")
    for speaker, argument in debate_history:
        logger.info(f"Agent {speaker}: {argument}")
    
    return debate_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5-Large for debate response generation using LoRA")
    
    # Data and model arguments
    parser.add_argument("--data_path", type=str, default="data/cmv_10_conversation_trees.json",
                        help="Path to CMV conversation data")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="models/flan-t5-large-cmv-debate-lora",
                        help="Directory to save fine-tuned model")
    
    # Training arguments
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of conversations to use for training (default: use all)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Maximum input sequence length")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length")
    
    # Evaluation arguments
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation after training")
    parser.add_argument("--skip_debate_test", action="store_true",
                        help="Skip debate system integration test")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune model with LoRA
    logger.info("Starting LoRA fine-tuning for debate response generation...")
    logger.info(f"Using {args.max_samples if args.max_samples else 'all'} conversations for training")
    
    trainer, eval_results = fine_tune_model_with_lora(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        max_samples=args.max_samples
    )
    
    logger.info("LoRA fine-tuning completed!")
    
    if not args.skip_evaluation:
        # Load test data for evaluation
        _, test_data = preprocess_data(args.data_path, args.max_samples)
        
        # Evaluate the fine-tuned model
        logger.info("Evaluating fine-tuned model...")
        sample_results = evaluate_debate_responses(args.output_dir, test_data[:20], args.model_name)
    
    if not args.skip_debate_test:
        # Test with debate system
        logger.info("Testing with debate system...")
        debate_results = test_with_debate_system(args.output_dir, args.model_name)
    
    logger.info("All tasks completed!")