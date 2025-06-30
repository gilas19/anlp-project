#!/usr/bin/env python3
"""
Fine-tuning script for Flan-T5-large on Reddit ChangeMyView dataset
for improved argumentative reasoning in fact verification debates.
"""

import pandas as pd
import torch
import os
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
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess CMV dataset"""
        # Handle relative path from project root
        if not os.path.isabs(file_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            file_path = os.path.join(project_root, 'data', file_path)
        
        df = pd.read_csv(file_path)
        df = df.dropna()
        logger.info(f"Loaded {len(df)} argument pairs")
        return df
    
    def create_training_prompts(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Create training prompts for argumentative reasoning"""
        prompts = []
        
        for _, row in df.iterrows():
            claim = row['claim'].strip()
            reply = row['reply'].strip()
            
            # Create argumentative reasoning prompt
            input_text = f"Generate a thoughtful argument in response to: {claim}"
            target_text = reply
            
            prompts.append({
                'input_text': input_text,
                'target_text': target_text
            })
        
        return prompts
    
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
    data_path: str = "cmv_argument_pairs_unique_claims.csv",
    output_dir: str = "./finetuned_model",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """Fine-tune Flan-T5 on CMV dataset"""
    
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Process data
    processor = CMVDataProcessor(tokenizer)
    df = processor.load_data(data_path)
    
    # Create training data (use subset for faster training)
    training_data = processor.create_training_prompts(df.head(1000))
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
    fine_tune_model()