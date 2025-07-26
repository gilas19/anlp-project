"""
Fine-tuning script for google/flan-t5-large on CMV conversation data
for generating better debate responses using LoRA (Low-Rank Adaptation)
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Tuple
import os
import re
import yaml
import argparse
import wandb

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
        input_text = item["prompt"]
        target_text = item["response"]

        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text, max_length=self.max_target_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": target_encoding.input_ids.squeeze(),
        }


def extract_debate_training_pairs(conversation_data: List[Dict]) -> List[Dict]:
    """
    Extract debate training pairs from CMV conversation trees using existing stance property.
    Uses text from delta-awarded users as evidence.
    """
    training_pairs = []

    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove reddit formatting
        text = re.sub(r"&gt;.*?\n", "", text)  # Remove quotes
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove links
        text = re.sub(r"\n+", " ", text)  # Replace newlines with spaces
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text.strip()

    def find_delta_awarded_users(tree_nodes: List[Dict]) -> Dict[str, str]:
        """Find users who were awarded deltas and their text"""
        delta_awarded_users = {}

        def search_tree(node):
            # Check if this is a DeltaBot confirmation
            if node.get("speaker") == "DeltaBot":
                text = node.get("text", "")
                # Extract username from DeltaBot confirmation
                import re

                match = re.search(r"delta awarded to /u/(\w+)", text)
                if match:
                    username = match.group(1)
                    # Find the user's text by looking for their node in the tree
                    user_text = find_user_text_in_tree(tree_nodes, username)
                    if user_text:
                        delta_awarded_users[username] = user_text

            # Recursively search children
            for child in node.get("children", []):
                search_tree(child)

        for node in tree_nodes:
            search_tree(node)

        return delta_awarded_users

    def find_user_text_in_tree(tree_nodes: List[Dict], username: str) -> str:
        """Find the text of a specific user in the conversation tree"""

        def search_for_user(node):
            if node.get("speaker") == username:
                text = node.get("text", "")
                if text and text != "[deleted]" and len(text.strip()) > 30:
                    return clean_text(text)

            # Search children
            for child in node.get("children", []):
                result = search_for_user(child)
                if result:
                    return result

            return None

        for node in tree_nodes:
            result = search_for_user(node)
            if result:
                return result

        return ""

    for conversation in conversation_data:
        title = conversation.get("title", "")
        op_text = conversation.get("op_text", "")

        # Extract claim from title
        claim = title.replace("CMV:", "").strip()
        if not claim or len(claim) < 10:
            continue

        # Find delta-awarded users and their evidence
        tree = conversation.get("tree", [])
        delta_awarded_users = find_delta_awarded_users(tree)

        # If no delta-awarded users, fall back to OP text
        if delta_awarded_users:
            # Use the first delta-awarded user's text as evidence
            evidence_text = list(delta_awarded_users.values())[0][:400]  # Truncate
        else:
            # Fallback to OP text if no deltas found
            evidence_text = clean_text(op_text)[:300]

        def extract_from_tree(node, parent_text="", depth=0, history=[]):
            if depth > 4 or not node:  # Limit depth
                return

            node_text = clean_text(node.get("text", ""))
            speaker = node.get("speaker", "")
            stance = node.get("stance", "")  # Use existing stance property

            if not node_text or node_text == "[deleted]" or len(node_text) < 20:
                return

            # Skip very short responses, system messages, or nodes without valid stance
            if len(node_text) < 30 or speaker in ["DeltaBot", "[deleted]"] or stance in ["NO_STANCE", ""]:
                # Still process children even if we skip this node
                for child in node.get("children", []):
                    extract_from_tree(child, node_text, depth + 1, history)
                return

            # Create debate history context
            history_context = ""
            if history:
                recent_history = history[-2:]  # Last 2 exchanges
                for hist_speaker, hist_text in recent_history:
                    history_context += f"Agent {hist_speaker}: {hist_text[:100]}...\n"

            # Create training prompt similar to debate system
            prompt = f"""
Claim: "{claim}"
Evidence: "{evidence_text}"

Your role is to {stance} the claim using the evidence and respond thoughtfully to your opponent's arguments.
Reference specific points where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences.

Debate so far:
{history_context}

Agent {stance[0].upper()}:
"""

            # The response is the actual text from the conversation
            response = node_text[:400]  # Truncate response

            training_pairs.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "claim": claim,
                    "stance": stance,
                    "conversation_id": conversation.get("conversation_id", ""),
                    "speaker": speaker,
                }
            )

            # Update history for children
            new_history = history + [(stance[0].upper(), node_text[:100])]

            # Process children
            for child in node.get("children", []):
                extract_from_tree(child, node_text, depth + 1, new_history)

        # Process the conversation tree
        tree = conversation.get("tree", [])
        for node in tree:
            extract_from_tree(node)

    return training_pairs


def preprocess_data(data_path: str, max_samples: int = None) -> Tuple[List[Dict], List[Dict]]:
    """Load and preprocess CMV data for debate response training"""
    logger.info(f"Loading data from {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
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
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING EXAMPLES:")
        logger.info("=" * 50)
        for i, pair in enumerate(training_pairs[:3]):
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(f"Claim: {pair['claim']}")
            logger.info(f"Stance: {pair['stance']}")
            logger.info(f"PROMPT:\n{pair['prompt']}")
            logger.info(f"TARGET:\n{pair['response']}")
            logger.info("-" * 30)
        logger.info("=" * 50)

    # Filter out very short responses
    training_pairs = [pair for pair in training_pairs if len(pair["response"]) > 50]
    logger.info(f"After filtering: {len(training_pairs)} pairs")

    # Split into train/validation
    train_data, val_data = train_test_split(training_pairs, test_size=0.2, random_state=42)

    logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    return train_data, val_data


def fine_tune_model_with_lora(config: Dict):
    """Fine-tune FLAN-T5-Large on CMV debate data using LoRA"""

    # Extract config values
    model_name = config["model"]["name"]
    data_path = config["data"]["path"]
    output_dir = config["model"]["output_dir"]
    batch_size = config["training"]["batch_size"]
    learning_rate = float(config["training"]["learning_rate"])
    num_epochs = config["training"]["num_epochs"]
    max_input_length = config["data"]["max_input_length"]
    max_target_length = config["data"]["max_target_length"]
    max_samples = config["data"]["max_samples"]

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=f"{output_dir}/logs",
        logging_steps=config["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["evaluation"]["eval_steps"],
        save_steps=config["evaluation"]["save_steps"],
        save_total_limit=config["evaluation"]["save_total_limit"],
        learning_rate=learning_rate,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    return model, tokenizer


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    config_path = os.path.join("config", "finetune", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5-Large with LoRA on CMV debate data")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    wandb.init(project="debate-response-generation", config=config, name=f"lora-finetune-{config['model']['name']}", reinit=True)

    # Create output directory
    os.makedirs(config["model"]["output_dir"], exist_ok=True)

    # Fine-tune model with LoRA
    logger.info("Starting LoRA fine-tuning for debate response generation...")
    logger.info(f"Using {config['data']['max_samples'] if config['data']['max_samples'] else 'all'} conversations for training")

    trainer, eval_results = fine_tune_model_with_lora(config)

    logger.info("LoRA fine-tuning completed!")
    logger.info(f"Evaluation results: {eval_results}")

    wandb.finish()


if __name__ == "__main__":
    main()
