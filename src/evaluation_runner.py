#!/usr/bin/env python3
"""
Main experiment runner for comparing baseline and fine-tuned models
"""
"""
python evaluation_runner.py --config configs/baseline/config_debate.yaml
python evaluation_runner.py --config configs/baseline/config_dialogs_1.yaml
python evaluation_runner.py --config configs/baseline/config_dialogs_2.yaml
"""

import yaml
import json
import time
import torch
import pandas as pd
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from fever_loader import FEVERDataLoader
from datetime import datetime
import logging
import os
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation across different configurations"""

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.fever_loader = FEVERDataLoader()

    def load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_model(self):
        """Load either baseline or fine-tuned model"""
        model_name = self.config['model']['name']

        logger.info(f"Loading {self.config['model']['type']} model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.config['model']['type'] == 'baseline':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None
            )
        else:  # fine-tuned
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                "/content/anlp-project/models/flan-t5-large-cmv-debate-lora"
            )

        self.model.eval()

    def generate_response(self, prompt: str) -> Tuple[str, float]:
        """Generate response with timing measurement"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=3,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response_time = time.time() - start_time

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_length = len(response.split())

        return response, response_time, response_length

    def simulate_debate(self, claim: str, evidence: str, num_rounds: int = 3) -> Dict:
        """Simulate debate between two agents"""
        debate_history = []
        metrics = {
            'response_times': [],
            'response_lengths': [],
            'agent_1_responses': 0,
            'agent_2_responses': 0
        }

        # Initial prompts for both agents
        agent1_prompt = f"""
Claim: "{claim}"
Evidence: "{evidence}"

Your role is to SUPPORT the claim using the evidence. 
Reference specific points where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences.
"""
        agent2_prompt = f"""
Claim: "{claim}"
Evidence: "{evidence}"

Your role is to REFUTE the claim using the evidence. 
Reference specific points where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences.
"""

        # Alternate between agents
        current_agent = 1 if self.config['debate'].get('initiator', 1) == 1 else 2

        for round_num in range(num_rounds):
            if current_agent == 1:
                prompt = agent1_prompt
                if debate_history:
                    prompt += "\nPrevious arguments:\n" + "\n".join(debate_history[-2:])
            else:
                prompt = agent2_prompt
                if debate_history:
                    prompt += "\nPrevious arguments:\n" + "\n".join(debate_history[-2:])

            response, time_taken, length = self.generate_response(prompt)
            debate_history.append(f"Agent {current_agent}: {response}")

            # Record metrics
            metrics['response_times'].append(time_taken)
            metrics['response_lengths'].append(length)
            if current_agent == 1:
                metrics['agent_1_responses'] += 1
            else:
                metrics['agent_2_responses'] += 1

            # Switch agents
            current_agent = 2 if current_agent == 1 else 1

        return {
            'history': debate_history,
            'metrics': metrics,
            'final_response': debate_history[-1]
        }

    def evaluate_claim(self, claim: str, evidence: str, label: str) -> Dict:
        """Evaluate a single claim with configured parameters"""
        result = {
            'claim': claim,
            'evidence': evidence,
            'label': label,
            'config': self.config['experiment'],
            'model_type': self.config['model']['type']
        }

        if self.config['experiment']['dimension'] == 'debate':
            # Debate mode evaluation
            debate_result = self.simulate_debate(
                claim,
                evidence,
                num_rounds=self.config['debate']['rounds']
            )

            result.update({
                'method': 'debate',
                'num_rounds': self.config['debate']['rounds'],
                'initiator': self.config['debate'].get('initiator', 1),
                'debate_history': debate_result['history'],
                'response': debate_result['final_response'],
                'response_time': np.mean(debate_result['metrics']['response_times']),
                'response_length': np.mean(debate_result['metrics']['response_lengths']),
                'agent_1_responses': debate_result['metrics']['agent_1_responses'],
                'agent_2_responses': debate_result['metrics']['agent_2_responses']
            })
        else:
            # Direct prediction mode
            prompt = f"""
Claim: "{claim}"
Evidence: "{evidence}"

Based on the evidence, does the claim SUPPORT, REFUTE, or is there NOT ENOUGH INFO?
Answer with just one of: SUPPORTS, REFUTES, NOT ENOUGH INFO
"""
            response, time_taken, length = self.generate_response(prompt)

            result.update({
                'method': 'direct',
                'response': response,
                'response_time': time_taken,
                'response_length': length
            })

        # Determine accuracy
        predicted_label = result['response'].strip().upper()
        if any(x in predicted_label for x in ['SUPPORTS', 'SUPPORT']):
            predicted_label = 'SUPPORTS'
        elif any(x in predicted_label for x in ['REFUTES', 'REFUTE']):
            predicted_label = 'REFUTES'
        else:
            predicted_label = 'NOT ENOUGH INFO'

        result['predicted_label'] = predicted_label
        result['correct'] = predicted_label == label

        return result

    def run_evaluation(self):
        """Run full evaluation based on config"""
        self.load_model()

        # Load evaluation data
        eval_data = self.fever_loader.get_evaluation_data(
            split='validation',
            sample_size=self.config['data'].get('sample_size', 50)
        )

        if eval_data is None:
            logger.error("Failed to load evaluation data")
            return None

        results = []
        for _, row in tqdm(eval_data.iterrows(), total=len(eval_data)):
            result = self.evaluate_claim(row['claim'], row['evidence_text'], row['label'])
            results.append(result)

        # Save results
        output_dir = self.config['output']['dir']
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir,
            f"{self.config['output']['name']}_{timestamp}.json"
        )

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {output_file}")

        # Calculate and log summary metrics
        df = pd.DataFrame(results)
        accuracy = df['correct'].mean()
        avg_time = df['response_time'].mean()
        avg_length = df['response_length'].mean()

        logger.info("\n=== Summary Metrics ===")
        logger.info(f"Accuracy: {accuracy:.2f}")
        logger.info(f"Avg Response Time: {avg_time:.2f}s")
        logger.info(f"Avg Response Length: {avg_length:.2f} tokens")

        if 'method' in df.columns:
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                logger.info(f"\nMethod: {method}")
                logger.info(f"Accuracy: {method_df['correct'].mean():.2f}")
                logger.info(f"Avg Time: {method_df['response_time'].mean():.2f}s")
                logger.info(f"Avg Length: {method_df['response_length'].mean():.2f} tokens")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.config)
    evaluator.run_evaluation()