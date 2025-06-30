#!/usr/bin/env python3
"""
Evaluation pipeline comparing debate-enhanced vs baseline fact verification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from debate_system import DebateSimulator, BaselineClassifier
from fever_loader import FEVERDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Comprehensive evaluation comparing debate vs baseline approaches"""
    
    def __init__(self, model_name="google/flan-t5-large"):
        self.debate_sim = DebateSimulator(model_name)
        self.baseline = BaselineClassifier(model_name)
        self.fever_loader = FEVERDataLoader()
        
    def evaluate_single_example(self, claim: str, evidence: str, true_label: str) -> Dict:
        """Evaluate single example with both methods"""
        
        # Debate approach
        start_time = time.time()
        debate_history = self.debate_sim.simulate_enhanced_debate(claim, evidence)
        judgment_prompt = self.debate_sim.create_judgment_prompt(claim, evidence, debate_history)
        debate_pred = self.debate_sim.generate_response(judgment_prompt).strip().upper()
        debate_time = time.time() - start_time
        
        # Clean debate prediction
        if "SUPPORTS" in debate_pred:
            debate_pred = "SUPPORTS"
        elif "REFUTES" in debate_pred:
            debate_pred = "REFUTES"
        else:
            debate_pred = "NOT ENOUGH INFO"
        
        # Baseline approach
        start_time = time.time()
        baseline_pred = self.baseline.classify_direct(claim, evidence)
        baseline_time = time.time() - start_time
        
        return {
            'claim': claim,
            'evidence': evidence[:100] + "...",
            'true_label': true_label,
            'debate_pred': debate_pred,
            'baseline_pred': baseline_pred,
            'debate_correct': debate_pred == true_label,
            'baseline_correct': baseline_pred == true_label,
            'debate_time': debate_time,
            'baseline_time': baseline_time,
            'debate_history': debate_history
        }
    
    def run_evaluation(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Run full evaluation on dataset"""
        
        results = []
        total_examples = len(eval_df)
        
        logger.info(f"Starting evaluation on {total_examples} examples...")
        
        for idx, row in eval_df.iterrows():
            logger.info(f"Processing example {idx + 1}/{total_examples}")
            
            try:
                result = self.evaluate_single_example(
                    row['claim'], 
                    row['evidence_text'], 
                    row['label']
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing example {idx}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        logger.info("Evaluation completed")
        
        return results_df
    
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Compute comprehensive metrics"""
        
        # Accuracy
        debate_accuracy = results_df['debate_correct'].mean()
        baseline_accuracy = results_df['baseline_correct'].mean()
        
        # Per-class metrics
        labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
        
        debate_report = classification_report(
            results_df['true_label'], 
            results_df['debate_pred'], 
            labels=labels,
            output_dict=True
        )
        
        baseline_report = classification_report(
            results_df['true_label'], 
            results_df['baseline_pred'], 
            labels=labels,
            output_dict=True
        )
        
        # Timing
        avg_debate_time = results_df['debate_time'].mean()
        avg_baseline_time = results_df['baseline_time'].mean()
        
        # Agreement analysis
        agreement = (results_df['debate_pred'] == results_df['baseline_pred']).mean()
        
        metrics = {
            'debate_accuracy': debate_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_improvement': debate_accuracy - baseline_accuracy,
            'debate_report': debate_report,
            'baseline_report': baseline_report,
            'avg_debate_time': avg_debate_time,
            'avg_baseline_time': avg_baseline_time,
            'time_ratio': avg_debate_time / avg_baseline_time,
            'agreement_rate': agreement,
            'total_examples': len(results_df)
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print formatted results"""
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nAccuracy Comparison:")
        print(f"  Debate Method:    {metrics['debate_accuracy']:.3f}")
        print(f"  Baseline Method:  {metrics['baseline_accuracy']:.3f}")
        print(f"  Improvement:      {metrics['accuracy_improvement']:+.3f}")
        
        print(f"\nTiming Analysis:")
        print(f"  Avg Debate Time:  {metrics['avg_debate_time']:.2f}s")
        print(f"  Avg Baseline Time: {metrics['avg_baseline_time']:.2f}s")
        print(f"  Time Ratio:       {metrics['time_ratio']:.1f}x slower")
        
        print(f"\nAgreement Rate: {metrics['agreement_rate']:.3f}")
        print(f"Total Examples: {metrics['total_examples']}")
        
    def analyze_disagreements(self, results_df: pd.DataFrame):
        """Analyze cases where methods disagree"""
        
        disagreements = results_df[results_df['debate_pred'] != results_df['baseline_pred']]
        
        print(f"\n📊 Disagreement Analysis ({len(disagreements)} cases):")
        
        # Where debate is correct but baseline is wrong
        debate_better = disagreements[
            (disagreements['debate_correct'] == True) & 
            (disagreements['baseline_correct'] == False)
        ]
        
        # Where baseline is correct but debate is wrong  
        baseline_better = disagreements[
            (disagreements['debate_correct'] == False) & 
            (disagreements['baseline_correct'] == True)
        ]
        
        print(f"  Debate better: {len(debate_better)} cases")
        print(f"  Baseline better: {len(baseline_better)} cases")
        
        if len(debate_better) > 0:
            print(f"\n✅ Examples where debate improved prediction:")
            for _, row in debate_better.head(3).iterrows():
                print(f"  Claim: {row['claim'][:80]}...")
                print(f"  True: {row['true_label']}, Debate: {row['debate_pred']}, Baseline: {row['baseline_pred']}")
                print("")

def main():
    """Run evaluation pipeline"""
    
    # Load evaluation data
    loader = FEVERDataLoader()
    eval_df = loader.get_evaluation_data(sample_size=60)  # Small sample for testing
    
    if eval_df is None:
        print("Failed to load evaluation data")
        return
    
    # Run evaluation
    pipeline = EvaluationPipeline()
    results_df = pipeline.run_evaluation(eval_df)
    
    # Compute and display metrics
    metrics = pipeline.compute_metrics(results_df)
    pipeline.print_results(metrics)
    pipeline.analyze_disagreements(results_df)
    
    # Save results
    results_df.to_csv('evaluation_results.csv', index=False)
    logger.info("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()