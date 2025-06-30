#!/usr/bin/env python3
"""
Main experiment runner for the fact verification debate study.
"""

import argparse
import logging
import json
import os
from datetime import datetime
from evaluation_pipeline import EvaluationPipeline
from fever_loader import FEVERDataLoader

# Set up results directory path
project_root = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(project_root, 'results')
os.makedirs(results_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(results_dir, 'experiment.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(sample_size=150, model_name="google/flan-t5-large"):
    """Run the main experiment comparing debate vs baseline approaches"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting experiment at {timestamp}")
    logger.info(f"Sample size: {sample_size}, Model: {model_name}")
    
    # Load data
    logger.info("Loading FEVER evaluation data...")
    loader = FEVERDataLoader()
    eval_df = loader.get_evaluation_data(sample_size=sample_size)
    
    if eval_df is None:
        logger.error("Failed to load evaluation data")
        return None
    
    logger.info(f"Loaded {len(eval_df)} examples")
    print(f"Label distribution:\n{eval_df['label'].value_counts()}")
    
    # Run evaluation
    logger.info("Initializing evaluation pipeline...")
    pipeline = EvaluationPipeline(model_name)
    
    logger.info("Running evaluation (this may take a while)...")
    results_df = pipeline.run_evaluation(eval_df)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = pipeline.compute_metrics(results_df)
    
    # Display results
    pipeline.print_results(metrics)
    pipeline.analyze_disagreements(results_df)
    
    # Save results
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")
    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.json")
    
    results_df.to_csv(results_file, index=False)
    
    # Save metrics (remove non-serializable objects)
    metrics_clean = {k: v for k, v in metrics.items() 
                    if k not in ['debate_report', 'baseline_report']}
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Metrics saved to {metrics_file}")
    
    return results_df, metrics

def main():
    parser = argparse.ArgumentParser(description="Run fact verification debate experiments")
    parser.add_argument("--sample-size", type=int, default=150, 
                       help="Number of examples to evaluate (default: 150)")
    parser.add_argument("--model", type=str, default="google/flan-t5-large",
                       help="Model to use (default: google/flan-t5-large)")
    
    args = parser.parse_args()
    
    try:
        result = run_experiment(args.sample_size, args.model)
        
        if result is not None:
            results_df, metrics = result
            print(f"\n🎉 Experiment completed successfully!")
            print(f"   Final accuracy improvement: {metrics['accuracy_improvement']:+.3f}")
            print(f"   Debate accuracy: {metrics['debate_accuracy']:.3f}")
            print(f"   Baseline accuracy: {metrics['baseline_accuracy']:.3f}")
        else:
            print("❌ Experiment failed - could not load data or run evaluation")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()