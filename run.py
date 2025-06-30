#!/usr/bin/env python3
"""
Simple runner script for the fact verification debate experiments.
Handles the organized project structure.
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run fact verification debate experiments")
    parser.add_argument("--sample-size", type=int, default=6, 
                       help="Number of examples to evaluate (default: 6)")
    parser.add_argument("--model", type=str, default="google/flan-t5-large",
                       help="Model to use (default: google/flan-t5-large)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Fact Verification Debate Experiment")
    print(f"   Sample size: {args.sample_size}")
    print(f"   Model: {args.model}")
    print()
    
    try:
        # Import the experiment runner
        from run_experiments import run_experiment
        
        # Update data path for new structure
        import fever_loader
        fever_loader.FEVERDataLoader.__init__ = lambda self: setattr(self, 'splits', {
            'train': 'train.jsonl',
            'validation': 'valid.jsonl', 
            'test': 'test.jsonl'
        })
        
        # Run the experiment
        result = run_experiment(args.sample_size, args.model)
        
        if result is not None:
            results_df, metrics = result
            print(f"\n🎉 Experiment completed successfully!")
            print(f"   Final accuracy improvement: {metrics['accuracy_improvement']:+.3f}")
            print(f"   Debate accuracy: {metrics['debate_accuracy']:.3f}")
            print(f"   Baseline accuracy: {metrics['baseline_accuracy']:.3f}")
            print(f"\n📁 Results saved in results/ directory")
        else:
            print("❌ Experiment failed - could not load data or run evaluation")
            return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
