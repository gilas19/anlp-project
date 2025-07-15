#!/usr/bin/env python3
"""
Aggregate results from multiple experiments for comparison
"""

import json
import pandas as pd
import glob
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(result_dir: str) -> pd.DataFrame:
    """Load all JSON result files from a directory"""
    all_files = glob.glob(os.path.join(result_dir, "*.json"))
    dfs = []

    for file in all_files:
        with open(file, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)

            # Extract config info
            config_info = df.iloc[0]['config']
            model_type = df.iloc[0]['model_type']

            df['experiment_dimension'] = config_info['dimension']
            df['experiment_value'] = config_info['value']
            df['model_type'] = model_type

            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def compare_models(baseline_dir: str, fine_tuned_dir: str) -> Dict:
    """Compare baseline and fine-tuned model performance"""
    baseline_df = load_results(baseline_dir)
    fine_tuned_df = load_results(fine_tuned_dir)

    if baseline_df.empty or fine_tuned_df.empty:
        return {}

    comparison = {}

    # Compare overall accuracy
    comparison['overall_accuracy'] = {
        'baseline': baseline_df['correct'].mean(),
        'fine_tuned': fine_tuned_df['correct'].mean()
    }

    # Compare by dimension
    for dimension in baseline_df['experiment_dimension'].unique():
        dim_df = baseline_df[baseline_df['experiment_dimension'] == dimension]
        comparison[dimension] = {
            'baseline': dim_df.groupby('experiment_value')['correct'].mean().to_dict(),
            'fine_tuned': fine_tuned_df[
                fine_tuned_df['experiment_dimension'] == dimension
                ].groupby('experiment_value')['correct'].mean().to_dict()
        }

    return comparison


def generate_plots(comparison: Dict, output_dir: str):
    """Generate comparison plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Overall accuracy plot
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=list(comparison['overall_accuracy'].keys()),
        y=list(comparison['overall_accuracy'].values())
    )
    plt.title("Overall Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(output_dir, "overall_accuracy.png"))
    plt.close()

    # Dimension-specific plots
    for dimension in comparison:
        if dimension == 'overall_accuracy':
            continue

        plt.figure(figsize=(10, 6))
        df = pd.DataFrame({
            'value': list(comparison[dimension]['baseline'].keys()),
            'baseline': list(comparison[dimension]['baseline'].values()),
            'fine_tuned': list(comparison[dimension]['fine_tuned'].values())
        })

        df = df.melt(id_vars='value', var_name='model', value_name='accuracy')

        sns.barplot(data=df, x='value', y='accuracy', hue='model')
        plt.title(f"Accuracy by {dimension.replace('_', ' ').title()}")
        plt.xlabel(dimension.replace('_', ' '))
        plt.ylabel("Accuracy")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dimension}_comparison.png"))
        plt.close()


if __name__ == "__main__":
    baseline_dir = "results/baseline"
    fine_tuned_dir = "results/fine_tuned"
    output_dir = "analysis/plots"

    comparison = compare_models(baseline_dir, fine_tuned_dir)

    if comparison:
        print("\n=== Comparison Results ===")
        print("Overall Accuracy:")
        print(f"Baseline: {comparison['overall_accuracy']['baseline']:.2f}")
        print(f"Fine-tuned: {comparison['overall_accuracy']['fine_tuned']:.2f}")

        for dim in comparison:
            if dim != 'overall_accuracy':
                print(f"\nDimension: {dim}")
                print("Baseline:")
                for val, acc in comparison[dim]['baseline'].items():
                    print(f"  {val}: {acc:.2f}")
                print("Fine-tuned:")
                for val, acc in comparison[dim]['fine_tuned'].items():
                    print(f"  {val}: {acc:.2f}")

        generate_plots(comparison, output_dir)
        print("\nGenerated comparison plots in analysis/plots")
    else:
        print("No results found for comparison")