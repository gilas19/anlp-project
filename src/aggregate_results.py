import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

# Get the parent directory of src (anlp-project)
PROJECT_ROOT = Path(__file__).parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    """Load results from JSON files (now organized by model type)"""
    all_results = []

    # Process both baseline and fine_tuned directories
    for model_type in ['baseline', 'fine_tuned']:
        model_dir = RESULTS_DIR / model_type
        if not model_dir.exists():
            continue

        for result_file in model_dir.glob('*.json'):
            with open(result_file, 'r') as f:
                data = json.load(f)

                # Extract configuration and metrics
                config = data['config']
                stats = data['stats']

                all_results.append({
                    "config_id": config['config_id'],
                    "model": model_type,
                    "with_debate": config['with_debate'],
                    "initiator": config.get('initiator'),
                    "num_turns": config.get('num_turns', 0),
                    "accuracy": stats['accuracy'],
                    "avg_length": stats['avg_length'],
                    "avg_time": stats['avg_time'],
                    "num_claims": len(data['results'])  # Track how many claims were processed
                })

    return pd.DataFrame(all_results)


def generate_plots(df):
    """Generate all analysis plots with improved formatting"""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Main Accuracy Comparison Plot
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        x="config_id",
        y="accuracy",
        hue="model",
        data=df.sort_values(['model', 'with_debate', 'num_turns']),
        palette=['#1f77b4', '#ff7f0e']  # Blue for baseline, orange for fine-tuned
    )
    plt.title("Fact Verification Accuracy by Configuration", pad=20, fontsize=14)
    plt.xlabel("Configuration")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode='anchor'
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_all_configs.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2-4. Debate-Specific Plots
    debate_df = df[df["with_debate"]].copy()
    if not debate_df.empty:
        # Convert num_turns to categorical for better plotting
        debate_df['num_turns'] = debate_df['num_turns'].astype(str) + " turns"

        # 2. Accuracy by Turns
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x="num_turns",
            y="accuracy",
            hue="model",
            style="initiator",
            markers=True,
            dashes=False,
            data=debate_df,
            linewidth=2.5,
            markersize=10,
            palette=['#1f77b4', '#ff7f0e']
        )
        plt.title("Debate Accuracy by Number of Turns")
        plt.xlabel("Number of Turns per Agent")
        plt.ylabel("Accuracy")
        plt.legend(title="Model/Initiator", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "accuracy_vs_turns.png", dpi=300)
        plt.close()

        # 3. Response Length Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="num_turns",
            y="avg_length",
            hue="model",
            data=debate_df,
            palette=['#1f77b4', '#ff7f0e']
        )
        plt.title("Response Length Distribution by Turns")
        plt.xlabel("Number of Turns per Agent")
        plt.ylabel("Average Response Length (tokens)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "length_vs_turns.png", dpi=300)
        plt.close()

        # 4. Time vs Accuracy
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="avg_time",
            y="accuracy",
            hue="model",
            style="initiator",
            data=debate_df,
            s=150,
            palette=['#1f77b4', '#ff7f0e']
        )
        plt.title("Response Time vs Accuracy")
        plt.xlabel("Average Response Time (seconds)")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "time_vs_accuracy.png", dpi=300)
        plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_df = df[["num_turns", "accuracy", "avg_length", "avg_time"]].corr()
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=.5
    )
    plt.title("Metric Correlations", pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=300)
    plt.close()


def calculate_correlations(df):
    """Enhanced correlation analysis with more metrics"""
    debate_df = df[df["with_debate"]]

    if debate_df.empty:
        print("No debate configurations found for correlation analysis")
        return

    print("\nEnhanced Correlation Analysis:")
    print("=" * 50)

    metrics = [
        ("Turns vs Accuracy", "num_turns", "accuracy"),
        ("Turns vs Length", "num_turns", "avg_length"),
        ("Turns vs Time", "num_turns", "avg_time"),
        ("Length vs Accuracy", "avg_length", "accuracy"),
        ("Time vs Accuracy", "avg_time", "accuracy"),
        ("Length vs Time", "avg_length", "avg_time")
    ]

    for name, x, y in metrics:
        r, p = pearsonr(debate_df[x], debate_df[y])
        stars = "*" * min(3, int(-np.log10(p)))  # Add significance stars
        print(f"{name:<20}: r = {r:.3f}{stars}, p = {p:.3f}")


def main():
    df = load_results()

    if df.empty:
        print("No results found - please run experiments first")
        return

    # Save raw and processed data
    df.to_csv(PLOTS_DIR / "all_results.csv", index=False)

    # Generate analysis
    generate_plots(df)
    calculate_correlations(df)

    # Save summary stats
    summary = df.groupby(['model', 'with_debate']).agg({
        'accuracy': ['mean', 'std'],
        'avg_length': 'mean',
        'avg_time': 'mean',
        'num_claims': 'sum'
    })
    summary.to_csv(PLOTS_DIR / "experiment_summary.csv")
    print("\nAnalysis complete - results saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()