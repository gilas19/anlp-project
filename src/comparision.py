import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentAnalyzer:
    def __init__(self, fine_tuned_results_path, baseline_results_path, fever_data_path):
        """
        Initialize the analyzer with paths to results and FEVER data.

        Args:
            fine_tuned_results_path: Path to JSON file with fine-tuned model results
            baseline_results_path: Path to JSON file with baseline model results
            fever_data_path: Path to FEVER dataset file
        """
        self.fine_tuned_results = self.load_results(fine_tuned_results_path)
        self.baseline_results = self.load_results(baseline_results_path)
        self.fever_data = self.load_fever_data(fever_data_path)

    def load_results(self, file_path):
        """Load experiment results from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_fever_data(self, file_path):
        """Load FEVER dataset."""
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]

    def compare_num_dialogs(self):
        """
        Compare accuracy for different numbers of dialogs (1, 2, 6, 12).
        Returns accuracy for both fine-tuned and baseline models.
        """
        dialog_counts = [1, 2, 6, 12]
        results = {'fine_tuned': {}, 'baseline': {}}

        for count in dialog_counts:
            # Filter results for this dialog count
            ft_filtered = [r for r in self.fine_tuned_results if len(r['dialogs']) == count]
            bl_filtered = [r for r in self.baseline_results if len(r['dialogs']) == count]

            # Calculate accuracy
            ft_accuracy = self.calculate_accuracy(ft_filtered)
            bl_accuracy = self.calculate_accuracy(bl_filtered)

            results['fine_tuned'][count] = ft_accuracy
            results['baseline'][count] = bl_accuracy

        return results

    def calculate_accuracy(self, results):
        """Calculate accuracy for a set of results."""
        correct = 0
        total = 0

        for res in results:
            claim_id = res['claim_id']
            predicted_label = res['final_prediction']
            true_label = self.fever_data[claim_id]['label']

            if predicted_label.lower() == true_label.lower():
                correct += 1
            total += 1

        return correct / total if total > 0 else 0

    def compare_agent_initiation(self):
        """
        Analyze differences between experiments initiated by Agent 1 vs Agent 2.
        Returns accuracy and average response length for each case.
        """
        analysis = {
            'fine_tuned': {'agent1': {'accuracy': 0, 'response_length': 0, 'count': 0},
                           'agent2': {'accuracy': 0, 'response_length': 0, 'count': 0}},
            'baseline': {'agent1': {'accuracy': 0, 'response_length': 0, 'count': 0},
                         'agent2': {'accuracy': 0, 'response_length': 0, 'count': 0}}
        }

        # Process fine-tuned results
        for res in self.fine_tuned_results:
            claim_id = res['claim_id']
            initiator = 'agent1' if res['initiator'] == 'agent1' else 'agent2'
            true_label = self.fever_data[claim_id]['label']

            # Update accuracy
            if res['final_prediction'].lower() == true_label.lower():
                analysis['fine_tuned'][initiator]['accuracy'] += 1

            # Update response length (average length of all responses)
            total_length = sum(len(d['response']) for d in res['dialogs'])
            avg_length = total_length / len(res['dialogs']) if len(res['dialogs']) > 0 else 0
            analysis['fine_tuned'][initiator]['response_length'] += avg_length

            analysis['fine_tuned'][initiator]['count'] += 1

        # Process baseline results
        for res in self.baseline_results:
            claim_id = res['claim_id']
            initiator = 'agent1' if res['initiator'] == 'agent1' else 'agent2'
            true_label = self.fever_data[claim_id]['label']

            # Update accuracy
            if res['final_prediction'].lower() == true_label.lower():
                analysis['baseline'][initiator]['accuracy'] += 1

            # Update response length
            total_length = sum(len(d['response']) for d in res['dialogs'])
            avg_length = total_length / len(res['dialogs']) if len(res['dialogs']) > 0 else 0
            analysis['baseline'][initiator]['response_length'] += avg_length

            analysis['baseline'][initiator]['count'] += 1

        # Calculate averages
        for model in ['fine_tuned', 'baseline']:
            for agent in ['agent1', 'agent2']:
                if analysis[model][agent]['count'] > 0:
                    analysis[model][agent]['accuracy'] /= analysis[model][agent]['count']
                    analysis[model][agent]['response_length'] /= analysis[model][agent]['count']

        return analysis

    def compare_response_length(self):
        """
        Compare the length of responses between fine-tuned and baseline models.
        Returns average response length for both models.
        """
        ft_lengths = []
        bl_lengths = []

        for res in self.fine_tuned_results:
            lengths = [len(d['response']) for d in res['dialogs']]
            ft_lengths.extend(lengths)

        for res in self.baseline_results:
            lengths = [len(d['response']) for d in res['dialogs']]
            bl_lengths.extend(lengths)

        return {
            'fine_tuned_avg_length': np.mean(ft_lengths) if ft_lengths else 0,
            'baseline_avg_length': np.mean(bl_lengths) if bl_lengths else 0,
            'fine_tuned_median_length': np.median(ft_lengths) if ft_lengths else 0,
            'baseline_median_length': np.median(bl_lengths) if bl_lengths else 0
        }

    def compare_response_timing(self):
        """
        Compare the timing of responses between fine-tuned and baseline models.
        Returns average response time for both models.
        """
        ft_times = []
        bl_times = []

        for res in self.fine_tuned_results:
            times = [d['response_time'] for d in res['dialogs'] if 'response_time' in d]
            ft_times.extend(times)

        for res in self.baseline_results:
            times = [d['response_time'] for d in res['dialogs'] if 'response_time' in d]
            bl_times.extend(times)

        return {
            'fine_tuned_avg_time': np.mean(ft_times) if ft_times else 0,
            'baseline_avg_time': np.mean(bl_times) if bl_times else 0,
            'fine_tuned_median_time': np.median(ft_times) if ft_times else 0,
            'baseline_median_time': np.median(bl_times) if bl_times else 0
        }

    def analyze_correlations(self):
        """
        Analyze correlations between different metrics.
        Returns correlation coefficients between:
        - Accuracy and number of dialogs
        - Response length and accuracy
        - Response time and accuracy
        """
        # Prepare data for correlation analysis
        data = []

        for res in self.fine_tuned_results + self.baseline_results:
            claim_id = res['claim_id']
            true_label = self.fever_data[claim_id]['label']
            correct = 1 if res['final_prediction'].lower() == true_label.lower() else 0
            num_dialogs = len(res['dialogs'])
            avg_length = sum(len(d['response']) for d in res['dialogs']) / num_dialogs if num_dialogs > 0 else 0
            avg_time = sum(d['response_time'] for d in res['dialogs'] if
                           'response_time' in d) / num_dialogs if num_dialogs > 0 else 0

            data.append({
                'correct': correct,
                'num_dialogs': num_dialogs,
                'avg_length': avg_length,
                'avg_time': avg_time,
                'model': 'fine_tuned' if res in self.fine_tuned_results else 'baseline'
            })

        df = pd.DataFrame(data)

        # Calculate correlations
        correlations = {
            'num_dialogs_vs_accuracy': {
                'fine_tuned': pearsonr(
                    df[df['model'] == 'fine_tuned']['num_dialogs'],
                    df[df['model'] == 'fine_tuned']['correct']
                )[0],
                'baseline': pearsonr(
                    df[df['model'] == 'baseline']['num_dialogs'],
                    df[df['model'] == 'baseline']['correct']
                )[0]
            },
            'response_length_vs_accuracy': {
                'fine_tuned': pearsonr(
                    df[df['model'] == 'fine_tuned']['avg_length'],
                    df[df['model'] == 'fine_tuned']['correct']
                )[0],
                'baseline': pearsonr(
                    df[df['model'] == 'baseline']['avg_length'],
                    df[df['model'] == 'baseline']['correct']
                )[0]
            },
            'response_time_vs_accuracy': {
                'fine_tuned': pearsonr(
                    df[df['model'] == 'fine_tuned']['avg_time'],
                    df[df['model'] == 'fine_tuned']['correct']
                )[0],
                'baseline': pearsonr(
                    df[df['model'] == 'baseline']['avg_time'],
                    df[df['model'] == 'baseline']['correct']
                )[0]
            }
        }

        return correlations

    def generate_report(self, output_path):
        """Generate a comprehensive report of all analyses."""
        report = {
            'num_dialogs_comparison': self.compare_num_dialogs(),
            'agent_initiation_comparison': self.compare_agent_initiation(),
            'response_length_comparison': self.compare_response_length(),
            'response_timing_comparison': self.compare_response_timing(),
            'correlation_analysis': self.analyze_correlations()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def visualize_results(self, report):
        """Generate visualizations of the analysis results."""
        # Number of dialogs comparison
        dialog_data = report['num_dialogs_comparison']
        counts = list(dialog_data['fine_tuned'].keys())
        ft_acc = [dialog_data['fine_tuned'][c] for c in counts]
        bl_acc = [dialog_data['baseline'][c] for c in counts]

        plt.figure(figsize=(10, 6))
        plt.plot(counts, ft_acc, marker='o', label='Fine-tuned')
        plt.plot(counts, bl_acc, marker='o', label='Baseline')
        plt.xlabel('Number of Dialogs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Dialogs')
        plt.legend()
        plt.savefig('num_dialogs_comparison.png')
        plt.close()

        # Agent initiation comparison
        agent_data = report['agent_initiation_comparison']
        agents = ['agent1', 'agent2']
        ft_acc = [agent_data['fine_tuned'][a]['accuracy'] for a in agents]
        bl_acc = [agent_data['baseline'][a]['accuracy'] for a in agents]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(agents))
        width = 0.35
        plt.bar(x - width / 2, ft_acc, width, label='Fine-tuned')
        plt.bar(x + width / 2, bl_acc, width, label='Baseline')
        plt.xlabel('Initiating Agent')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Initiating Agent')
        plt.xticks(x, agents)
        plt.legend()
        plt.savefig('agent_initiation_comparison.png')
        plt.close()


if __name__ == "__main__":
    # Example usage
    analyzer = ExperimentAnalyzer(
        fine_tuned_results_path='fine_tuned_results.json',
        baseline_results_path='baseline_results.json',
        fever_data_path='fever_data.jsonl'
    )
    report = analyzer.generate_report('analysis_report.json')
    analyzer.visualize_results(report)
    print("Analysis complete. Report saved to 'analysis_report.json'")