# Simulated Argumentative Discourse for Improved Fact Verification

This repository contains the implementation for our research project investigating whether simulated debates between opposing model-based agents improve fact verification accuracy compared to direct prediction methods.

## Project Overview

Current fact verification approaches rely on single-step model decisions, potentially missing nuanced reasoning that emerges from argumentative discourse. This project explores whether structured argumentative reasoning through simulated debates can improve automated fact verification.

## Project Structure

```
├── src/                        # Source code
│   ├── debate_system.py        # Enhanced debate simulation system
│   ├── fever_loader.py         # FEVER dataset integration
│   ├── evaluation_pipeline.py  # Comparison evaluation framework
│   ├── fine_tune_model.py      # Model fine-tuning on CMV data
│   └── run_experiments.py      # Main experiment runner
├── data/                       # Data files
│   └── cmv_argument_pairs_unique_claims.csv
├── experiments/                # Notebooks and experimental code
│   └── ANLP_project.ipynb      # Initial prototype notebook
├── results/                    # Generated results
│   ├── results_*.csv           # Detailed evaluation results
│   ├── metrics_*.json          # Summary metrics
│   └── experiment.log          # Execution logs
├── paper/                      # LaTeX paper files
│   ├── project.tex             # Main paper
│   ├── custom.bib              # Bibliography
│   └── ...                     # Other LaTeX files
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── run.py                      # Simple runner script
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Experiment
```bash
# Quick test with 6 examples
python run.py --sample-size 6

# Full experiment with 150 examples
python run.py --sample-size 150
```

## Key Components

1. **Fine-tuned Model**: Flan-T5-large fine-tuned on Reddit's r/ChangeMyView dataset
2. **Debate System**: Two opposing agents (support/refute) engage in structured debates
3. **Evaluation Pipeline**: Comprehensive comparison between debate-enhanced and baseline approaches
4. **FEVER Integration**: Uses FEVER benchmark for evaluation with gold-standard evidence

## Methodology

### 1. Model Preparation
- Fine-tune Flan-T5-large on Reddit ChangeMyView argumentative discourse
- Enhanced prompting for debate generation

### 2. Debate Simulation
- Two agents: one supporting, one refuting each claim
- Structured multi-round debates with evidence grounding
- Final judgment based on debate transcript

### 3. Evaluation
- FEVER benchmark with three-class labels (SUPPORTS/REFUTES/NOT ENOUGH INFO)
- Comparison metrics: accuracy, per-class performance, timing analysis
- Disagreement analysis between methods

## Advanced Usage

### Individual Components

**Fine-tune the model:**
```bash
python src/fine_tune_model.py
```

**Run evaluation pipeline:**
```bash
python src/evaluation_pipeline.py
```

**Test debate system:**
```python
import sys
sys.path.append('src')
from debate_system import DebateSimulator

sim = DebateSimulator()
claim = "The Eiffel Tower is in Paris."
evidence = "The Eiffel Tower is located on the Champ de Mars in Paris, France."

debate_history = sim.simulate_enhanced_debate(claim, evidence)
```

## Expected Results

- **Accuracy**: Debate method shows improvement over baseline
- **Reasoning Quality**: Enhanced argumentation through multi-agent discourse
- **Trade-offs**: Higher computational cost vs. improved performance

## Contributors

- Shir Babian
- Yael Batat  
- Gilad Ticher

Hebrew University of Jerusalem