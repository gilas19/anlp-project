import json
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import jsonlines
from tqdm import tqdm
import os
from typing import List, Dict, Any
import os
from pathlib import Path

# Get the parent directory of src (anlp-project)
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths relative to project root
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "analysis" / "plots"
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_fever_data():
    """Load and preprocess FEVER dataset, extracting clean evidence text only"""
    try:
        dataset = load_dataset("copenlu/fever_gold_evidence")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    # Check available splits
    print(f"Available splits: {list(dataset.keys())}")

    claims = []
    for split in ['train', 'validation', 'test']:
        if split not in dataset:
            print(f"Warning: Split {split} not found in dataset")
            continue

        for item in dataset[split]:
            # Handle label
            label = item.get('gold_label') or item.get('label')
            if label is None:
                print(f"Warning: No label found for item {item.get('id')}")
                continue

            # Process evidence - extract only text
            raw_evidence = item.get("evidence", [])
            clean_evidence = []

            if isinstance(raw_evidence, list):
                for e in raw_evidence:
                    if isinstance(e, list) and len(e) >= 3:  # [title, line_num, text] format
                        clean_evidence.append(e[2])  # Keep only the text
                    elif isinstance(e, str):  # Handle unexpected string format
                        clean_evidence.append(e)

            # Join all evidence texts with spaces
            evidence_text = ' '.join(clean_evidence) if clean_evidence else "No evidence provided"

            claims.append({
                "id": str(item.get("id", "")),
                "claim": item.get("claim", ""),
                "gold_label": str(label).upper().strip(),
                "evidence": evidence_text[:1000]  # Truncate long evidence
            })

    if not claims:
        raise ValueError("No valid claims were loaded from the dataset")

    print(f"Successfully loaded {len(claims)} claims with clean evidence text")
    return claims
def load_models():
    """Load baseline and fine-tuned models"""
    baseline_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained("/Users/yaelbatat/Desktop/pythonProject/ANLP/anlpProject/anlp-project/models/flan-t5-small-cmv-debate-lora")  # Update this
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    return {
        "baseline": baseline_model,
        "fine_tuned": fine_tuned_model,
        "tokenizer": tokenizer
    }


def generate_response(model, tokenizer, prompt: str) -> Dict[str, Any]:
    """Generate a response from the model with proper length handling"""
    start_time = time.time()

    # Tokenize with truncation but without padding
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    # Move input_ids tensor to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    duration = time.time() - start_time
    length = len(tokenizer.tokenize(text))

    return {
        "text": text,
        "length": length,
        "time": duration
    }


def standardize_prediction(text: str) -> str:
    """Force prediction into one of the three valid categories"""
    text = text.strip().upper()
    if "SUPPORT" in text:
        return "SUPPORTS"
    elif "REFUTE" in text:
        return "REFUTES"
    elif "NOT ENOUGH" in text or "ENOUGH INFO" in text:
        return "NOT ENOUGH INFO"
    return "NOT ENOUGH INFO"  # Default fallback


def run_no_debate(config: Dict[str, Any], models: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the no-debate configuration with evidence and strict output formatting"""
    model = models[config["model"]]
    tokenizer = models["tokenizer"]
    results = []

    for claim in tqdm(claims, desc=f"Running {config['config_id']}"):
        # Prepare evidence snippet (first 500 chars if available)
        evidence = claim.get("evidence", "")
        evidence_snippet = evidence[:500] if evidence else "No evidence provided"

        # Create prompt with evidence
        prompt = (
            f"Fact Verification Task:\n"
            f"Claim: {claim['claim']}\n"
            f"Available Evidence: {evidence_snippet}\n\n"
            f"Based on this information, the claim is: (SUPPORTS/REFUTES/NOT ENOUGH INFO)\n"
            f"Answer:"
        )

        # Generate and standardize response
        response = generate_response(model, tokenizer, prompt)
        pred = standardize_prediction(response["text"])

        # Build result record
        record = {
            "claim_id": claim["id"],
            "claim": claim["claim"],
            "gold_label": claim["gold_label"],
            "model": config["model"],
            "with_debate": False,
            "initiator": None,
            "num_turns": 0,
            "evidence_used": evidence_snippet,
            "final_prediction": pred,
            "accuracy": pred == claim["gold_label"],
            "avg_length": response["length"],
            "avg_time": response["time"],
            "dialogue": [],
            "prompt_used": prompt  # For debugging
        }
        results.append(record)

    return results

def create_agent_prompt(agent, claim, evidence, history):
    """Creates a structured prompt for debate agents to either support or refute a claim"""

    role = "support" if agent == "A" else "refute"
    opponent = "B" if agent == "A" else "A"

    instructions = f"""Your role is to {role} the claim using the evidence and respond thoughtfully to your opponent's arguments.
Reference specific points made by Agent {opponent} where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences."""

    prompt = f"""Claim: "{claim}"
Evidence: "{evidence}"

{instructions}

Debate so far:
"""
    for i, (speaker, utterance) in enumerate(history):
        prompt += f"Agent {speaker}: {utterance}\n"

    prompt += f"Agent {agent}:"
    return prompt
# Example conversion from string to list of (speaker, utterance)
def parse_transcript(transcript_str):
    history = []
    for line in transcript_str.strip().split("\n"):
        if line.startswith("Agent A:"):
            history.append(("A", line[len("Agent A:"):].strip()))
        elif line.startswith("Agent B:"):
            history.append(("B", line[len("Agent B:"):].strip()))
    return history
def create_final_judgment_prompt(claim, evidence, history):
    """Creates a prompt for final judgment on whether evidence supports the claim"""

    prompt = f"""Claim: "{claim}"
Evidence: "{evidence}"

Here is a debate about whether the evidence supports the claim:\n"""

    for speaker, utterance in history:
        prompt += f"Agent {speaker}: {utterance}\n"

    prompt += """\nFinal Task: Based on the above, classify the claim as one of the following:
- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

Answer:"""
    return prompt
def run_debate(config: Dict[str, Any], models: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the debate configuration using create_agent_prompt"""
    model = models[config["model"]]
    tokenizer = models["tokenizer"]
    results = []

    for claim in tqdm(claims, desc=f"Running {config['config_id']}"):
        evidence = claim.get("evidence", "")
        evidence_snippet = evidence[:500] if evidence else "No evidence provided"

        dialogue = []
        history = []  # history is a list of (agent, text)
        current_agent = "A" if config["initiator"] == "agent1" else "B"
        total_length = 0
        total_time = 0

        for turn in range(config["num_turns"] * 2):
            prompt = create_agent_prompt(current_agent, claim['claim'], evidence_snippet, history)
            response = generate_response(model, tokenizer, prompt)

            stance = "SUPPORTS" if current_agent == "A" else "REFUTES"
            prev_agent = "B" if current_agent == "A" else "A"

            dialogue.append({
                "turn": turn + 1,
                "agent": f"agent1" if current_agent == "A" else "agent2",
                "stance": stance,
                "reply_to": "claim" if turn == 0 else f"agent{2 if current_agent == 'A' else 1}",
                "text": response["text"],
                "length": response["length"],
                "time": response["time"],
                "prompt_used": prompt
            })

            history.append((current_agent, response["text"]))
            current_agent = prev_agent
            total_length += response["length"]
            total_time += response["time"]

        # Build debate transcript (Agent A/B format)
        debate_transcript = "\n".join(
            f"Turn {d['turn']} ({d['agent']} - {d['stance']}): {d['text']}"
            for d in dialogue
        )

        history = parse_transcript(debate_transcript)
        final_prompt = create_final_judgment_prompt(claim["claim"], evidence_snippet, history)

        final_response = generate_response(model, tokenizer, final_prompt)
        pred = standardize_prediction(final_response["text"])

        record = {
            "claim_id": claim["id"],
            "claim": claim["claim"],
            "gold_label": claim["gold_label"],
            "model": config["model"],
            "with_debate": True,
            "initiator": config["initiator"],
            "num_turns": config["num_turns"],
            "evidence_used": evidence_snippet,
            "final_prediction": pred,
            "accuracy": pred == claim["gold_label"],
            "avg_length": total_length / (config["num_turns"] * 2),
            "avg_time": total_time / (config["num_turns"] * 2),
            "dialogue": dialogue,
            "final_prompt_used": final_prompt
        }
        results.append(record)

    return results
def save_results(results: List[Dict[str, Any]], config: Dict[str, Any]):
    """Save results to JSON file"""
    model_dir = "baseline" if config["model"] == "baseline" else "fine_tuned"
    os.makedirs(OUTPUT_DIR / model_dir, exist_ok=True)

    filename = OUTPUT_DIR / model_dir / f"{config['config_id']}.json"
    with open(filename, 'w') as f:
        json.dump({
            "config": config,
            "results": results,
            "stats": {
                "accuracy": sum(r["accuracy"] for r in results) / len(results),
                "avg_length": sum(r["avg_length"] for r in results) / len(results),
                "avg_time": sum(r["avg_time"] for r in results) / len(results)
            }
        }, f, indent=2)

def generate_configurations():
    """Generate all experiment configurations with organized directory structure"""
    configs = []

    # Create model-specific directories
    os.makedirs(CONFIG_DIR / "baseline", exist_ok=True)
    os.makedirs(CONFIG_DIR / "fine_tuned", exist_ok=True)

    # No-debate configurations
    baseline_no_debate = {
        "config_id": "no_debate",
        "model": "baseline",
        "with_debate": False,
        "initiator": None,
        "num_turns": 0
    }
    configs.append(baseline_no_debate)

    fine_tuned_no_debate = {
        "config_id": "no_debate",
        "model": "fine_tuned",
        "with_debate": False,
        "initiator": None,
        "num_turns": 0
    }
    configs.append(fine_tuned_no_debate)

    # Debate configurations
    debate_configs = [
        {"num_turns": 1, "initiator": "agent1"},
        {"num_turns": 1, "initiator": "agent2"},
        {"num_turns": 2, "initiator": "agent1"},
        {"num_turns": 2, "initiator": "agent2"},
        {"num_turns": 6, "initiator": "agent1"},
        {"num_turns": 6, "initiator": "agent2"},
        {"num_turns": 12, "initiator": "agent1"},
        {"num_turns": 12, "initiator": "agent2"},
    ]

    for model in ["baseline", "fine_tuned"]:
        for debate_config in debate_configs:
            config = {
                "config_id": f"{debate_config['initiator']}_{debate_config['num_turns']}turns",
                "model": model,
                "with_debate": True,
                "initiator": debate_config["initiator"],
                "num_turns": debate_config["num_turns"]
            }
            configs.append(config)

    for config in configs:
        model_dir = "baseline" if config["model"] == "baseline" else "fine_tuned"
        config_path = CONFIG_DIR / model_dir / f"{config['config_id']}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    return configs


def main():
    """Main execution function"""
    # Load data and models
    claims = load_fever_data()
    models = load_models()
    configs = generate_configurations()

    # For testing, use a subset of claims
    claims = claims[:100]  # Remove this line to use full dataset

    # Run all configurations
    for config in configs:
        if config["with_debate"]:
            results = run_debate(config, models, claims)
        else:
            results = run_no_debate(config, models, claims)
        save_results(results, config)  # Pass the full config now


if __name__ == "__main__":
    main()