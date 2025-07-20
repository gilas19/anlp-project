#!/usr/bin/env python3
"""
Optimized script to add stance classification (support/refute) to CMV conversation data
using sentence similarity between OP text and node text for stance detection.
"""

import json
import re
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import logging
from dataclasses import dataclass
from collections import OrderedDict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StanceClassifier:
    """Optimized stance classifier using batch sentence similarity"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the stance classifier with sentence similarity model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Similarity threshold for determining support vs refute
        self.similarity_threshold = 0.3

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        text = re.sub(r"&gt;.*?\n", "", text)  # Remove quotes
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove links
        text = re.sub(r"\n+", " ", text)  # Replace newlines with spaces
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text.strip()

    def classify_stances_batch(self, texts: List[str], op_text: str) -> List[str]:
        """Classify multiple texts at once for efficiency"""
        if not texts:
            return []

        cleaned_texts = [self.clean_text(text) for text in texts]
        cleaned_op_text = self.clean_text(op_text)

        # Filter out very short texts
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(cleaned_texts):
            if len(text) >= 20:
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return ["NO_STANCE"] * len(texts)

        # Encode all texts in batch
        all_texts = valid_texts + [cleaned_op_text]
        embeddings = self.model.encode(all_texts)

        # Calculate similarities
        op_embedding = embeddings[-1].reshape(1, -1)
        text_embeddings = embeddings[:-1]

        similarities = cosine_similarity(text_embeddings, op_embedding).flatten()

        # Determine stances
        stances = ["NO_STANCE"] * len(texts)
        for i, valid_idx in enumerate(valid_indices):
            similarity = similarities[i]
            if similarity >= self.similarity_threshold:
                stances[valid_idx] = "support"
            else:
                stances[valid_idx] = "refute"

        return stances


def collect_all_texts(node: Dict, texts: List[str], node_info: List[Dict]) -> None:
    """Recursively collect all texts from conversation tree"""
    if not node:
        return

    text = node.get("text", "")
    speaker = node.get("speaker", "")

    # Skip system messages but still process children
    if speaker not in ["DeltaBot", "[deleted]"] and text and text != "[deleted]":
        texts.append(text)
        node_info.append({"node": node, "text_index": len(texts) - 1})

    # Process children
    for child in node.get("children", []):
        collect_all_texts(child, texts, node_info)


def apply_stances_to_tree(node: Dict, stances: List[str], node_info: List[Dict], stance_index: int = 0) -> Dict:
    """Apply stance classifications to tree nodes with proper ordering"""
    if not node:
        return node

    # Create ordered node with proper property sequence
    ordered_node = OrderedDict()

    # Add properties in the desired order
    for prop in ["id", "speaker", "text"]:
        if prop in node:
            ordered_node[prop] = node[prop]

    # Add stance
    text = node.get("text", "")
    speaker = node.get("speaker", "")

    if speaker in ["DeltaBot", "[deleted]"] or not text or text == "[deleted]" or len(text.strip()) < 20:
        ordered_node["stance"] = "NO_STANCE"
    else:
        # Find corresponding stance
        stance = "NO_STANCE"
        for info in node_info:
            if info["node"] is node:
                text_idx = info["text_index"]
                if text_idx < len(stances):
                    stance = stances[text_idx]
                break
        ordered_node["stance"] = stance

    # Process children recursively
    children = node.get("children", [])
    processed_children = []
    for child in children:
        processed_child = apply_stances_to_tree(child, stances, node_info, stance_index)
        processed_children.append(processed_child)

    ordered_node["children"] = processed_children

    # Add any remaining properties
    for key, value in node.items():
        if key not in ordered_node:
            ordered_node[key] = value

    return dict(ordered_node)


def add_stance_classifications(input_file: str, output_file: str, model_name: str = None):
    """Add stance classifications directly to CMV conversation tree nodes"""
    logger.info(f"Loading data from: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    logger.info(f"Loaded {len(conversations)} conversations")

    # Initialize classifier
    classifier = StanceClassifier(model_name) if model_name else StanceClassifier()

    enhanced_conversations = []

    for i, conversation in enumerate(conversations):
        logger.info(f"Processing conversation {i+1}/{len(conversations)}: {conversation.get('title', '')[:50]}...")

        # Get OP text for similarity comparison
        op_text = conversation.get("op_text", "")

        if not op_text:
            logger.warning(f"No OP text found in conversation {i+1}")
            continue

        # Collect all texts from the conversation tree
        all_texts = []
        node_info = []

        tree = conversation.get("tree", [])
        for node in tree:
            collect_all_texts(node, all_texts, node_info)

        logger.info(f"  Found {len(all_texts)} texts to classify")

        # Classify all stances in batch
        if all_texts:
            stances = classifier.classify_stances_batch(all_texts, op_text)
        else:
            stances = []

        # Apply stances to tree
        enhanced_conversation = conversation.copy()
        processed_tree = []

        for node in tree:
            processed_node = apply_stances_to_tree(node, stances, node_info)
            processed_tree.append(processed_node)

        enhanced_conversation["tree"] = processed_tree

        # Count stances for summary
        def count_stances(node):
            counts = {"support": 0, "refute": 0, "NO_STANCE": 0}
            stance = node.get("stance", "NO_STANCE")
            counts[stance] += 1

            for child in node.get("children", []):
                child_counts = count_stances(child)
                for key in counts:
                    counts[key] += child_counts[key]

            return counts

        total_counts = {"support": 0, "refute": 0, "NO_STANCE": 0}
        for node in processed_tree:
            node_counts = count_stances(node)
            for key in total_counts:
                total_counts[key] += node_counts[key]

        total_responses = sum(total_counts.values())
        support_count = total_counts["support"]
        refute_count = total_counts["refute"]

        enhanced_conversations.append(enhanced_conversation)

        logger.info(
            f"  Processed {total_responses} responses: {support_count} support, {refute_count} refute, {total_counts['NO_STANCE']} no stance"
        )

    # Save enhanced data
    logger.info(f"Saving enhanced data to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_conversations, f, indent=2, ensure_ascii=False)

    logger.info("Stance classification completed!")

    # Print summary
    total_responses = 0
    total_support = 0
    total_refute = 0

    for conversation in enhanced_conversations:

        def count_total_stances(node):
            counts = {"support": 0, "refute": 0, "NO_STANCE": 0}
            stance = node.get("stance", "NO_STANCE")
            counts[stance] += 1

            for child in node.get("children", []):
                child_counts = count_total_stances(child)
                for key in counts:
                    counts[key] += child_counts[key]

            return counts

        conv_counts = {"support": 0, "refute": 0, "NO_STANCE": 0}
        for node in conversation.get("tree", []):
            node_counts = count_total_stances(node)
            for key in conv_counts:
                conv_counts[key] += node_counts[key]

        total_responses += sum(conv_counts.values())
        total_support += conv_counts["support"]
        total_refute += conv_counts["refute"]

    logger.info(f"\nSUMMARY:")
    logger.info(f"Total conversations processed: {len(enhanced_conversations)}")
    logger.info(f"Total responses classified: {total_responses}")
    logger.info(f"Support responses: {total_support} ({total_support/total_responses*100:.1f}%)")
    logger.info(f"Refute responses: {total_refute} ({total_refute/total_responses*100:.1f}%)")


def main():
    with open("config/preprocess/cmv.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    add_stance_classifications(config["input"], config["output"], config["stance"]["model"])


if __name__ == "__main__":
    main()
