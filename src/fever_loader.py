#!/usr/bin/env python3
"""
FEVER dataset loader and processor for fact verification experiments.
"""

import pandas as pd
import json
from typing import List, Dict, Tuple
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FEVERDataLoader:
    """Handles FEVER dataset loading and processing"""
    
    def __init__(self):
        self.splits = {
            'train': 'train.jsonl',
            'validation': 'valid.jsonl', 
            'test': 'test.jsonl'
        }
    
    def load_split(self, split='train', limit=None):
        """Load FEVER dataset split"""
        logger.info(f"Loading FEVER {split} split...")
        
        try:
            # Try loading with datasets library first
            from datasets import load_dataset
            
            dataset = load_dataset("copenlu/fever_gold_evidence", split=split)
            df = dataset.to_pandas()
            
            if limit:
                df = df.head(limit)
                
            logger.info(f"Loaded {len(df)} examples from {split}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading FEVER {split}: {e}")
            logger.info("Trying alternative loading method...")
            
            # Fallback: create sample data for testing
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for testing when FEVER loading fails"""
        logger.info("Creating sample test data...")
        
        sample_data = [
            {
                'claim': 'The Eiffel Tower is located in Paris.',
                'label': 'SUPPORTS',
                'evidence': [[[0, 2, 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.']]]
            },
            {
                'claim': 'The Eiffel Tower is located in Berlin.',
                'label': 'REFUTES', 
                'evidence': [[[0, 2, 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.']]]
            },
            {
                'claim': 'Gabrielle Union was in a movie.',
                'label': 'SUPPORTS',
                'evidence': [[[0, 2, 'She co-starred in film The Birth of a Nation (2016), and next appeared in Almost Christmas (2016) and Sleepless (2017).']]]
            },
            {
                'claim': 'Eleveneleven was founded by a chef.',
                'label': 'REFUTES',
                'evidence': [[[0, 2, 'eleveneleven is a record label founded in 2010 by Mike Hamlin, Ellen DeGeneres and her production company, in association with Warner Bros.']]]
            },
            {
                'claim': 'Tennis is a sport.',
                'label': 'SUPPORTS',
                'evidence': [[[0, 2, 'Tennis is played by millions of recreational players and is also a popular worldwide spectator sport.']]]
            },
            {
                'claim': 'Basketball was invented in 1891.',
                'label': 'SUPPORTS',
                'evidence': [[[0, 2, 'Basketball was invented in December 1891 by Canadian-American gym instructor James Naismith.']]]
            }
        ]
        
        # Duplicate data to create more examples
        extended_data = sample_data * 25  # 150 examples
        
        return pd.DataFrame(extended_data)
    
    def extract_evidence_text(self, evidence_entry):
        """Extract concatenated evidence text from nested structure"""
        texts = []
        
        # Check if evidence_entry is None or empty
        if evidence_entry is None or len(evidence_entry) == 0:
            return ""
            
        try:
            # Handle numpy array structure
            import numpy as np
            
            for item in evidence_entry:
                if isinstance(item, np.ndarray) and len(item) > 2:
                    # Direct numpy array with [page, sent_id, text]
                    texts.append(item[2])
                elif isinstance(item, list):
                    # List structure - check if it contains the triple
                    if len(item) > 2:
                        texts.append(item[2])
                    else:
                        # Nested list structure
                        for sentence in item:
                            if isinstance(sentence, (list, np.ndarray)) and len(sentence) > 2:
                                texts.append(sentence[2])
                                
        except (TypeError, IndexError) as e:
            # Handle malformed evidence entries
            return ""
                    
        return " ".join(texts)
    
    def process_for_evaluation(self, df):
        """Process FEVER data for evaluation"""
        # Extract evidence text
        df = df.copy()
        df['evidence_text'] = df['evidence'].apply(self.extract_evidence_text)
        
        # Filter out examples without evidence
        mask = df['evidence_text'].str.len() > 0
        df = df[mask].copy()
        
        # Keep only necessary columns
        processed_df = df[['claim', 'label', 'evidence_text']].copy()
        
        logger.info(f"Processed {len(processed_df)} examples with evidence")
        return processed_df
    
    def get_balanced_sample(self, df, n_per_class=50):
        """Get balanced sample across all three classes"""
        samples = []
        
        # If total requested samples is very small, just return first few examples
        total_requested = n_per_class * 3
        if total_requested <= 3:
            logger.info(f"Small sample size requested, returning first {total_requested} examples")
            return df.head(total_requested)
        
        for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
            label_df = df[df['label'] == label]
            if len(label_df) >= n_per_class:
                sample = label_df.sample(n=n_per_class, random_state=42)
                samples.append(sample)
            else:
                logger.warning(f"Only {len(label_df)} examples for {label}")
                samples.append(label_df)
        
        if samples:
            balanced_df = pd.concat(samples, ignore_index=True)
        else:
            # Fallback to first few examples
            balanced_df = df.head(total_requested)
            
        logger.info(f"Created balanced sample: {len(balanced_df)} examples")
        
        return balanced_df
    
    def get_evaluation_data(self, split='validation', sample_size=150):
        """Get processed evaluation data"""
        df = self.load_split(split)
        if df is None:
            return None
            
        processed_df = self.process_for_evaluation(df)
        
        if sample_size:
            n_per_class = max(1, sample_size // 3)  # Ensure at least 1 per class
            return self.get_balanced_sample(processed_df, n_per_class)
        
        return processed_df

def create_test_samples():
    """Create some test examples for quick validation"""
    test_examples = [
        {
            "claim": "The Eiffel Tower is in Paris.",
            "evidence": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "label": "SUPPORTS"
        },
        {
            "claim": "The Eiffel Tower is in Berlin.", 
            "evidence": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "label": "REFUTES"
        },
        {
            "claim": "The Eiffel Tower was built by aliens.",
            "evidence": "The Eiffel Tower is a wrought-iron lattice tower designed by Gustave Eiffel.",
            "label": "REFUTES"
        }
    ]
    
    return pd.DataFrame(test_examples)

if __name__ == "__main__":
    # Test the loader
    loader = FEVERDataLoader()
    test_df = loader.get_evaluation_data(sample_size=30)
    
    if test_df is not None:
        print(f"Sample data shape: {test_df.shape}")
        print("\nLabel distribution:")
        print(test_df['label'].value_counts())
        print("\nSample claims:")
        for _, row in test_df.head(3).iterrows():
            print(f"Claim: {row['claim']}")
            print(f"Label: {row['label']}")
            print(f"Evidence: {row['evidence_text'][:100]}...")
            print("---")