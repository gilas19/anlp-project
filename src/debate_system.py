#!/usr/bin/env python3
"""
Enhanced debate simulation system for fact verification using multi-agent reasoning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebateSimulator:
    """Enhanced debate simulation with improved prompting"""
    
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def generate_response(self, prompt, max_tokens=128, temperature=0.9):
        """Generate model response with parameters"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_length=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_enhanced_agent_prompt(self, agent, claim, evidence, history):
        """Enhanced prompting with better context and instructions"""
        
        role = "support" if agent == "A" else "refute"
        opponent = "B" if agent == "A" else "A"
        
#         context = f"""You are participating in a structured fact-verification debate.
        
# Claim: "{claim}"
# Evidence: "{evidence}"

# Your Role: {role.upper()} the claim using the evidence provided.

# Guidelines:
# - Reference specific details from the evidence
# - Build logical arguments connecting evidence to your position
# - Address your opponent's key points when relevant
# - Stay focused and persuasive (max 3 sentences)
# - Base arguments on factual analysis

# Debate History:
# """
        context = f"""Claim: "{claim}"
Evidence: "{evidence}"

Your role is to {role} the claim using the evidence and respond thoughtfully to your opponent's arguments.
Reference specific points made by Agent {opponent} where appropriate, and try to persuade a neutral judge.
Limit your response to 3 sentences.

Debate so far:
"""
        for speaker, argument in history:
            context += f"Agent {speaker}: {argument}\n"
            
        context += f"\nAgent {agent}: "
        return context

    def simulate_enhanced_debate(self, claim, evidence, rounds=3):
        """Run enhanced debate simulation"""
        history = []
        current_agent = "A"
        
        for round_num in range(rounds * 2):
            prompt = self.create_enhanced_agent_prompt(current_agent, claim, evidence, history)
            response = self.generate_response(prompt).strip()
            history.append((current_agent, response))
            current_agent = "B" if current_agent == "A" else "A"
            
        return history

    def create_judgment_prompt(self, claim, evidence, history):
        """Enhanced final judgment prompt"""
        prompt = f"""Fact Verification Task

Claim: "{claim}"
Evidence: "{evidence}"

Debate Analysis:
"""
        for speaker, argument in history:
            prompt += f"Agent {speaker}: {argument}\n"
            
        prompt += """
Task: Based on the evidence and debate analysis, classify the claim:

- SUPPORTS: Evidence clearly validates the claim
- REFUTES: Evidence clearly contradicts the claim  
- NOT ENOUGH INFO: Evidence insufficient for determination

Consider argument strength and evidence quality in your decision.

Classification:"""
        return prompt

class BaselineClassifier:
    """Direct classification without debate for comparison"""
    
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=50, do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def classify_direct(self, claim, evidence):
        """Direct classification without debate"""
        prompt = f"""Classify the relationship between claim and evidence.

Claim: "{claim}"
Evidence: "{evidence}"

Options: SUPPORTS, REFUTES, NOT ENOUGH INFO

Answer:"""
        
        response = self.generate_response(prompt).strip().upper()
        
        if "SUPPORTS" in response:
            return "SUPPORTS"
        elif "REFUTES" in response:
            return "REFUTES"
        else:
            return "NOT ENOUGH INFO"