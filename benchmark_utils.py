import re
import datasets
import torch
import numpy as np

def load_gsm8k(n_samples=50, split="test"):
    """
    Loads GSM8K dataset.
    Returns a list of dicts: {'question': str, 'answer': str, 'numerical_answer': str}
    """
    try:
        dataset = datasets.load_dataset("gsm8k", "main", split=split)
        # Shuffle and select n_samples to avoid bias if using a subset
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        
        formatted_data = []
        for item in dataset:
            # GSM8K answer format: "Reasoning... #### Number"
            full_answer = item['answer']
            numeric_match = re.search(r"####\s*(-?\d+\.?\d*)", full_answer)
            numerical_answer = numeric_match.group(1) if numeric_match else None
            
            if numerical_answer:
                formatted_data.append({
                    "category": "Math (GSM8K)",
                    "question": item['question'],
                    "ground_truth": numerical_answer,
                    "full_solution": full_answer
                })
        return formatted_data
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []

def load_mmlu_logic(n_samples=50, split="test"):
    """
    Loads a logic-heavy subset of MMLU (e.g., formal_logic, logical_fallacies).
    Returns list of dicts: {'question': str, 'ground_truth': str (A/B/C/D)}
    """
    subsets = ["formal_logic", "logical_fallacies", "philosophy"]
    formatted_data = []
    
    try:
        # Load a mix from subsets
        for sub in subsets:
            ds = datasets.load_dataset("cais/mmlu", sub, split=split)
            ds = ds.shuffle(seed=42).select(range(min(n_samples // len(subsets), len(ds))))
            
            for item in ds:
                options = item['choices']
                question = f"{item['question']}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\nAnswer:"
                answer_idx = item['answer']
                answer_char = ["A", "B", "C", "D"][answer_idx]
                
                formatted_data.append({
                    "category": f"Logic ({sub})",
                    "question": question,
                    "ground_truth": answer_char
                })
                
        # Limit to exact n_samples if slight overflow
        return formatted_data[:n_samples]
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return []

def extract_number(text):
    """
    Extracts the last number found in the text, commonly used for math answers.
    """
    # Look for "The answer is X" or just numbers at the end
    # Simple heuristic: find all numbers, take the last one.
    # Handles integers and floats.
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    return numbers[-1]

def extract_option(text):
    """
    Extracts A, B, C, D from the generated text.
    Prioritizes explicit "Answer: X" format.
    """
    # Pattern 1: "Answer: A"
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: Last single letter A-D
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
        
    return None

def check_correctness(prediction, ground_truth, category):
    """
    Checks if prediction matches ground truth based on category logic.
    """
    if "Math" in category:
        pred_num = extract_number(prediction)
        return pred_num == ground_truth
    elif "Logic" in category:
        pred_option = extract_option(prediction)
        return pred_option == ground_truth
    return False
