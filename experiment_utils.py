
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from IPython.display import display, HTML
import benchmark_utils
import decoding
from decoding import inspect_sampling, generate_with_temporal_decay

# --- Metrics ---

def calculate_perplexity(model, tokenizer, text):
    """
    Calculates Perplexity (PPL) of the generated text using the model itself.
    Note: For diffusion models, this is a 'Pseudo-PPL' or 'Reconstruction PPL'.
    """
    if not text.strip(): return 0.0
    
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits 
        
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = input_ids.view(-1)
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)
        
        ppl = torch.exp(loss)
        return ppl.item()

def calculate_diversity(text):
    """
    Calculates Distinct-1 and Distinct-2 scores.
    """
    if not text: return 0.0, 0.0
    tokens = text.split()
    if not tokens: return 0.0, 0.0
    
    # Dist-1
    unique_1 = len(set(tokens))
    total_1 = len(tokens)
    dist_1 = unique_1 / total_1 if total_1 > 0 else 0.0
    
    # Dist-2
    bi_grams = list(zip(tokens[:-1], tokens[1:]))
    if not bi_grams: return dist_1, 0.0
    
    unique_2 = len(set(bi_grams))
    total_2 = len(bi_grams)
    dist_2 = unique_2 / total_2 if total_2 > 0 else 0.0
    
    return dist_1, dist_2

class DiffusionMetrics:
    @staticmethod
    def compute_stability(x0_history):
        """
        Stability Score: % of time the predicted token stays the same as previous step.
        """
        if not x0_history: return 0.0
        # Stack history: [Steps, SeqLen]
        # x0_history is list of tensors, ensure all on same device
        if isinstance(x0_history[0], torch.Tensor):
             device = x0_history[0].device
             hist = torch.stack(x0_history).to(device)
        else:
             # handle case where it might be numpy or list
             hist = torch.tensor(x0_history)
             
        if hist.shape[0] < 2: return 1.0
        
        # Compare t with t-1
        changes = (hist[1:] != hist[:-1]).float()
        # Stability = 1 - change_rate
        mean_change = changes.mean().item()
        return 1.0 - mean_change
    
    @staticmethod
    def compute_correction_efficacy(token_logs, final_tokens):
        """
        Did remasking lead to a different (and hopefully better) token?
        """
        remask_count = 0
        changed_count = 0
        
        for idx, log in token_logs.items():
            events = log['remask_events']
            if not events: continue
            
            remask_count += len(events)
            first_fix = log['fix_token']
            final_tok = final_tokens[idx].item() if idx < len(final_tokens) else -1
            
            if first_fix != -1 and final_tok != first_fix:
                changed_count += 1
        
        efficacy = changed_count / remask_count if remask_count > 0 else 0.0
        return efficacy, remask_count
    
    @staticmethod
    def compute_survival_rate(token_logs, final_tokens):
        """
        Survival Rate: % of first-fixed tokens that survived to the end.
        """
        total = 0
        survived = 0
        
        for idx, log in token_logs.items():
            first_fix = log['fix_token']
            if first_fix == -1: continue # Never fixed (prompt?)
            
            total += 1
            final_tok = final_tokens[idx].item() if idx < len(final_tokens) else -2
            
            if final_tok == first_fix:
                survived += 1
        
        return survived / total if total > 0 else 1.0

# --- Benchmark Loop ---

def run_academic_benchmark(model, tokenizer, 
                           baseline_fn=None,
                           experimental_fn=None,
                           thresholds=[0.05, 0.07, 0.10], 
                           samples=50, 
                           steps=64, gen_length=64, block_length=64, 
                           remask_budget=0.05):
    """
    Flexible benchmark function that compares baseline vs experimental sampling.
    
    Args:
        baseline_fn: Function(model, tokenizer, prompt, steps, gen_length, block_length) -> (result, history)
                    If None, uses inspect_sampling
        experimental_fn: Function(model, prompt_tokens, steps, gen_length, block_length, alpha_decay, remask_budget) -> (result, logs)
                        If None, uses generate_with_temporal_decay
        thresholds: List of alpha_decay values to test
    """
    print(f"Loading Academic Benchmarks (N={samples} per task)...")
    
    # Default functions
    if baseline_fn is None:
        baseline_fn = inspect_sampling
    if experimental_fn is None:
        experimental_fn = generate_with_temporal_decay
    
    # Load via benchmark_utils
    gsm8k_data = benchmark_utils.load_gsm8k(n_samples=samples)
    mmlu_data = benchmark_utils.load_mmlu_logic(n_samples=samples)
    
    # Combine
    full_dataset = gsm8k_data + mmlu_data
    print(f"Loaded {len(full_dataset)} total samples.")
    
    results = []
    total_runs = len(full_dataset) * len(thresholds)
    current_run = 0
    
    for item in full_dataset:
        category = item['category']
        prompt = item['question']
        ground_truth = item['ground_truth']
        
        # 1. Baseline
        res_base, hist_base = baseline_fn(
            model, tokenizer, prompt, steps=steps, 
            gen_length=gen_length, block_length=block_length
        )
        text_base = tokenizer.decode(res_base[0], skip_special_tokens=True)
        
        # Metrics Base
        ppl_base = calculate_perplexity(model, tokenizer, text_base)
        correct_base = benchmark_utils.check_correctness(text_base, ground_truth, category)
        stab_base = DiffusionMetrics.compute_stability([h['x0_pred'][0] for h in hist_base])
        
        # 2. Experimental - Test each threshold (alpha_decay value)
        for alpha_decay in thresholds:
            current_run += 1
            if current_run % 10 == 0:
                print(f"Progress: [{current_run}/{total_runs}] - Testing alpha_decay={alpha_decay}")
                
            # Prepare prompt tokens
            if prompt:
                prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
            else:
                prompt_tokens = torch.tensor([[]], dtype=torch.long, device=model.device)
                
            # Run experimental sampling with current alpha_decay
            res_exp, logs_exp = experimental_fn(
                model=model,
                prompt=prompt_tokens,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                remask_budget=remask_budget,
                alpha_decay=alpha_decay  # NOW ACTUALLY USING THE THRESHOLD!
            )
            
            text_exp = tokenizer.decode(res_exp[0], skip_special_tokens=True)
            
            # Metrics Exp
            ppl_exp = calculate_perplexity(model, tokenizer, text_exp)
            correct_exp = benchmark_utils.check_correctness(text_exp, ground_truth, category)
            
            # Extract logs if available
            if isinstance(logs_exp, dict):
                stab_exp = DiffusionMetrics.compute_stability(logs_exp.get('x0_history', []))
                eff_exp, _ = DiffusionMetrics.compute_correction_efficacy(logs_exp.get('token_logs', {}), res_exp[0])
                surv_exp = DiffusionMetrics.compute_survival_rate(logs_exp.get('token_logs', {}), res_exp[0])
            else:
                stab_exp, eff_exp, surv_exp = 0.0, 0.0, 0.0

            results.append({
                "Category": category,
                "Prompt": prompt,
                "GroundTruth": ground_truth,
                "AlphaDecay": alpha_decay,  # Clearer naming
                # Accuracy (Boolean 1/0 for averaging)
                "Acc_Base": 1.0 if correct_base else 0.0,
                "Acc_Exp": 1.0 if correct_exp else 0.0,
                "Acc_Delta": (1.0 if correct_exp else 0.0) - (1.0 if correct_base else 0.0),
                # PPL
                "PPL_Base": ppl_base,
                "PPL_Exp": ppl_exp,
                "PPL_Delta": ppl_exp - ppl_base,
                # Stability
                "Stability_Delta": stab_exp - stab_base,
                "Survival": surv_exp,
                "Correction_Eff": eff_exp
            })
            
    return pd.DataFrame(results)

# --- Analysis ---

def analyze_icml_results(df):
    print("\\n===== ICML Benchmark Results =====")
    
    # 1. Main Table: Accuracy & PPL per AlphaDecay
    # Ensure numeric columns
    numeric_cols = ["Acc_Base", "Acc_Exp", "Acc_Delta", "PPL_Delta", "Stability_Delta"]
    summary = df.groupby(["Category", "AlphaDecay"])[numeric_cols].mean()
    display(summary)
    
    # 2. Visualization
    plt.figure(figsize=(15, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x="AlphaDecay", y="Acc_Exp", hue="Category")
    plt.axhline(df["Acc_Base"].mean(), color='red', linestyle='--', label="Baseline Avg")
    plt.title("Accuracy vs Alpha Decay (Higher is Better)")
    plt.legend()
    
    # PPL Delta Plot
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x="AlphaDecay", y="PPL_Delta", hue="Category", marker="o")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Perplexity Delta (Lower is Better)")
    
    plt.tight_layout()
    plt.show()
    
    # 3. Statistical Highlight
    print("\\n[Key Findings]")
    # Mean across all categories for each alpha_decay
    global_stats = df.groupby("AlphaDecay")["Acc_Exp"].mean()
    if not global_stats.empty:
        best_alpha = global_stats.idxmax()
        best_acc = global_stats.max()
        base_acc = df["Acc_Base"].mean()
        
        print(f"Best Alpha Decay: {best_alpha}")
        print(f"Optimal Accuracy: {best_acc:.2%}")
        print(f"Baseline Accuracy: {base_acc:.2%}")
        print(f"Improvement: {best_acc - base_acc:+.2%}")
