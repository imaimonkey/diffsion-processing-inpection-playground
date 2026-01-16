# TTA Uncertainty Sampling - Usage Guide

## Quick Start

### 1. Basic Usage (Default Settings)

```python
from tta_uncertainty_sampling import generate_with_tta_uncertainty
from modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer

# Load model
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Base")
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base")

# Prepare prompt
prompt_text = "The capital of France is"
prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(model.device)

# Generate with TTA
result, logs = generate_with_tta_uncertainty(
    model=model,
    prompt=prompt_tokens,
    steps=64,
    gen_length=64,
    tta_k=5,  # 5 forward passes per step
    seed=42
)

# Decode result
generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
print(generated_text)
```

### 2. Benchmark Comparison

```python
import experiment_utils

# Compare Baseline vs TTA Uncertainty
results = experiment_utils.run_academic_benchmark(
    model=model,
    tokenizer=tokenizer,
    baseline_fn=None,  # Uses default inspect_sampling
    experimental_fn=generate_with_tta_uncertainty,
    thresholds=[3, 5, 7],  # Different tta_k values to test
    samples=20
)

# Analyze
experiment_utils.analyze_icml_results(results)
```

## Configuration Parameters

### TTA Settings
```python
tta_k=5                           # Number of forward passes (higher = better uncertainty estimation, slower)
tta_temperature_range=(0.0, 0.3) # Temperature range for diversity
tta_use_dropout=False             # Use dropout for TTA (requires model.train())
```

### Uncertainty Weights
```python
alpha_entropy=1.0        # Weight for predictive entropy
beta_disagreement=0.5    # Weight for disagreement rate
gamma_violation=0.3      # Weight for constraint violations
lambda_change=0.2        # Weight for change penalty (prevents oscillation)
```

### Remasking Policy
```python
remask_top_k=None        # Fixed number to remask (overrides top_pct)
remask_top_pct=0.1       # Remask top 10% highest uncertainty tokens
```

### Freeze Window (Anti-Oscillation)
```python
freeze_window=5          # Track last 5 steps
freeze_threshold=3       # If token flips 3+ times in window
freeze_duration=3        # Freeze for 3 steps
```

## Custom Constraint Functions

### Template
```python
def my_constraint_fn(tokens: torch.Tensor) -> torch.Tensor:
    """
    Args:
        tokens: [batch, seq_len] token IDs
        
    Returns:
        violation_scores: [batch, seq_len] penalty scores (higher = more violation)
    """
    batch_size, seq_len = tokens.shape
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    
    # Your constraint logic here
    # Example: penalize specific token IDs
    forbidden_tokens = [123, 456, 789]
    for token_id in forbidden_tokens:
        scores[tokens == token_id] += 1.0
    
    return scores
```

### Example 1: No Repeated N-grams
```python
def no_repeat_ngrams(tokens: torch.Tensor, n: int = 3) -> torch.Tensor:
    """Penalize repeated n-grams"""
    batch_size, seq_len = tokens.shape
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    
    for b in range(batch_size):
        seen_ngrams = set()
        for i in range(n-1, seq_len):
            ngram = tuple(tokens[b, i-n+1:i+1].cpu().numpy())
            if ngram in seen_ngrams:
                scores[b, i] += 1.0
            seen_ngrams.add(ngram)
    
    return scores

# Use it
result, logs = generate_with_tta_uncertainty(
    model, prompt_tokens,
    constraint_fn=lambda t: no_repeat_ngrams(t, n=3)
)
```

### Example 2: Format Constraints (JSON)
```python
def json_format_constraint(tokens: torch.Tensor, tokenizer) -> torch.Tensor:
    """Encourage valid JSON structure"""
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    
    # Decode to check structure
    for b in range(tokens.shape[0]):
        text = tokenizer.decode(tokens[b], skip_special_tokens=True)
        
        # Simple heuristics
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        # Penalize imbalance
        imbalance = abs(open_braces - close_braces) + abs(open_brackets - close_brackets)
        
        # Apply penalty to last token (could be more sophisticated)
        if imbalance > 0:
            scores[b, -1] = imbalance
    
    return scores
```

### Example 3: Domain-Specific Rules (Math)
```python
def math_constraint(tokens: torch.Tensor, tokenizer) -> torch.Tensor:
    """Penalize invalid math expressions"""
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    
    for b in range(tokens.shape[0]):
        text = tokenizer.decode(tokens[b], skip_special_tokens=True)
        
        # Check for common math errors
        violations = 0
        
        # Division by zero
        if '/0' in text or '/ 0' in text:
            violations += 2.0
        
        # Unmatched parentheses
        if text.count('(') != text.count(')'):
            violations += 1.0
        
        # Apply to last few tokens
        if violations > 0:
            scores[b, -3:] = violations
    
    return scores
```

## Logging and Analysis

### Access TTA Statistics
```python
result, logs = generate_with_tta_uncertainty(model, prompt_tokens, tta_k=5)

# TTA-specific stats
tta_stats = logs['tta_stats']

print(f"Avg entropy per step: {tta_stats['avg_entropy_per_step']}")
print(f"Avg disagreement per step: {tta_stats['avg_disagreement_per_step']}")
print(f"Remask ratio per step: {tta_stats['remask_ratio_per_step']}")
print(f"Frozen ratio per step: {tta_stats['frozen_ratio_per_step']}")

# Standard logs (compatible with existing metrics)
print(f"Total steps: {logs['total_steps']}")
print(f"NFE (forward passes): {logs['nfe']}")  # = total_steps * tta_k
```

### Visualize Uncertainty Evolution
```python
import matplotlib.pyplot as plt

entropy_history = logs['tta_stats']['avg_entropy_per_step']
disagreement_history = logs['tta_stats']['avg_disagreement_per_step']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(entropy_history)
plt.title('Average Entropy per Step')
plt.xlabel('Step')
plt.ylabel('Entropy')

plt.subplot(1, 2, 2)
plt.plot(disagreement_history)
plt.title('Average Disagreement per Step')
plt.xlabel('Step')
plt.ylabel('Disagreement Rate')

plt.tight_layout()
plt.show()
```

## Performance Optimization

### Trade-off: Speed vs Uncertainty Quality

```python
# Fast (K=3, ~3x slower than baseline)
result, logs = generate_with_tta_uncertainty(
    model, prompt_tokens,
    tta_k=3,
    remask_top_pct=0.05
)

# Balanced (K=5, ~5x slower)
result, logs = generate_with_tta_uncertainty(
    model, prompt_tokens,
    tta_k=5,
    remask_top_pct=0.1
)

# High Quality (K=10, ~10x slower)
result, logs = generate_with_tta_uncertainty(
    model, prompt_tokens,
    tta_k=10,
    remask_top_pct=0.15
)
```

## Troubleshooting

### Issue: Too slow
- Reduce `tta_k` (try 3 instead of 5)
- Reduce `remask_top_pct` (less remasking)
- Reduce `steps` or `gen_length`

### Issue: Poor quality
- Increase `tta_k` for better uncertainty estimation
- Adjust weights (`alpha_entropy`, `beta_disagreement`)
- Add domain-specific `constraint_fn`

### Issue: Oscillation (tokens keep flipping)
- Increase `lambda_change` (stronger change penalty)
- Decrease `freeze_threshold` (freeze earlier)
- Increase `freeze_duration` (freeze longer)

### Issue: Constraint function not working
- Check return shape: must be `[batch, seq_len]`
- Check dtype: must be `torch.float32`
- Verify scores are positive (higher = more violation)
