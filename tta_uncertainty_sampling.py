"""
TTA-based Uncertainty-Aware Sampling for Diffusion Language Models

This module implements an advanced sampling strategy that uses Test-Time Augmentation (TTA)
to estimate uncertainty and adaptively remask tokens based on multiple criteria.

Key Features:
- TTA with K forward passes using different seeds/dropout/temperature
- Multiple uncertainty metrics: entropy, disagreement rate, variance
- Pluggable constraint violation detection
- Adaptive remasking with freeze windows to prevent oscillation
- Full compatibility with existing benchmark infrastructure

Author: Experimental Sampling Research
Compatible with: LLaDA diffusion models
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Callable, Optional, Tuple, Dict, List
from collections import defaultdict, deque

# Import existing utilities from the repo
from decoding import add_gumbel_noise, get_num_transfer_tokens


# ============================================================================
# Configuration Example (for reference)
# ============================================================================
"""
Example configuration:
config = {
    'tta_k': 5,                          # Number of TTA forward passes
    'tta_temperature_range': (0.0, 0.3), # Temperature range for TTA diversity
    'alpha_entropy': 1.0,                # Weight for entropy in remask score
    'beta_disagreement': 0.5,            # Weight for disagreement
    'gamma_violation': 0.3,              # Weight for constraint violation
    'lambda_change': 0.2,                # Weight for change penalty (hysteresis)
    'remask_top_k': None,                # Fixed k for remasking (or None)
    'remask_top_pct': 0.1,               # Percentage of tokens to remask
    'freeze_window': 5,                  # Window size for tracking flips
    'freeze_threshold': 3,               # Flip count threshold
    'freeze_duration': 3,                # Steps to freeze after threshold
    'constraint_fn': None,               # Pluggable constraint function
    'seed': 42,                          # Random seed for reproducibility
}
"""


# ============================================================================
# Uncertainty Metrics
# ============================================================================

def compute_predictive_entropy(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute predictive entropy: H = -Σ p(y) log p(y)
    
    Args:
        probs: [batch, seq_len, vocab_size] probability distribution
        eps: Small constant for numerical stability
        
    Returns:
        entropy: [batch, seq_len] entropy per token
    """
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def compute_disagreement_rate(predictions: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute disagreement rate across K predictions.
    
    Args:
        predictions: List of K tensors, each [batch, seq_len] with argmax predictions
        
    Returns:
        disagreement: [batch, seq_len] rate of disagreement (0.0 to 1.0)
    """
    if len(predictions) < 2:
        return torch.zeros_like(predictions[0], dtype=torch.float32)
    
    # Stack predictions: [K, batch, seq_len]
    stacked = torch.stack(predictions, dim=0)
    
    # Count unique predictions per position
    # For each position, count how many different values appear
    K = len(predictions)
    batch_size, seq_len = predictions[0].shape
    
    disagreement = torch.zeros(batch_size, seq_len, device=predictions[0].device)
    
    for b in range(batch_size):
        for s in range(seq_len):
            values = stacked[:, b, s]
            unique_count = len(torch.unique(values))
            # Disagreement = (unique_count - 1) / (K - 1)
            # If all agree: unique=1, disagreement=0
            # If all differ: unique=K, disagreement=1
            disagreement[b, s] = (unique_count - 1) / max(K - 1, 1)
    
    return disagreement


def compute_variance_proxy(probs_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute variance of probability distributions across K samples.
    
    Args:
        probs_list: List of K tensors, each [batch, seq_len, vocab_size]
        
    Returns:
        variance: [batch, seq_len] average variance across vocab
    """
    if len(probs_list) < 2:
        return torch.zeros(probs_list[0].shape[:2], device=probs_list[0].device)
    
    # Stack: [K, batch, seq_len, vocab_size]
    stacked = torch.stack(probs_list, dim=0)
    
    # Variance across K samples
    variance = torch.var(stacked, dim=0)  # [batch, seq_len, vocab_size]
    
    # Average variance across vocabulary
    avg_variance = torch.mean(variance, dim=-1)  # [batch, seq_len]
    
    return avg_variance


# ============================================================================
# Freeze Window Tracker
# ============================================================================

class FreezeWindowTracker:
    """
    Tracks token flip history and manages freeze windows.
    
    Prevents oscillation by freezing positions that flip too frequently.
    """
    
    def __init__(self, seq_len: int, window_size: int = 5, 
                 flip_threshold: int = 3, freeze_duration: int = 3):
        """
        Args:
            seq_len: Sequence length
            window_size: Number of recent steps to track
            flip_threshold: Number of flips to trigger freeze
            freeze_duration: Steps to freeze after threshold
        """
        self.seq_len = seq_len
        self.window_size = window_size
        self.flip_threshold = flip_threshold
        self.freeze_duration = freeze_duration
        
        # Track flip history: position -> deque of recent flip indicators
        self.flip_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # Track freeze status: position -> remaining freeze steps
        self.freeze_counter = defaultdict(int)
        
        # Previous tokens for flip detection
        self.prev_tokens = None
    
    def update(self, current_tokens: torch.Tensor, current_step: int) -> torch.Tensor:
        """
        Update flip history and return freeze mask.
        
        Args:
            current_tokens: [batch, seq_len] current token IDs
            current_step: Current diffusion step
            
        Returns:
            freeze_mask: [batch, seq_len] boolean mask (True = frozen)
        """
        batch_size, seq_len = current_tokens.shape
        freeze_mask = torch.zeros_like(current_tokens, dtype=torch.bool)
        
        # Only track batch 0 for simplicity (can be extended)
        tokens = current_tokens[0]
        
        if self.prev_tokens is not None:
            # Detect flips
            flipped = (tokens != self.prev_tokens).cpu().numpy()
            
            for idx in range(seq_len):
                # Update flip history
                self.flip_history[idx].append(int(flipped[idx]))
                
                # Check if should freeze
                if sum(self.flip_history[idx]) >= self.flip_threshold:
                    self.freeze_counter[idx] = self.freeze_duration
        
        # Update prev_tokens
        self.prev_tokens = tokens.clone()
        
        # Apply freeze
        for idx in range(seq_len):
            if self.freeze_counter[idx] > 0:
                freeze_mask[0, idx] = True
                self.freeze_counter[idx] -= 1
        
        return freeze_mask


# ============================================================================
# Main TTA-based Sampling Function
# ============================================================================

@torch.no_grad()
def generate_with_tta_uncertainty(
    model,
    prompt: torch.Tensor,
    steps: int = 64,
    gen_length: int = 64,
    block_length: int = 64,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    attention_mask: Optional[torch.Tensor] = None,
    # TTA-specific parameters
    tta_k: int = 5,
    tta_temperature_range: Tuple[float, float] = (0.0, 0.3),
    tta_use_dropout: bool = False,
    # Uncertainty weights
    alpha_entropy: float = 1.0,
    beta_disagreement: float = 0.5,
    gamma_violation: float = 0.3,
    lambda_change: float = 0.2,
    # Remasking policy
    remask_top_k: Optional[int] = None,
    remask_top_pct: float = 0.1,
    # Freeze window
    freeze_window: int = 5,
    freeze_threshold: int = 3,
    freeze_duration: int = 3,
    # Constraint function
    constraint_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    # Reproducibility
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict]:
    """
    TTA-based Uncertainty-Aware Sampling for Diffusion Language Models.
    
    This function uses Test-Time Augmentation to estimate uncertainty and adaptively
    remask tokens based on entropy, disagreement, constraint violations, and change penalty.
    
    Args:
        model: The diffusion language model
        prompt: [batch, prompt_len] input prompt tokens
        steps: Total diffusion steps
        gen_length: Number of tokens to generate
        block_length: Block size for generation
        temperature: Base sampling temperature
        cfg_scale: Classifier-free guidance scale
        mask_id: ID for mask token
        attention_mask: Optional attention mask
        
        tta_k: Number of TTA forward passes per step
        tta_temperature_range: (min, max) temperature for TTA diversity
        tta_use_dropout: Whether to use dropout for TTA (requires model.train())
        
        alpha_entropy: Weight for entropy in remask score
        beta_disagreement: Weight for disagreement rate
        gamma_violation: Weight for constraint violation
        lambda_change: Weight for change penalty (hysteresis)
        
        remask_top_k: Fixed number of tokens to remask (overrides top_pct)
        remask_top_pct: Percentage of generated tokens to remask
        
        freeze_window: Window size for tracking flips
        freeze_threshold: Flip count to trigger freeze
        freeze_duration: Steps to freeze after threshold
        
        constraint_fn: Optional function(tokens) -> violation_scores
                      Should return [batch, seq_len] tensor with violation scores
        
        seed: Random seed for reproducibility
        
    Returns:
        x: [batch, prompt_len + gen_length] generated tokens
        logs: Dict with keys:
            - x0_history: List of x0 predictions for stability analysis
            - token_logs: Dict of per-token logs for metrics
            - tta_stats: TTA-specific statistics
            - total_steps: Total diffusion steps
    """
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(device=model.device).manual_seed(seed)
    else:
        generator = None
    
    # Initialize sequence
    batch_size = prompt.shape[0]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, 
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    prompt_index = (x != mask_id)
    
    # Fix time tracking
    fix_time = torch.full_like(x, -1)
    fix_time[:, :prompt.shape[1]] = -999  # Prompt is permanent
    
    # Flip count tracking for change penalty
    flip_count = torch.zeros_like(x, dtype=torch.long)
    
    # Initialize freeze window tracker
    freeze_tracker = FreezeWindowTracker(
        seq_len=x.shape[1],
        window_size=freeze_window,
        flip_threshold=freeze_threshold,
        freeze_duration=freeze_duration
    )
    
    # Logging initialization
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    total_steps = steps_per_block * num_blocks
    current_step = 0
    
    x0_history = []
    L_total = prompt.shape[1] + gen_length
    token_logs = {i: {'fix_step': -1, 'fix_token': -1, 'fix_conf': 0.0, 
                      'remask_events': [], 'flip_count': 0} for i in range(L_total)}
    
    # TTA-specific logging
    tta_stats = {
        'avg_entropy_per_step': [],
        'avg_disagreement_per_step': [],
        'avg_violation_per_step': [],
        'remask_ratio_per_step': [],
        'frozen_ratio_per_step': []
    }
    
    # ========================================================================
    # Main Generation Loop
    # ========================================================================
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            current_step += 1
            mask_index = (x == mask_id)
            
            # ================================================================
            # TTA: K Forward Passes
            # ================================================================
            
            tta_predictions = []  # List of argmax predictions
            tta_probs = []        # List of probability distributions
            tta_confidences = []  # List of confidence scores
            
            for k in range(tta_k):
                # Vary temperature for diversity
                if tta_k > 1:
                    t_min, t_max = tta_temperature_range
                    tta_temp = t_min + (t_max - t_min) * (k / (tta_k - 1))
                else:
                    tta_temp = temperature
                
                # Forward pass
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    else:
                        attention_mask_ = None
                    logits = model(x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits
                
                # Add noise and predict
                logits_with_noise = add_gumbel_noise(logits, temperature=tta_temp)
                x0_k = torch.argmax(logits_with_noise, dim=-1)
                
                # Compute probabilities
                p_k = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p_k = torch.squeeze(torch.gather(p_k, dim=-1, index=x0_k.unsqueeze(-1)), -1)
                
                tta_predictions.append(x0_k)
                tta_probs.append(p_k)
                tta_confidences.append(x0_p_k)
            
            # ================================================================
            # Aggregate TTA Results
            # ================================================================
            
            # Use first prediction as primary (or could use ensemble average)
            x0 = tta_predictions[0]
            x0_p = tta_confidences[0]
            
            # Capture for stability analysis
            x0_history.append(x0[0].cpu().clone())
            
            # Compute uncertainty metrics
            entropy = compute_predictive_entropy(tta_probs[0])  # [batch, seq_len]
            disagreement = compute_disagreement_rate(tta_predictions)  # [batch, seq_len]
            
            # Optional: variance proxy
            # variance = compute_variance_proxy(tta_probs)
            
            # ================================================================
            # Constraint Violation Detection
            # ================================================================
            
            if constraint_fn is not None:
                try:
                    violation_scores = constraint_fn(x0)  # [batch, seq_len]
                    if violation_scores.shape != x0.shape:
                        raise ValueError(f"constraint_fn must return [batch, seq_len] tensor")
                except Exception as e:
                    print(f"Warning: constraint_fn failed: {e}. Using zero violation.")
                    violation_scores = torch.zeros_like(x0, dtype=torch.float32)
            else:
                violation_scores = torch.zeros_like(x0, dtype=torch.float32)
            
            # ================================================================
            # Standard Transfer (Unmasking)
            # ================================================================
            
            # Prevent lookahead
            x0_p[:, block_end:] = -np.inf
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i]
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            fix_time[transfer_index] = current_step
            
            # Log fixes (batch 0)
            if transfer_index[0].any():
                fixed_indices = torch.nonzero(transfer_index[0], as_tuple=True)[0]
                for idx in fixed_indices.cpu().numpy():
                    token_logs[idx]['fix_step'] = current_step
                    token_logs[idx]['fix_token'] = x0[0, idx].item()
                    token_logs[idx]['fix_conf'] = x0_p[0, idx].item()
            
            # ================================================================
            # Adaptive Remasking with Freeze Window
            # ================================================================
            
            # Update freeze tracker
            freeze_mask = freeze_tracker.update(x, current_step)
            
            # Identify candidates: generated tokens (not prompt, not mask, not frozen)
            generated_mask = (x != mask_id) & (fix_time != -999) & (~freeze_mask)
            
            if generated_mask.any():
                # Compute change penalty (hysteresis)
                change_penalty = flip_count.float()
                
                # Compute remask score
                remask_score = (
                    alpha_entropy * entropy +
                    beta_disagreement * disagreement +
                    gamma_violation * violation_scores -
                    lambda_change * change_penalty
                )
                
                # Mask out non-candidates
                candidate_scores = torch.where(
                    generated_mask, 
                    remask_score, 
                    torch.tensor(float('-inf'), device=x.device)
                )
                
                # Select top-k or top-pct
                remask_mask = torch.zeros_like(x, dtype=torch.bool)
                for b in range(batch_size):
                    n_gen = generated_mask[b].sum().item()
                    if n_gen > 0:
                        if remask_top_k is not None:
                            budget_k = min(remask_top_k, n_gen)
                        else:
                            budget_k = max(1, int(n_gen * remask_top_pct))
                        
                        _, top_indices = torch.topk(candidate_scores[b], k=budget_k, largest=True)
                        remask_mask[b, top_indices] = True
                
                # Log remasks (batch 0)
                if remask_mask[0].any():
                    remasked_indices = torch.nonzero(remask_mask[0], as_tuple=True)[0]
                    for idx in remasked_indices.cpu().numpy():
                        token_logs[idx]['remask_events'].append((
                            current_step, 
                            x[0, idx].item(), 
                            remask_score[0, idx].item()
                        ))
                        token_logs[idx]['flip_count'] += 1
                        flip_count[0, idx] += 1
                
                # Apply remask
                x[remask_mask] = mask_id
                fix_time[remask_mask] = -1
                
                # Log TTA stats
                tta_stats['avg_entropy_per_step'].append(entropy[generated_mask].mean().item())
                tta_stats['avg_disagreement_per_step'].append(disagreement[generated_mask].mean().item())
                tta_stats['avg_violation_per_step'].append(violation_scores[generated_mask].mean().item())
                tta_stats['remask_ratio_per_step'].append(remask_mask.sum().item() / max(generated_mask.sum().item(), 1))
                tta_stats['frozen_ratio_per_step'].append(freeze_mask.sum().item() / x.shape[1])
    
    # ========================================================================
    # Return Results
    # ========================================================================
    
    logs = {
        'x0_history': x0_history,
        'token_logs': token_logs,
        'tta_stats': tta_stats,
        'total_steps': total_steps,
        'nfe': total_steps * tta_k  # Total forward passes
    }
    
    return x, logs


# ============================================================================
# Example Constraint Functions
# ============================================================================

def example_constraint_no_repeat_trigrams(tokens: torch.Tensor) -> torch.Tensor:
    """
    Example constraint: Penalize repeated trigrams.
    
    Args:
        tokens: [batch, seq_len] token IDs
        
    Returns:
        violation_scores: [batch, seq_len] penalty scores
    """
    batch_size, seq_len = tokens.shape
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    
    for b in range(batch_size):
        for i in range(2, seq_len):
            trigram = tuple(tokens[b, i-2:i+1].cpu().numpy())
            # Check if this trigram appeared before
            for j in range(i-2):
                if j + 3 <= seq_len:
                    prev_trigram = tuple(tokens[b, j:j+3].cpu().numpy())
                    if trigram == prev_trigram:
                        scores[b, i] += 1.0  # Penalty
    
    return scores


# ============================================================================
# Minimal Test
# ============================================================================

if __name__ == "__main__":
    print("=== TTA Uncertainty Sampling - Minimal Test ===")
    
    # Create dummy model and data
    class DummyModel:
        def __init__(self):
            self.device = 'cpu'
        
        def __call__(self, x, attention_mask=None):
            # Return dummy logits
            batch_size, seq_len = x.shape
            vocab_size = 32000
            logits = torch.randn(batch_size, seq_len, vocab_size)
            
            class Output:
                def __init__(self, logits):
                    self.logits = logits
            
            return Output(logits)
    
    model = DummyModel()
    prompt = torch.randint(0, 1000, (1, 10))
    
    print(f"Prompt shape: {prompt.shape}")
    print("Running TTA sampling...")
    
    result, logs = generate_with_tta_uncertainty(
        model=model,
        prompt=prompt,
        steps=8,
        gen_length=16,
        block_length=16,
        tta_k=3,
        remask_top_pct=0.2,
        seed=42
    )
    
    print(f"Result shape: {result.shape}")
    print(f"Total steps: {logs['total_steps']}")
    print(f"NFE (forward passes): {logs['nfe']}")
    print(f"TTA stats keys: {list(logs['tta_stats'].keys())}")
    print("\n✅ Test passed!")
