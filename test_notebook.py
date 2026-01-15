import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
# matplotlib initialization (non-interactive for script)
import matplotlib.pyplot as plt

print("Starting Notebook Execution Test...")

# --- Cell 2: Imports ---
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from modeling_llada import LLaDAModel
    from configuration_llada import LLaDAConfig
    from decoding import add_gumbel_noise, get_num_transfer_tokens
    from transformers import AutoTokenizer
    print("Local modules loaded successfully.")
except ImportError as e:
    print(f"CRITICAL: Failed to import local modules: {e}")
    sys.exit(1)

# --- Cell 3: Model Loading ---
# We will use a flag to skip actual heavy model loading if verifying logic only, 
# but user asked to run it. We will try HF load.
# To save time/bandwidth if it fails, we catch it.

LOCAL_MODEL_PATH = "../Grok-1-LLaDA-8B"
HF_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

model_path = HF_MODEL_ID
if os.path.exists(LOCAL_MODEL_PATH):
    model_path = LOCAL_MODEL_PATH
    print(f"Using Local Model: {model_path}")
else:
    print(f"Using HF Model: {model_path}")

model = None
tokenizer = None

try:
    print("Loading Config...")
    config = LLaDAConfig.from_pretrained(model_path)
    
    print("Loading Model... (This might take time)")
    # Using float16/bfloat16 to save memory if CUDA
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # We use 'cpu' or 'cuda' explicitly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Intentionally loading small if possible? No, LLaDA-8B is 8B.
    # If this is too heavy for the user's running environment, it will fail.
    # We will attempt to load.
    model = LLaDAModel.from_pretrained(model_path, config=config, torch_dtype=dtype)
    model.to(device)
    model.eval()
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model Loaded!")

except Exception as e:
    print(f"Model Load Failed: {e}")
    print("Skipping inference tests, but checking function definitions.")

# --- Cell 4: Inspection Function ---
@torch.no_grad()
def inspect_diffusion_process(
    model, tokenizer, prompt_text,
    steps=64, gen_length=64, block_length=64, temperature=0.0,
    mask_id=126336
):
    if prompt_text:
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(model.device)
    else:
        prompt_tokens = torch.tensor([[]], dtype=torch.long, device=model.device)

    B, L_prompt = prompt_tokens.shape
    total_len = L_prompt + gen_length
    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :L_prompt] = prompt_tokens.clone()

    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    history = []

    for num_block in range(num_blocks):
        block_start = L_prompt + num_block * block_length
        block_end = L_prompt + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)
            
            mask_index = (x == mask_id)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            if i < num_transfer_tokens.shape[1]:
                k = num_transfer_tokens[0, i].item()
            else: k = 0
            
            top_values, top_indices = torch.topk(confidence[0], k=k)
            transfer_mask = torch.zeros_like(x, dtype=torch.bool)
            transfer_mask[0, top_indices] = True
            x[transfer_mask] = x0[transfer_mask]
            
            history.append({
                'step': i, 'text': tokenizer.decode(x[0], skip_special_tokens=True),
                'confidence': confidence[0].detach().cpu().numpy()
            })
    return x, history

# --- Cell X: Custom Sampling ---
@torch.no_grad()
def custom_sampling(model, tokenizer, prompt_text, steps=64, gen_length=64, block_length=64, temperature=0.0):
    mask_id = 126336
    if prompt_text:
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(model.device)
    else:
        prompt_tokens = torch.tensor([[]], dtype=torch.long, device=model.device)

    B, L_prompt = prompt_tokens.shape
    x = torch.full((B, L_prompt + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :L_prompt] = prompt_tokens
    
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    
    history = []
    start_time = time.time()
    nfe = 0
    remask_threshold = 0.4

    for num_block in range(num_blocks):
        block_start = L_prompt + num_block * block_length
        block_end = L_prompt + (num_block + 1) * block_length
        
        for i in range(steps):
            logits = model(x).logits
            nfe += 1
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            x0_p = torch.gather(F.softmax(logits.to(torch.float64), dim=-1), -1, x0.unsqueeze(-1)).squeeze(-1)
            
            mask_index = (x == mask_id)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            current_masks = mask_index.sum().item()
            if current_masks == 0: break
            
            # Simple Linear Schedule
            k = max(1, current_masks // (steps - i + 1))
            
            _, top_indices = torch.topk(confidence[0], k=k)
            x[0, top_indices] = x0[0, top_indices]
            
            # Remasking
            generated_mask = (x != mask_id)
            generated_mask[:, :L_prompt] = False
            current_token_p = torch.gather(F.softmax(logits.to(torch.float64), dim=-1), -1, x.unsqueeze(-1)).squeeze(-1)
            low_conf_mask = (current_token_p < remask_threshold) & generated_mask
            
            if low_conf_mask.any():
                num_remask = min(low_conf_mask.sum().item(), max(1, k // 2))
                remask_candidates = torch.where(low_conf_mask, current_token_p, np.inf)
                _, remask_indices = torch.topk(remask_candidates[0], k=num_remask, largest=False)
                x[0, remask_indices] = mask_id

            history.append({'step': i, 'nfe': nfe, 'block': num_block, 'time': time.time()-start_time})
            
    return x, history, {'time': time.time()-start_time, 'nfe': nfe}


# --- Execution ---
if model is not None:
    print("\nRunning Inference Test (Short)...")
    prompt = "def hello_world():"
    try:
        # Standard
        print("1. Standard Inspection...")
        res, hist = inspect_diffusion_process(model, tokenizer, prompt, steps=10, gen_length=10, block_length=10)
        print("   Result:", tokenizer.decode(res[0], skip_special_tokens=True))
        
        # Custom
        print("2. Custom Sampling...")
        res_c, hist_c, stats = custom_sampling(model, tokenizer, prompt, steps=10, gen_length=10, block_length=10)
        print("   Result:", tokenizer.decode(res_c[0], skip_special_tokens=True))
        print(f"   Stats: {stats}")
        
        print("\nSUCCESS: All functions executed without error.")
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nWARNING: Verification incomplete because model could not be loaded.")
