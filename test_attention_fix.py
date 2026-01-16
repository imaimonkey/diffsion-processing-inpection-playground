"""
Quick test script to verify attention extraction works correctly
"""
import torch
from transformers import AutoTokenizer
from modeling_llada import LLaDAModelLM, LLaDAConfig
from decoding import extract_attention_influence

# Load model configuration
print("Loading model configuration...")
config = LLaDAConfig.from_pretrained("/nas-checkpoint/valentino/model/LLaDA-8B-Diffusion-v0.3-All")

# Create a small test - we'll just check if the function can access model structure
print("Checking model structure access...")

# Mock up a minimal test case
# Since loading the full model might be expensive, we'll just create a dummy input
# and check if the structure navigation works

try:
    model = LLaDAModelLM.from_pretrained(
        "/nas-checkpoint/valentino/model/LLaDA-8B-Diffusion-v0.3-All",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Model loaded successfully")
    
    # Create a dummy input
    tokenizer = AutoTokenizer.from_pretrained("/nas-checkpoint/valentino/model/LLaDA-8B-Diffusion-v0.3-All")
    test_input = tokenizer("Hello world", return_tensors="pt").input_ids.to(model.device)
    
    print(f"Test input shape: {test_input.shape}")
    
    # Try to extract attention - this should now work without errors
    print("Testing attention extraction...")
    attention_influence = extract_attention_influence(
        model, test_input, 
        layer_indices=[-1], 
        top_k=10
    )
    
    print(f"✓ Success! Attention influence shape: {attention_influence.shape}")
    print(f"✓ No 'Cannot find model layers' error!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")
