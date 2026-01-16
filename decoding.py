import torch
import torch.nn.functional as F
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def margin_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    sorted_probs, _ = torch.sort(probabilities, dim=-1, descending=True)
    top1_probs = sorted_probs[:, :, 0]
    top2_probs = sorted_probs[:, :, 1]
    confidence = top1_probs - top2_probs
    return confidence

def entropy_function(probabilities):
    if probabilities.dim() != 3:
        raise ValueError("Input tensor 'probabilities' must be a 3D tensor with shape [batch_size, sequence_len, vocab_size]")
    epsilon = 1e-12
    probs_safe = probabilities.clone() + epsilon
    entropy = torch.sum(probabilities.clone() * torch.log(probs_safe), dim=-1)
    return entropy

@ torch.no_grad()
def decoding_default(model, prompt, steps=256, gen_length=256, block_length=256, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, attention_mask=None):
    '''
    Default decoding function from LLaDA paper
    '''
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        attention_mask: Optional attention mask for the full sequence, expected shape (B, 1, T, T).
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
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
                #logits = model(x, attention_mask=attention_mask, position_ids=position_ids).logits
                logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, steps * num_blocks

@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=256, gen_length=256, block_length=256, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=1):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe

@torch.no_grad()
def generate_with_remdm(model, prompt, gen_length=256, init_unmask_ratio=0.875, unmask_k=1, loop_steps=32, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=None, block_length=128):
    steps = 0
    assert 0.0 <= init_unmask_ratio <= 1.0, "init_unmask_ratio must be between 0 and 1"
    num_initial_tokens = gen_length * init_unmask_ratio
    assert num_initial_tokens == int(num_initial_tokens), "gen_length * init_unmask_ratio must be an integer"
    num_initial_tokens = int(num_initial_tokens)
    assert gen_length % unmask_k == 0, "gen_length must be divisible by unmask_k"
    assert num_initial_tokens % unmask_k == 0, "init_unmask_ratio * gen_length must be divisible by unmask_k"
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    num_loops = num_initial_tokens // unmask_k
    for _ in range(num_loops):
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        steps = steps+1
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        _, select_indices = torch.topk(confidence, k=unmask_k)
        x[0, select_indices] = x0[0, select_indices]
        x_result = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
    for _ in range(loop_steps):
        unmasked_gen_index = (x != mask_id) & (~prompt_index)
        num_unmasked_gen = torch.sum(unmasked_gen_index).item()
        if num_unmasked_gen == 0:
            continue
        current_remask_k = min(unmask_k, num_unmasked_gen)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits
        p = F.softmax(logits, dim=-1)
        steps = steps+1
        current_token_p = torch.gather(p, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(unmasked_gen_index, current_token_p, torch.inf)
        _, remask_indices = torch.topk(confidence, k=current_remask_k, largest=False)
        x[0, remask_indices] = mask_id
        mask_index_after_remask = (x == mask_id)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        steps = steps+1
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p_new = F.softmax(logits, dim=-1)
        x0_p_new = torch.gather(p_new, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        confidence_for_unmasking = torch.where(mask_index_after_remask, x0_p_new, -torch.inf)
        _, unmask_indices = torch.topk(confidence_for_unmasking, k=current_remask_k)
        x[0, unmask_indices] = x0[0, unmask_indices]
        x_result = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
    while (x == mask_id).any():
        mask_index = (x == mask_id)
        num_masked_left = torch.sum(mask_index).item()
        if num_masked_left == 0:
            break
        current_unmask_k = min(unmask_k, num_masked_left)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        steps = steps+1
        x0 = torch.argmax(logits_with_noise, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        _, select_indices = torch.topk(confidence, k=current_unmask_k)
        x[0, select_indices] = x0[0, select_indices]
        x_result = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
    return x ,steps

@torch.no_grad()
def generate_with_entropy(model, prompt, steps=256, gen_length=256, block_length=256, temperature=0.,
                            cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False, attention_mask=None):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if return_order:
        orders = {}

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
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

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            x0_p = entropy_function(p[:, prompt.shape[1]:])
            confidence = torch.where(mask_index[:, prompt.shape[1]:], x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index + prompt.shape[1]] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x, steps * num_blocks

@torch.no_grad()
def generate_with_margin(model, prompt, steps=256, gen_length=256, block_length=256, temperature=0.,
                            cfg_scale=0., remasking='low_confidence', mask_id=126336, return_order=False, attention_mask=None):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if return_order:
        orders = {}

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
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

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            x0_p = margin_function(p[:, prompt.shape[1]:])
            confidence = torch.where(mask_index[:, prompt.shape[1]:], x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index + prompt.shape[1]] = True
                if return_order:
                    if num_block+1 not in orders:
                        orders[num_block+1] = []
                    orders[num_block+1].append(select_index.tolist())
            x[transfer_index] = x0[transfer_index]
    if return_order:
        return x, orders
    return x, steps * num_blocks

@torch.no_grad()
def decoding_wino(model, prompt, gen_length=256, block_length=256, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9):

    device = model.device
    x_block = torch.full((1, prompt.shape[1] + gen_length + block_length), mask_id, dtype=torch.long).to(model.device)
    x_block[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x_block != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    

    for num_block in range(num_blocks):
        block_step = 0
        mask_index_block = (x_block == mask_id) # b, l
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False
        
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:,  -block_length:] = ~mask_index_block[:, prompt.shape[1] + num_block* block_length: prompt.shape[1] + (num_block + 1) * block_length]
        position_ids = torch.cat([torch.arange(prompt.shape[1] + gen_length, device=device), torch.arange(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length, device=device)])
        attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = torch.ones(block_length, block_length, dtype=torch.bool).to(device)
        attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~torch.eye(block_length, dtype=torch.bool).to(device)
        last_accept = 30
        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits # b, l, vocab_size
            #logits = model(x_block, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1) # b, l, vocab_size
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            x0 = torch.where(mask_index_block, x0, x_block) # replace the masked tokens with the predicted tokens
            confidence = torch.where(mask_index_block, x0_p, -np.inf) # keep the confidence of the masked tokens
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                # get top max_accept tokens
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            
            # always transfer the max confidence token
            else:
                if not transfer_index.any():
                    max_confidence_index = torch.argmax(confidence)
                    transfer_index.view(-1)[max_confidence_index] = True
            x_block[transfer_index] = x0[transfer_index]
            
            num_accept = transfer_index.sum()
            
            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)
            
            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            last_accept = num_accept

        step += block_step

    return x_block[:, :prompt.shape[1] + gen_length], step

@torch.no_grad()
def generate_with_saber(model, prompt,n = 2,mu = 8, gen_length=256, block_length=256, temperature=0., mask_id=126336, attention_mask=None):

    step = 0
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    mask_index = (x == mask_id) 
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    global_transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    initial_confidence = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    final_confidence = torch.full_like(x, fill_value=-np.inf,dtype=torch.float32)
    last_confidence = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    for num_block in range(num_blocks):
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_start = prompt.shape[1] + num_block * block_length
        for i in range(1024):

            step += 1
            logits = model(x, attention_mask=attention_mask).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) 

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            confidence_delta = confidence - last_confidence
            mask_finite = torch.isfinite(confidence_delta)
            confidence_delta = torch.where(mask_finite, confidence_delta, torch.full_like(confidence_delta, float('inf')))
            last_confidence = confidence
            confidence = torch.where(global_transfer_index, -np.inf, confidence)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            fix_list = []
            for j in range(confidence.shape[0]):
                block_final_confidence = final_confidence[j, :block_end]
                #block_final_confidence = final_confidence[j, block_start:block_end] 
                generated_probs = block_final_confidence[block_final_confidence > -np.inf]
                if generated_probs.numel() > 0:
                    threshold = generated_probs.mean()
                else:
                    threshold = 1
                select_index = torch.where(confidence[j] > threshold)[0]
                if select_index.numel() < n:
                    _, select_index = torch.topk(confidence[j], k=n)

                fix_list.append(max(n, select_index.numel()))

                transfer_index[j, select_index] = True
                global_transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            
            new_generated_mask = transfer_index & (initial_confidence == 0.0)
            if new_generated_mask.any():
                initial_confidence = torch.where(new_generated_mask, x0_p, initial_confidence)
            
            final_confidence = torch.where(transfer_index, x0_p, final_confidence)

            all_masked_transferred = torch.all(global_transfer_index[mask_index]).item()
            
            if all_masked_transferred:
                return x, step   

            if torch.all(x[:, :block_end] != mask_id):
                break  
            delta_for_removal = confidence_delta.clone()

            delta_for_removal = torch.where(mask_index, delta_for_removal, torch.full_like(delta_for_removal, float('inf')))
            if block_end < delta_for_removal.shape[1]:
                delta_for_removal[:, block_end:] = float('inf')
            positions_mask_now = (x == mask_id)
            delta_for_removal[positions_mask_now] = float('inf')
            delta_for_removal[prompt_index] = float('inf')

            for j in range(delta_for_removal.shape[0]):
                num_to_remask = max(int(n/2),(fix_list[j] + mu-1) // mu)

                if num_to_remask > fix_list[j]-1:
                    num_to_remask = fix_list[j]-1

                _, remove_index = torch.topk(delta_for_removal[j], k=num_to_remask, largest=False)
                x[j, remove_index] = mask_id  
                initial_confidence[j, remove_index] = 0.0
                global_transfer_index[j, remove_index] = False
    return x, step

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
        
def get_num_transfer_tokens_ours(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base*2

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

@torch.no_grad()
def generate_with_temporal_decay(
    model, 
    prompt, 
    steps=64, 
    gen_length=64, 
    block_length=64, 
    temperature=0.0,
    cfg_scale=0., 
    remask_budget=0.0, 
    alpha_decay=0.0, 
    mask_id=126336, 
    attention_mask=None
):
    '''
    Generalized Sampling with Temporal Decay and Budgeted Remasking.
    '''
    # Initialize
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    prompt_index = (x != mask_id)
    
    # Fix Time Tracking: Initialize with -1
    fix_time = torch.full_like(x, -1)
    # Mark prompt as fixed at step -999 (permanent)
    fix_time[:, :prompt.shape[1]] = -999 

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    
    total_steps = steps * num_blocks
    current_step = 0

    # Logging Init
    # Track x0 prediction history for Stability Score [Steps, Gen_Length]
    x0_history = []
    
    # Token Log: Key=Index, Val={fix_step, fix_token, fix_conf, remask_events: [(step, prev_token, new_conf)]}
    # Initialize for the full sequence length (Prompt + Gen)
    L_total = prompt.shape[1] + gen_length
    token_logs = {i: {'fix_step': -1, 'fix_token': -1, 'fix_conf': 0.0, 'remask_events': []} for i in range(L_total)}

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        # Calculate schedule for this block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            current_step += 1
            mask_index = (x == mask_id)
            
            # --- Predict ---
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

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Capture x0 for stability analysis (batch 0 only for simplicity as metrics assume list/single)
            x0_history.append(x0[0].cpu().clone())

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            
            # Prevent lookahead
            x0_p[:, block_end:] = -np.inf
            
            # --- Transfer (Forward) ---
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            # Batched transfer
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i]
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
                
            x[transfer_index] = x0[transfer_index]
            fix_time[transfer_index] = current_step
            
            # Log Fixes (for batch 0)
            if transfer_index[0].any():
                fixed_indices = torch.nonzero(transfer_index[0], as_tuple=True)[0]
                for idx in fixed_indices.cpu().numpy():
                    token_logs[idx]['fix_step'] = current_step
                    token_logs[idx]['fix_token'] = x0[0, idx].item()
                    token_logs[idx]['fix_conf'] = x0_p[0, idx].item()

            # --- Remasking (Backward Correction) ---
            if remask_budget > 0.0:
                # 1. Identify Candidates: Generated tokens (not prompt, not mask)
                generated_mask = (x != mask_id) & (fix_time != -999)
                
                if generated_mask.any():
                    # 2. Calculate Stability Score
                    # Score = Confidence + Alpha * Age
                    # Age = current_step - fix_time
                    ages = current_step - fix_time
                    stability_score = x0_p + alpha_decay * ages.float()
                    
                    # Mask out non-candidates (score -> inf)
                    candidate_scores = torch.where(generated_mask, stability_score, torch.tensor(float('inf'), device=x.device))
                    
                    # 3. Budget Selection per batch
                    remask_mask = torch.zeros_like(x, dtype=torch.bool)
                    for b in range(x.shape[0]):
                         # Count total generated tokens for this batch
                        n_gen = generated_mask[b].sum().item()
                        if n_gen > 0:
                            budget_k = max(1, int(n_gen * remask_budget))
                            
                            # Select lowest scores
                            # Limit k to actual candidates
                            valid_k = min(budget_k, n_gen)
                            
                            _, bottom_indices = torch.topk(candidate_scores[b], k=valid_k, largest=False)
                            remask_mask[b, bottom_indices] = True
                    
                    # Log Remasks before applying (batch 0)
                    if remask_mask[0].any():
                        remasked_indices = torch.nonzero(remask_mask[0], as_tuple=True)[0]
                        for idx in remasked_indices.cpu().numpy():
                            # Log: (step, prev_token, new_conf_at_remask_time)
                            # Actually we track remask events. 
                            # The metric typically looks at: was it remasked? Did it change?
                            token_logs[idx]['remask_events'].append((current_step, x[0, idx].item(), stability_score[0, idx].item()))

                    # Apply Remask
                    x[remask_mask] = mask_id
                    fix_time[remask_mask] = -1

    return x, {'x0_history': x0_history, 'token_logs': token_logs, 'total_steps': total_steps}


@torch.no_grad()
def baseline_sampling(
    model,
    tokenizer,
    prompt_text,
    steps=64,
    gen_length=64,
    block_length=64,
    temperature=0.0,
):
    """
    Baseline: Standard Iterative Decoding (No Remasking)
    """
    # Init
    mask_id = 126336
    if prompt_text:
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(
            model.device
        )
    else:
        prompt_tokens = torch.tensor([[]], dtype=torch.long, device=model.device)

    B, L_prompt = prompt_tokens.shape
    x = torch.full(
        (B, L_prompt + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :L_prompt] = prompt_tokens

    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    history = []
    start_time = time.time()
    nfe = 0  # Number of Function Evaluations (Forward passes)

    for num_block in range(num_blocks):
        block_start = L_prompt + num_block * block_length
        block_end = L_prompt + (num_block + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            logits = model(x).logits
            nfe += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)

            mask_index = x == mask_id
            confidence = torch.where(mask_index, x0_p, -np.inf)

            if i < num_transfer_tokens.shape[1]:
                k = num_transfer_tokens[0, i].item()
            else:
                k = 0

            # Standard Transfer: Top-k confidence tokens are unmasked
            top_values, top_indices = torch.topk(confidence[0], k=k)
            transfer_mask = torch.zeros_like(x, dtype=torch.bool)
            transfer_mask[0, top_indices] = True
            x[transfer_mask] = x0[transfer_mask]

            # Logging
            history.append(
                {
                    "step": i,
                    "block": num_block,
                    "nfe": nfe,
                    "avg_confidence": x0_p.mean().item(),
                    "text": tokenizer.decode(x[0], skip_special_tokens=True),
                }
            )

    total_time = time.time() - start_time
    return x, history, {"time": total_time, "nfe": nfe}

@torch.no_grad()
def inspect_sampling(
    model,
    tokenizer,
    prompt_text,
    steps=64,
    gen_length=64,
    block_length=64,
    temperature=0.0,
):
    """
    Inspection: Detailed logging of the sampling process.
    Returns detailed history including token ids, predicted ids, and confidence at each step.
    """
    mask_id = 126336
    if prompt_text:
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(
            model.device
        )
    else:
        prompt_tokens = torch.tensor([[]], dtype=torch.long, device=model.device)

    B, L_prompt = prompt_tokens.shape
    x = torch.full(
        (B, L_prompt + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :L_prompt] = prompt_tokens

    num_blocks = max(1, gen_length // block_length)
    steps = steps // num_blocks

    detailed_history = []
    nfe = 0

    for num_block in range(num_blocks):
        block_start = L_prompt + num_block * block_length
        block_end = L_prompt + (num_block + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            logits = model(x).logits
            nfe += 1

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)

            mask_index = x == mask_id
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Capture State BEFORE update
            step_info = {
                "step": i + num_block * steps,
                "nfe": nfe,
                "x_curr": x.clone().cpu(),
                "x0_pred": x0.clone().cpu(),
                "confidence": x0_p.clone().cpu(),
                "mask_mask": mask_index.clone().cpu(),
            }
            detailed_history.append(step_info)

            if i < num_transfer_tokens.shape[1]:
                k = num_transfer_tokens[0, i].item()
            else:
                k = 0

            top_values, top_indices = torch.topk(confidence[0], k=k)
            transfer_mask = torch.zeros_like(x, dtype=torch.bool)
            transfer_mask[0, top_indices] = True
            x[transfer_mask] = x0[transfer_mask]

    return x, detailed_history


# ============================================================================
# Graph-Aware Historical Remasking Decoder - New Implementation
# ============================================================================

def calculate_uncertainty(logits, x0, method='neg_log_prob'):
    """
    토큰 예측의 불확실성을 계산합니다.
    
    Args:
        logits: 모델 로짓 [B, L, V]
        x0: 예측된 토큰 [B, L]
        method: 'neg_log_prob', 'entropy', 'margin' 중 선택
        
    Returns:
        uncertainty: [B, L] 불확실성 점수 (높을수록 불확실)
    """
    p = F.softmax(logits.to(torch.float64), dim=-1)
    
    if method == 'neg_log_prob':
        # -log(p(x0)): 선택된 토큰의 음의 로그 확률
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        uncertainty = -torch.log(x0_p + 1e-12)
    elif method == 'entropy':
        # H(p) = -Σ p*log(p)
        uncertainty = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
    elif method == 'margin':
        # 1 - (p1 - p2): top-2 확률 차이의 역수
        sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
        margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
        uncertainty = 1.0 - margin
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
    
    return uncertainty


def extract_attention_influence(model, x, attention_mask=None, layer_indices=None, top_k=10):
    """
    모델 forward pass에서 attention 기반 영향력을 추출합니다.
    
    LLaDA는 output_attentions를 지원하지 않으므로, 
    Q, K를 직접 계산하여 attention weights를 구합니다.
    
    Args:
        model: LLaDA 모델
        x: 입력 토큰 시퀀스 [B, L]
        attention_mask: Optional attention mask
        layer_indices: 사용할 레이어 인덱스 (None = 마지막 레이어만)
        top_k: 각 위치당 유지할 top-k attention weights
        
    Returns:
        influence_matrix: [B, L, L] sparse influence scores (j -> i)
                         influence_matrix[b, j, i] = token j가 token i에게 미친 영향
    """
    device = x.device
    B, L = x.shape
    
    # Attention weights를 저장할 텐서 초기화
    influence_matrix = torch.zeros(B, L, L, device=device, dtype=torch.float32)
    
    try:
        # 레이어 선택
        if layer_indices is None:
            layer_indices = [-1]  # 마지막 레이어만
        
        # 모델의 레이어에 접근
        # LLaDA 구조: model.model.transformer.blocks or model.model.transformer.block_groups
        # 또는 model.transformer.blocks or model.transformer.block_groups (direct LLaDAModel)
        layers = []
        transformer = None
        
        # Try to access transformer - handle both LLaDAModelLM wrapper and direct LLaDAModel
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            # LLaDAModelLM wrapper case
            transformer = model.model.transformer
        elif hasattr(model, 'transformer'):
            # Direct LLaDAModel case
            transformer = model.transformer
        else:
            raise AttributeError("Cannot find transformer in model structure")
        
        # Extract layers from transformer
        if hasattr(transformer, 'blocks'):
            # Simple blocks list
            layers = list(transformer.blocks)
        elif hasattr(transformer, 'block_groups'):
            # Block groups - need to extract blocks from each group
            for group in transformer.block_groups:
                if hasattr(group, 'blocks'):
                    layers.extend(list(group.blocks))
                else:
                    # group itself might be a block
                    layers.append(group)
        else:
            raise AttributeError("Cannot find blocks or block_groups in transformer")
        
        if len(layers) == 0:
            raise AttributeError("No layers found in transformer")
        
        # Embedding 계산
        if hasattr(transformer, 'wte'):
            hidden_states = transformer.wte(x)
        else:
            raise AttributeError("Cannot find embedding layer (wte)")
        
        # 선택된 레이어들에 대해 attention 계산
        attention_weights_list = []
        
        for layer_idx in layer_indices:
            if layer_idx < 0:
                layer_idx = len(layers) + layer_idx
            
            if layer_idx >= len(layers) or layer_idx < 0:
                continue
                
            layer = layers[layer_idx]
            
            # Layer의 attention 부분만 실행
            # LLaDASequentialBlock 구조 가정
            if hasattr(layer, 'attn_norm'):
                x_normed = layer.attn_norm(hidden_states)
            else:
                x_normed = hidden_states
            
            # Q, K, V 계산
            if hasattr(layer, 'att_proj'):
                qkv = layer.att_proj(x_normed)
                # Split into Q, K, V
                q, k, v = qkv.split(layer.fused_dims, dim=-1)
            else:
                raise AttributeError("Cannot find attention projection (att_proj)")
            
            # Reshape for multi-head attention
            # q: [B, L, d_model] -> [B, num_heads, L, head_dim]
            num_heads = layer.config.n_heads
            head_dim = layer.config.d_model // num_heads
            
            q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, layer.config.effective_n_kv_heads, head_dim).transpose(1, 2)
            
            # GQA 처리
            if num_heads != layer.config.effective_n_kv_heads:
                k = k.repeat_interleave(num_heads // layer.config.effective_n_kv_heads, dim=1)
            
            # Attention weights 계산: softmax(Q @ K^T / sqrt(d_k))
            # q: [B, num_heads, L, head_dim]
            # k: [B, num_heads, L, head_dim]
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, L, L]
            
            # Attention mask 적용 (있는 경우)
            if attention_mask is not None:
                # attention_mask: [B, 1, L, L] 형태 가정
                # Mask 값이 True인 곳은 attend 가능, False는 -inf로 마스킹
                mask_value = torch.finfo(attn_weights.dtype).min
                # attention_mask가 boolean이면 inverse해서 사용
                if attention_mask.dtype == torch.bool:
                    # True = attend 가능, False = mask
                    attn_mask_expanded = ~attention_mask
                else:
                    attn_mask_expanded = attention_mask == 0
                attn_weights = attn_weights.masked_fill(attn_mask_expanded, mask_value)
            
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, num_heads, L, L]
            
            # 헤드 평균
            attn_weights = attn_weights.mean(dim=1)  # [B, L, L]
            
            attention_weights_list.append(attn_weights)
        
        # 레이어 평균
        if attention_weights_list:
            influence_matrix = torch.stack(attention_weights_list).mean(dim=0)
        else:
            # Fallback: uniform attention
            print("Warning: No attention weights extracted. Using uniform approximation.")
            influence_matrix = torch.ones(B, L, L, device=device) / L
            
    except Exception as e:
        # 에러 발생 시 fallback: uniform attention
        print(f"Warning: Could not extract attention weights ({e}). Using uniform approximation.")
        import traceback
        traceback.print_exc()
        influence_matrix = torch.ones(B, L, L, device=device) / L
    
    # Top-k만 유지하여 sparse하게 만들기
    if top_k < L:
        for b in range(B):
            for j in range(L):
                # j번째 토큰이 참조한 토큰들 중 top-k만 유지
                values, indices = torch.topk(influence_matrix[b, j, :], k=min(top_k, L))
                mask = torch.zeros(L, device=device, dtype=torch.bool)
                mask[indices] = True
                influence_matrix[b, j, ~mask] = 0.0
    
    return influence_matrix


def propagate_responsibility(
    responsibility,
    attention_influence,
    uncertainty,
    mask_index,
    gamma_decay=0.95,
    distance_type='token'
):
    """
    Attention 그래프를 통해 불확실성을 역전파하여 책임도를 업데이트합니다.
    
    Args:
        responsibility: 현재 책임도 점수 [B, L]
        attention_influence: Attention weights [B, L, L] (j -> i)
        uncertainty: 현재 예측의 불확실성 [B, L]
        mask_index: 현재 마스크된 위치 [B, L]
        gamma_decay: 거리 감쇠 인자
        distance_type: 'token' (토큰 거리) 또는 'none' (감쇠 없음)
        
    Returns:
        updated_responsibility: [B, L]
    """
    B, L = responsibility.shape
    device = responsibility.device
    
    updated_responsibility = responsibility.clone()
    
    # 불확실한 토큰들에 대해서만 전파
    # 마스크되지 않은 토큰 중 불확실성이 높은 것들
    uncertain_mask = (~mask_index) & (uncertainty > uncertainty.median())
    
    for b in range(B):
        for j in range(L):
            if not uncertain_mask[b, j]:
                continue
            
            # j번째 토큰의 불확실성
            unc_j = uncertainty[b, j]
            
            # j가 참조한 토큰들 (attention_influence[b, j, i] > 0인 i들)
            attended_indices = torch.nonzero(attention_influence[b, j, :] > 0, as_tuple=True)[0]
            
            for i in attended_indices:
                # Attention weight
                attn_weight = attention_influence[b, j, i]
                
                # 거리 계산
                if distance_type == 'token':
                    distance = abs(j - i)
                    decay = gamma_decay ** distance
                else:
                    decay = 1.0
                
                # 책임도 전파: i가 j의 불확실성에 기여한 정도
                contribution = attn_weight * unc_j * decay
                updated_responsibility[b, i] += contribution
    
    return updated_responsibility


@torch.no_grad()
def decoding_graph_remask(
    model, 
    prompt, 
    gen_length=256, 
    block_length=256, 
    temperature=0., 
    mask_id=126336,
    threshold_forward=0.6,      # Accept 임계값
    threshold_back=0.9,         # Remask 임계값 (confidence)
    resp_threshold=0.3,         # Remask 임계값 (responsibility)
    gamma_decay=0.95,           # 거리 감쇠 인자
    use_attention_layers=[-1],  # 사용할 attention 레이어
    top_k_attention=10,         # Top-k attention weights
    max_remask_ratio=0.3        # 최대 remask 비율
):
    """
    Graph-aware historical remasking decoder.
    
    핵심 아이디어:
    - Attention 기반으로 토큰 간 영향력(influence) 추적
    - 불확실한 토큰이 참조한 과거 토큰들에게 책임도(responsibility) 전파
    - Local confidence와 historical responsibility를 모두 고려하여 remask
    
    Args:
        model: LLaDA 모델
        prompt: 프롬프트 토큰 [B, L_prompt]
        gen_length: 생성할 길이
        block_length: 블록 길이
        temperature: 샘플링 온도
        mask_id: 마스크 토큰 ID
        threshold_forward: 토큰 accept 임계값
        threshold_back: Confidence 기반 remask 임계값
        resp_threshold: Responsibility 기반 remask 임계값
        gamma_decay: 거리 감쇠 인자
        use_attention_layers: 사용할 레이어 인덱스
        top_k_attention: 유지할 top-k attention
        max_remask_ratio: 최대 remask 비율
        
    Returns:
        x: 생성된 시퀀스 [B, L_prompt + gen_length]
        stats: 통계 정보 딕셔너리
    """
    device = model.device
    B = prompt.shape[0]
    L_prompt = prompt.shape[1]
    
    # 초기화
    x_block = torch.full((B, L_prompt + gen_length + block_length), mask_id, dtype=torch.long, device=device)
    x_block[:, :L_prompt] = prompt.clone()
    
    prompt_index = (x_block != mask_id)
    
    # Responsibility 텐서 초기화
    responsibility = torch.zeros_like(x_block, dtype=torch.float32, device=device)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    total_steps = 0
    total_remasks = 0
    
    # 통계 수집
    stats = {
        'steps_per_block': [],
        'remask_counts': [],
        'avg_responsibility': []
    }
    
    for num_block in range(num_blocks):
        block_start = L_prompt + num_block * block_length
        block_end = L_prompt + (num_block + 1) * block_length
        
        block_step = 0
        mask_index_block = (x_block == mask_id)
        mask_index_block[:, block_end:] = False
        
        # 이미 unmask된 토큰 추적
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:, -block_length:] = ~mask_index_block[:, block_start:block_end]
        
        # Position IDs 설정 (WINO와 동일)
        position_ids = torch.cat([
            torch.arange(L_prompt + gen_length, device=device), 
            torch.arange(block_start, block_end, device=device)
        ])
        
        # Attention mask 설정 (WINO와 동일)
        attention_mask = torch.ones(B, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool, device=device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = torch.ones(block_length, block_length, dtype=torch.bool, device=device)
        attention_mask[:, :, -block_length:, block_start:block_end] = ~torch.eye(block_length, dtype=torch.bool, device=device)
        
        last_accept = 30
        
        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            
            # Forward pass with attention extraction
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # 이미 unmask된 토큰은 유지
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, block_start:block_end] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]
            
            # Confidence 계산
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            
            x0 = torch.where(mask_index_block, x0, x_block)
            confidence = torch.where(mask_index_block, x0_p, -np.inf)
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            
            # === Graph-Aware Component: Attention 추출 및 Responsibility 전파 ===
            try:
                # Attention influence 추출
                attention_influence = extract_attention_influence(
                    model, x_block, attention_mask=attention_mask, 
                    layer_indices=use_attention_layers, top_k=top_k_attention
                )
                
                # Uncertainty 계산
                uncertainty = calculate_uncertainty(logits, x0, method='neg_log_prob')
                
                # Responsibility 전파
                responsibility = propagate_responsibility(
                    responsibility, attention_influence, uncertainty,
                    mask_index_block, gamma_decay=gamma_decay
                )
            except Exception as e:
                # Attention 추출 실패 시 기본 동작
                print(f"Warning: Attention extraction failed ({e}), using confidence-only mode")
            
            # === Transfer (Accept) ===
            transfer_index = confidence > threshold_forward
            if transfer_index.sum() > max_accept:
                # Top max_accept개만 선택
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            else:
                # 최소 1개는 transfer
                if not transfer_index.any():
                    max_confidence_index = torch.argmax(confidence)
                    transfer_index.view(-1)[max_confidence_index] = True
            
            x_block[transfer_index] = x0[transfer_index]
            num_accept = transfer_index.sum()
            
            # === Remask (Backward Correction) ===
            remask_index = torch.zeros_like(transfer_index)
            
            if num_accept > 1:
                # Condition 1: Low confidence (기존 방식)
                low_conf_mask = confidence_back < threshold_back
                
                # Condition 2: High responsibility (새로운 방식)
                # Unmask된 토큰 중에서만 고려
                resp_scores = torch.where(unmask_index_block, responsibility, -np.inf)
                high_resp_mask = resp_scores > resp_threshold
                
                # 두 조건 결합
                remask_candidates = low_conf_mask | high_resp_mask
                
                # Budget 제한
                max_remask = min(int(unmask_index_block.sum() * max_remask_ratio), last_accept - 1)
                
                if remask_candidates.sum() >= max_remask:
                    # Priority: responsibility * (1 if low_conf else 0.5)
                    priority = responsibility.clone()
                    priority[low_conf_mask] *= 2.0  # Low confidence에 더 높은 우선순위
                    priority[~unmask_index_block] = -np.inf
                    
                    _, indices = torch.topk(priority.view(-1), k=max_remask, largest=True)
                    temp_mask = torch.zeros_like(priority.view(-1), dtype=torch.bool)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(remask_candidates.shape)
                else:
                    remask_index = remask_candidates
            
            # Remask 적용
            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, block_start:block_end] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            
            # Remask된 토큰의 responsibility 리셋
            responsibility[remask_index_shift] = 0.0
            
            # 인덱스 업데이트
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, block_start:block_end]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            
            last_accept = num_accept
            block_step += 1
            total_steps += 1
            total_remasks += remask_index.sum().item()
        
        stats['steps_per_block'].append(block_step)
        stats['remask_counts'].append(total_remasks)
        stats['avg_responsibility'].append(responsibility.mean().item())
    
    stats['total_steps'] = total_steps
    stats['total_remasks'] = total_remasks
    
    return x_block[:, :L_prompt + gen_length], stats
