import torch
import numpy as np
from tqdm import tqdm
import random
import os
import time
import json
import pandas as pd  # Make sure pandas is in your environment.yml or run: pip install pandas

import config
import data_utils
from model import Transformer
import generate

def run_kv_experiment(num_prompts=20, max_gen_tokens=50):
    """
    Runs the full KV Cache experiment
    Compares runtime of generate_stochastic (with cache) vs generate_stochastic_no_cache (baseline).
    """
    print(f"Starting KV Cache experiment on device: {config.DEVICE}")

    # --- 1. Load Vocab ---
    print(f"Loading vocabulary from {config.VOCAB_SAVE_PATH}...")
    with open(config.VOCAB_SAVE_PATH, 'r') as f:
        vocab_data = json.load(f)
    wtoi = vocab_data['wtoi']
    vocab_list = vocab_data['vocab_list']
    itow = {int(k): v for k, v in vocab_data['itow'].items()}
    vocab_size = len(wtoi)
    eos_idx = wtoi.get('<eos>')
    if eos_idx is None:
        raise ValueError("<eos> token not found in vocabulary!")
    print(f"Vocabulary size: {vocab_size}")

    # --- 2. Load Model ---
    print("Loading trained model...")
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        head_dim=config.HEAD_DIM,
        ff_hidden_dim=config.FF_HIDDEN_DIM,
        context_size=config.CONTEXT_LENGTH,
        vocab_list=vocab_list
    )
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # --- 3. Get the Test Prompts ---
    print(f"Preparing {num_prompts} evaluation prompts...")
    # Set seed to get the *same* 20 prompts every time
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    _, _, official_val_texts = data_utils.load_and_split_data(
        config.DATASET_NAME,
        config.DATA_SUBSET,
        config.VAL_SET_SIZE
    )
    val_corpus = data_utils.get_data_for_batching(official_val_texts, wtoi)

    prompts = []
    for _ in range(num_prompts):
        max_start = len(val_corpus) - (config.EVAL_PROMPT_LENGTH + max_gen_tokens) - 1
        start_index = random.randint(0, max_start)
        prompt_indices = val_corpus[start_index : start_index + config.EVAL_PROMPT_LENGTH]
        prompts.append(torch.tensor(prompt_indices).unsqueeze(0).to(config.DEVICE))

    # --- 4. Run Experiment ---
    results = []

    # --- Test WITH KV CACHE ---
    print("\n--- Testing WITH KV Cache (generate_stochastic) ---")
    total_time_with_cache = 0
    total_tokens_with_cache = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_prompts)):
            prompt_tensor = prompts[i]
            
            start_time = time.time()
            generated_tensor = generate.generate_stochastic(
                model, prompt_tensor,
                max_tokens=max_gen_tokens,
                eos_idx=eos_idx,
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K
            )
            end_time = time.time()
            
            total_time_with_cache += (end_time - start_time)
            total_tokens_with_cache += (generated_tensor.shape[1] - config.EVAL_PROMPT_LENGTH)

    tokens_sec_with_cache = total_tokens_with_cache / total_time_with_cache
    results.append({
        "Strategy": "With KV Cache",
        "Tokens/Second": tokens_sec_with_cache,
        "Total Time (s)": total_time_with_cache
    })

    # --- Test WITHOUT KV CACHE ---
    print("\n--- Testing WITHOUT KV Cache (generate_stochastic_no_cache) ---")
    total_time_no_cache = 0
    total_tokens_no_cache = 0

    with torch.no_grad():
        for i in tqdm(range(num_prompts)):
            prompt_tensor = prompts[i]
            
            start_time = time.time()
            generated_tensor = generate.generate_stochastic_no_cache(
                model, prompt_tensor,
                max_tokens=max_gen_tokens,
                eos_idx=eos_idx,
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K
            )
            end_time = time.time()
            
            total_time_no_cache += (end_time - start_time)
            total_tokens_no_cache += (generated_tensor.shape[1] - config.EVAL_PROMPT_LENGTH)

    tokens_sec_no_cache = total_tokens_no_cache / total_time_no_cache
    results.append({
        "Strategy": "Without KV Cache",
        "Tokens/Second": tokens_sec_no_cache,
        "Total Time (s)": total_time_no_cache
    })

    # --- 5. Print Results Table ---
    print("\n--- Experiment Results ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n--- Speedup Analysis ---")
    speedup = tokens_sec_with_cache / tokens_sec_no_cache
    print(f"KV Caching provided a {speedup:.2f}x speedup.")

def generate_with_prompt_kv(prompt_text, use_cache):
    """
    Generates text from a custom prompt, with or without KV cache.
    """
    strategy = "With KV Cache" if use_cache else "Without KV Cache"
    print(f"\n--- Generating custom prompt ({strategy}) ---")
    
    # --- 1. Load Vocab ---
    with open(config.VOCAB_SAVE_PATH, 'r') as f:
        vocab_data = json.load(f)
    wtoi = vocab_data['wtoi']
    itow = {int(k): v for k, v in vocab_data['itow'].items()}
    vocab_list = vocab_data['vocab_list']
    vocab_size = len(wtoi)
    eos_idx = wtoi.get('<eos>')

    # --- 2. Load Model ---
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        head_dim=config.HEAD_DIM,
        ff_hidden_dim=config.FF_HIDDEN_DIM,
        context_size=config.CONTEXT_LENGTH,
        vocab_list=vocab_list
    )
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # --- 3. Generate ---
    with torch.no_grad():
        prompt_tensor = data_utils.encode(prompt_text, wtoi).unsqueeze(0).to(config.DEVICE)
        
        start_time = time.time()
        if use_cache:
            generated_tensor = generate.generate_stochastic(
                model, prompt_tensor,
                max_tokens=40, eos_idx=eos_idx,
                temperature=config.TEMPERATURE, top_k=config.TOP_K
            )
        else:
            generated_tensor = generate.generate_stochastic_no_cache(
                model, prompt_tensor,
                max_tokens=40, eos_idx=eos_idx,
                temperature=config.TEMPERATURE, top_k=config.TOP_K
            )
        end_time = time.time()
            
        generated_text = data_utils.decode(generated_tensor[0].cpu().tolist(), itow)
        print(f"\nPROMPT: '{prompt_text}'")
        print(f"MODEL:  '{generated_text}'")
        print(f"(Time taken: {end_time - start_time:.4f} seconds)")


if __name__ == "__main__":
    # Run the 20-prompt experiment
    run_kv_experiment(num_prompts=20, max_gen_tokens=50)
    
    # my_prompt = "Once upon a time in a land far away"
    # generate_with_prompt_kv(my_prompt, use_cache=True)
    # generate_with_prompt_kv(my_prompt, use_cache=False)