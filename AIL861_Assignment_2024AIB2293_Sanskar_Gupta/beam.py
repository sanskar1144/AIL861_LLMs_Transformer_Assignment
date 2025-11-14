import torch
import numpy as np
import evaluate
from tqdm import tqdm
import random
import os
import time
import json
import pandas as pd  
import config
import data_utils
from model import Transformer
import generate

def run_beam_experiment(k_values, num_prompts):
    """
    Runs the full Beam Search experiment as per Part 2.1.
    """
    print(f"Starting Beam Search experiment on device: {config.DEVICE}")

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
    # Load the model onto the correct device
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # --- 3. Get the Test Prompts ---
    print(f"Preparing {num_prompts} evaluation prompts...")
    # Set seed to get the *same* prompts as run_evaluation.py
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
    references = []
    for _ in range(num_prompts):
        max_start = len(val_corpus) - (config.EVAL_PROMPT_LENGTH + config.EVAL_MAX_TOKENS) - 1
        start_index = random.randint(0, max_start)
        
        prompt_indices = val_corpus[start_index : start_index + config.EVAL_PROMPT_LENGTH]
        reference_indices = val_corpus[start_index + config.EVAL_PROMPT_LENGTH : 
                                       start_index + config.EVAL_PROMPT_LENGTH + config.EVAL_MAX_TOKENS]
        
        prompts.append(torch.tensor(prompt_indices).unsqueeze(0).to(config.DEVICE))
        references.append(data_utils.decode(reference_indices, itow))

    # --- 4. Run Experiment ---
    bleu_metric = evaluate.load("bleu")
    results = []

    # Add Stochastic (k=0) as a baseline comparison
    # We use k=1 for greedy (beam search with width 1)
    k_values_to_test = [0, 1] + k_values 

    print(f"\nRunning experiment for k = {k_values_to_test}...")
    for k in k_values_to_test:
        total_time = 0
        total_tokens = 0
        all_predictions = []
        
        strategy_name = f"Stochastic (T={config.TEMPERATURE}, k={config.TOP_K})" if k == 0 else f"Beam Search (k={k})"
        print(f"\n--- Testing Strategy: {strategy_name} ---")
        
        with torch.no_grad():
            for i in tqdm(range(num_prompts)):
                prompt_tensor = prompts[i]
                
                start_time = time.time()
                
                if k == 0: # Use k=0 as a flag for stochastic sampling
                    generated_tensor = generate.generate_stochastic(
                        model, prompt_tensor,
                        max_tokens=config.EVAL_MAX_TOKENS,
                        eos_idx=eos_idx,
                        temperature=config.TEMPERATURE,
                        top_k=config.TOP_K
                    )
                else:
                    generated_tensor = generate.generate_beam_search(
                        model, prompt_tensor,
                        beam_width=k,
                        max_tokens=config.EVAL_MAX_TOKENS,
                        context_size=config.CONTEXT_LENGTH,
                        eos_idx=eos_idx
                    )
                
                end_time = time.time()
                total_time += (end_time - start_time)
                
                # Get generated part
                generated_indices = generated_tensor[0, config.EVAL_PROMPT_LENGTH:].cpu().tolist()
                generated_continuation = data_utils.decode(generated_indices, itow)
                
                all_predictions.append(generated_continuation)
                total_tokens += len(generated_indices)

        # Calculate metrics for this k
        avg_time_per_prompt = total_time / num_prompts
        tokens_per_second = total_tokens / total_time
        
        # Calculate BLEU score
        bleu_results = bleu_metric.compute(
            predictions=all_predictions, 
            references=[[ref] for ref in references] # Wrap each ref in a list
        )
        avg_bleu = bleu_results['bleu']
        
        results.append({
            "Strategy": strategy_name,
            "BLEU Score": avg_bleu,
            "Tokens/Second": tokens_per_second,
            "Avg. Time/Prompt (s)": avg_time_per_prompt
        })
        
        # Print a sample generation
        print(f"Sample Generation ({strategy_name}):")
        print(f"  PROMPT:    {data_utils.decode(prompts[0].squeeze(0).tolist(), itow)}")
        print(f"  MODEL:     {all_predictions[0]}")
        print(f"  REFERENCE: {references[0]}")

    # --- 5. Print Results Table ---
    print("\n--- Experiment Results ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    

def generate_with_prompt(prompt_text, k_value):
    
    # Generates text from a custom prompt using a specific beam width.
    # k_value=0 means use stochastic sampling.
    
    print(f"\n--- Generating custom prompt ---")
    
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
        
        if k_value == 0:
            strategy = f"Stochastic (T={config.TEMPERATURE}, k={config.TOP_K})"
            print(f"(Using {strategy})")
            generated_tensor = generate.generate_stochastic(
                model, prompt_tensor,
                max_tokens=40, eos_idx=eos_idx,
                temperature=config.TEMPERATURE, top_k=config.TOP_K
            )
        else:
            strategy = f"Beam Search (k={k_value})"
            print(f"(Using {strategy})")
            generated_tensor = generate.generate_beam_search(
                model, prompt_tensor,
                beam_width=k_value,
                max_tokens=40,
                context_size=config.CONTEXT_LENGTH,
                eos_idx=eos_idx
            )
            
        generated_text = data_utils.decode(generated_tensor[0].cpu().tolist(), itow)
        print(f"\nPROMPT: '{prompt_text}'")
        print(f"MODEL ({strategy}):  '{generated_text}'")


if __name__ == "__main__":
    run_beam_experiment(k_values=[5, 10], num_prompts=50)    # To Run the 5-prompt experiment for k=5 and k=10
    # # ---To Run Custom Prompt Generation ---
    # my_prompt = "The cat sat on the"
    # generate_with_prompt(my_prompt, k_value=0) # Stochastic
    # generate_with_prompt(my_prompt, k_value=1) # Greedy (Beam k=1)
    # generate_with_prompt(my_prompt, k_value=5) # Beam k=5






# # ---------------------------------------------
# print("\n--- Running Beam Search Custom Prompt Test ---")


# my_prompt = "The cat sat on the"
# k_width = 5 # This is the beam_width

# print(f"PROMPT: '{my_prompt}' (k={k_width})")

# with torch.no_grad():
#     # 2. Encode the prompt
#     prompt_tensor = data_utils.encode(my_prompt, wtoi).unsqueeze(0).to(config.DEVICE)

#     # 3. Call the beam search function
#     generated_tensor = generate.generate_beam_search(
#         model=model,
#         x_batch=prompt_tensor,
#         beam_width=k_width,
#         max_tokens=config.EVAL_MAX_TOKENS, # Or set a new value like 30
#         context_size=config.CONTEXT_LENGTH,
#         eos_idx=eos_idx
#     )

#     # 4. Decode and print the result
#     generated_text = data_utils.decode(generated_tensor[0].cpu().tolist(), itow)
#     print(f"MODEL:  '{generated_text}'")

# print("-------------------------------------------\n")


# # --- 8. Visualize Attention ---
# visualize_attention(model, wtoi, itow)


