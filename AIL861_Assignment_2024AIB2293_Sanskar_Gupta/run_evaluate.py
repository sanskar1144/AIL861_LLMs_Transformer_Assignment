import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm import tqdm
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import config
import data_utils
from model import Transformer
import generate

def visualize_attention(model, wtoi, itow):
    """
    Visualizes attention maps for a sample sentence.
    """
    print("\nVisualizing attention...")
    os.makedirs(config.ATTN_PLOT_DIR, exist_ok=True)
    
    # Use a sample prompt
    prompt = "Once upon a time, there"
    x = data_utils.encode(prompt, wtoi).unsqueeze(0).to(config.DEVICE)
    labels = [itow.get(int(tok.item()), str(tok.item())) for tok in x[0]]
    
    model.eval()
    with torch.no_grad():
        # Use the special forward pass to capture attention
        _, captured_attns = model.forward_with_attn_capture(x, config.NUM_HEADS)

    # Visualize each head
    for head_idx, attn in enumerate(captured_attns):
        if attn is None: 
            continue
        
        # [1, T, T] -> [T, T]
        attn_map = attn.squeeze(0).cpu().numpy() 

        plt.figure(figsize=(10, 10))
        sns.heatmap(attn_map, xticklabels=labels, yticklabels=labels, annot=False)
        plt.title(f"Attention Head {head_idx}", fontsize=16)
        plt.xlabel("Key")
        plt.ylabel("Query")
        plt.tight_layout()
        save_path = os.path.join(config.ATTN_PLOT_DIR, f"attn_heatmap_head{head_idx}.png")
        plt.savefig(save_path)
        plt.close()

    print(f" Saved attention visualizations to '{config.ATTN_PLOT_DIR}'")

    # Write code to run inference and print output on the prompt
    with torch.no_grad():
        generated_tensor = generate.generate_stochastic(
            model,
            x,
            max_tokens=40,
            eos_idx=wtoi.get('<eos>'),
            temperature=1.0,
            top_k=10
        )
        generated_text = data_utils.decode(generated_tensor[0].cpu().tolist(), itow)
        print(f"\nPROMPT: '{prompt}'")
        print(f"MODEL:  '{generated_text}'\n")

def main():
    
    # --- ADD THIS BLOCK ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # --- END OF BLOCK ---

    print(f"Starting evaluation on device: {config.DEVICE}")

    # --- 1. Load Data and Vocab ---
    print(f"Loading vocabulary from {config.VOCAB_SAVE_PATH}...")
    with open(config.VOCAB_SAVE_PATH, 'r') as f:
        vocab_data = json.load(f)
    
    wtoi = vocab_data['wtoi']
    vocab_list = vocab_data['vocab_list']
    # CRITICAL: JSON saves integer keys as strings. We must convert them back.
    itow_str_keys = vocab_data['itow']
    itow = {int(k): v for k, v in itow_str_keys.items()}
    
    vocab_size = len(wtoi)
    print(f"Vocabulary size: {vocab_size}")

    # --- GET THE EOS INDEX ---
    eos_idx = wtoi.get('<eos>')
    if eos_idx is None:
        raise ValueError("<eos> token not found in vocabulary!")
    # --- END OF NEW BLOCK --

    # Now, load the *official* validation data
    print("Loading official validation data...")
    # We call the *same* function but only use the 3rd return value
    _, _, official_val_texts = data_utils.load_and_split_data(
        config.DATASET_NAME,
        config.DATA_SUBSET,
        config.VAL_SET_SIZE
    )
    val_corpus = data_utils.get_data_for_batching(official_val_texts, wtoi)
    print(f"Loaded official validation set: {len(val_corpus)} tokens")

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
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.to(config.DEVICE)
    model.eval()

    # --- 3. Run BLEU and Perplexity Evaluation ---
    print(f"Running evaluation on {config.EVAL_SAMPLES} samples...")
    bleu_metric = evaluate.load("bleu")
    perplexities = []
    bleu_scores = []
    
    generation_samples = []

    with torch.no_grad():
        for _ in tqdm(range(config.EVAL_SAMPLES)):
            # 1. Get a random sample
            max_start = len(val_corpus) - (config.EVAL_PROMPT_LENGTH + config.EVAL_MAX_TOKENS) - 1
            start_index = random.randint(0, max_start)
            
            # 2. Get prompt and reference
            prompt_indices = val_corpus[start_index : start_index + config.EVAL_PROMPT_LENGTH]
            reference_indices = val_corpus[start_index + config.EVAL_PROMPT_LENGTH : 
                                           start_index + config.EVAL_PROMPT_LENGTH + config.EVAL_MAX_TOKENS]
            reference_continuation = data_utils.decode(reference_indices, itow)
            
            prompt_tensor = torch.tensor(prompt_indices).unsqueeze(0).to(config.DEVICE)

            # 3. Generate continuation
            generated_tensor = generate.generate_stochastic(
                model,
                prompt_tensor,
                max_tokens=config.EVAL_MAX_TOKENS,
                eos_idx=eos_idx,  # <-- PASS IT HERE
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K
            )
            
            # Isolate just the generated part
            generated_indices = generated_tensor[0, config.EVAL_PROMPT_LENGTH:].cpu().tolist()
            generated_continuation = data_utils.decode(generated_indices, itow)
            
            # --- 4. Compute Perplexity ---
            full_seq = generated_tensor[:, :-1]
            targets = generated_tensor[:, 1:]
            
            logits, _ = model(full_seq)
            
            # Only get logits/targets for the generated part
            continuation_logits = logits[:, config.EVAL_PROMPT_LENGTH-1:, :]
            continuation_targets = targets[:, config.EVAL_PROMPT_LENGTH-1:]
            
            loss = F.cross_entropy(
                continuation_logits.reshape(-1, vocab_size),
                continuation_targets.reshape(-1)
            )
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())
            
            # --- 5. Compute BLEU Score ---
            results = bleu_metric.compute(
                predictions=[generated_continuation], 
                references=[[reference_continuation]]
            )
            bleu_scores.append(results['bleu'])
            
            if len(generation_samples) < 6: # Save a few samples to print
                generation_samples.append((
                    data_utils.decode(prompt_indices, itow),
                    generated_continuation,
                    reference_continuation
                ))

    # --- 6. Report Averages ---
    avg_perplexity = np.mean(perplexities)
    avg_bleu = np.mean(bleu_scores)

    print("\n--- Evaluation Results (on Official Val Set) ---")
    print(f"Samples: {config.EVAL_SAMPLES}")
    print(f"Average Perplexity per Token: {avg_perplexity:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print("------------------------------------------------\n")
    
    # --- 7. Print Sample Generations ---
    print("--- Sample Generations ---")
    for i, (prompt, gen, ref) in enumerate(generation_samples):
        print(f"\n[Sample {i+1}]")
        print(f"  PROMPT: {prompt}")
        print(f"  MODEL: {gen}")
        print(f"  REFERENCE: {ref}")
    print("--------------------------\n")

    # --- 8. Visualize Attention ---
    visualize_attention(model, wtoi, itow)

if __name__ == "__main__":
    main()