import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json 
import config
import data_utils
from model import Transformer
import time 

def main():
    print("Starting training script...")
    print(f"Using device: {config.DEVICE}")
    start_time = time.time()
    print("Time at the start:", start_time)

    # --- 1. Load Data ---
    print("Loading and preprocessing data...")
    train_texts, val_texts, _ = data_utils.load_and_split_data(
        config.DATASET_NAME, #The name of the dataset is directly taken from config file
        config.DATA_SUBSET, # dat subset defines how much data to use for training
        config.VAL_SET_SIZE # portion of training data to use for validation
    )
    
    # --- 2. Build Vocabulary ---
    wtoi, itow, vocab_list = data_utils.build_vocabulary(train_texts, config.VOCAB_MIN_FREQ)
    # wtoi is a word to index mapping
    # itow is an index to word mapping
    vocab_size = len(wtoi)
    # print(len(vocab_list), len(wtoi), len(itow))
    print(f"Vocabulary size: {vocab_size}")
    
    # Save Vocabulary ---
    print(f"Saving vocabulary to {config.VOCAB_SAVE_PATH}...")
    vocab_data = {
        "wtoi": wtoi,
        "itow": itow,
        "vocab_list": vocab_list
    }
    with open(config.VOCAB_SAVE_PATH, 'w') as f:
        json.dump(vocab_data, f, indent=4)


    # # --- 3. Create Corpus for Batching ---
    # Convert train and val texts into flat lists of token indices
    # Append all stories into one long list of indices for efficient batching
    # encoding each story and concatenating them
    train_corpus = data_utils.get_data_for_batching(train_texts, wtoi)
    val_corpus = data_utils.get_data_for_batching(val_texts, wtoi)
    print(f"Training tokens: {len(train_corpus)}, Validation tokens: {len(val_corpus)}")


    # --- 4. Initialize Model ---
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        head_dim=config.HEAD_DIM,
        ff_hidden_dim=config.FF_HIDDEN_DIM,
        context_size=config.CONTEXT_LENGTH,
        vocab_list=vocab_list
    ).to(config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- 5. Training Setup ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    steps = []

    # --- 6. Training Loop ---
    print("Starting training...")
    optimizer.zero_grad() # Initialize gradient
    
    for i in tqdm(range(config.N_STEPS)):
        model.train()
        
        # --- Training Step ---
        x, y = data_utils.get_batch(train_corpus, config.BATCH_SIZE, config.CONTEXT_LENGTH, config.DEVICE)
        
        logits, _ = model(x, kv_caches=None) # No cache during training
        B, T, D = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, D), y.reshape(B * T))
        
        # Scale loss for accumulation
        loss = loss / config.ACCUMULATION_STEPS 
        loss.backward()

        # Gradient accumulation step
        if (i + 1) % config.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        # --- Validation Step ---
        if (i + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Get full train loss (not scaled)
                train_loss = (loss.item() * config.ACCUMULATION_STEPS)
                train_perplexity = np.exp(train_loss)
                train_losses.append(train_loss)
                train_perplexities.append(train_perplexity)

                # Validation metrics
                x_val, y_val = data_utils.get_batch(val_corpus, config.BATCH_SIZE, config.CONTEXT_LENGTH, config.DEVICE)
                logits_val, _ = model(x_val)
                B_val, T_val, D_val = logits_val.shape
                val_loss = F.cross_entropy(logits_val.reshape(B_val* T_val, D_val), y_val.reshape(B_val* T_val)).item()
                val_perplexity = np.exp(val_loss)
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                
                steps.append(i + 1)

            tqdm.write(f"Step {i+1:05d} | Train Loss: {train_loss:.4f} | Train PPL: {train_perplexity:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.4f}")

    # --- 7. Plot and Save Metrics ---
    print("Training complete. Plotting metrics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(steps, train_losses, label='Training Loss')
    ax1.plot(steps, val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(steps, train_perplexities, label='Training Perplexity')
    ax2.plot(steps, val_perplexities, label='Validation Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(config.METRICS_PLOT_PATH)
    print(f"Metrics plot saved to {config.METRICS_PLOT_PATH}")

    # --- 8. Save Model ---
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved as {config.MODEL_SAVE_PATH}")

    print("Training script finished.")
    end_time = time.time()
    print("Time at the end:", end_time)
    print("Total training time (seconds):", end_time - start_time)

if __name__ == "__main__":
    main()