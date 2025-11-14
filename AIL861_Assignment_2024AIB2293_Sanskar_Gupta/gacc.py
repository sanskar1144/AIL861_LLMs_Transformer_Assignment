import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import pandas as pd
import os

import config
import data_utils
from model import Transformer

STEPS_PER_RUN = 2000 # Number of training steps per accumulation setting
VALIDATION_INTERVAL = 100 # Validation interval in steps
ACCUMULATION_SETTINGS = [1, 2, 4, 8] # Different gradient accumulation steps to test

def run_training_run(acc_steps, num_steps, val_interval, train_corpus, val_corpus, vocab_size, vocab_list):
    """
    Runs a single training experiment for a given accumulation setting.
    This is a refactored version of your train.py's main().
    
    Returns:
        (step_history, train_loss_history, val_loss_history, total_time)
    """
    print(f"\n--- Running experiment: ACC_STEPS = {acc_steps} ---")
    
    # --- 1. Initialize Model & Optimizer ---
    # We re-initialize the model from scratch for each run
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 2. Data Lists for Plotting ---
    step_history = []
    train_loss_history = []
    val_loss_history = []
    
    # --- 3. Training Loop ---
    print(f"Training for {num_steps} steps...")
    start_time = time.time()
    optimizer.zero_grad()
    
    for i in tqdm(range(num_steps)):
        model.train()
        
        # --- Training Step ---
        x, y = data_utils.get_batch(train_corpus, config.BATCH_SIZE, config.CONTEXT_LENGTH, config.DEVICE)
        
        logits, _ = model(x, kv_caches=None)
        B, T, D = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, D), y.reshape(B * T))
        
        # Scale loss for accumulation
        loss = loss / acc_steps
        loss.backward()

        # Gradient accumulation step
        if (i + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # --- Validation Step ---
        if (i + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                # Get full train loss (not scaled)
                train_loss = (loss.item() * acc_steps)
                train_loss_history.append(train_loss)

                # Validation metrics
                x_val, y_val = data_utils.get_batch(val_corpus, config.BATCH_SIZE, config.CONTEXT_LENGTH, config.DEVICE)
                logits_val, _ = model(x_val)
                B_val, T_val, D_val = logits_val.shape
                val_loss = F.cross_entropy(logits_val.reshape(B_val* T_val, D_val), y_val.reshape(B_val* T_val)).item()
                val_loss_history.append(val_loss)
                
                step_history.append(i + 1)

            tqdm.write(f"Step {i+1:05d} (Acc={acc_steps}) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"--- Run (Acc={acc_steps}) finished in {total_time:.2f} seconds ---")
    
    return step_history, train_loss_history, val_loss_history, total_time


def main():
    """
    Main function to orchestrate the gradient accumulation experiment.
    """
    print(f"Starting Gradient Accumulation experiment")
    print(f"Running {len(ACCUMULATION_SETTINGS)} configurations for {STEPS_PER_RUN} steps each.")
    
    # --- 1. Load Data and Vocab (Done once) ---
    print("Loading and preprocessing data...")
    train_texts, val_texts, _ = data_utils.load_and_split_data(
        config.DATASET_NAME,
        config.DATA_SUBSET,
        config.VAL_SET_SIZE
    )
    
    wtoi, itow, vocab_list = data_utils.build_vocabulary(train_texts, config.VOCAB_MIN_FREQ)
    vocab_size = len(wtoi)
    print(f"Vocabulary size: {vocab_size}")
    
    # We don't need to save the vocab for this experiment, but we need the data.
    print("Creating token corpuses...")
    train_corpus = data_utils.get_data_for_batching(train_texts, wtoi)
    val_corpus = data_utils.get_data_for_batching(val_texts, wtoi)
    
    # --- 2. Run All Experiments ---
    all_plot_data = {}
    all_runtimes = []

    for acc_steps in ACCUMULATION_SETTINGS:
        # Run the training
        step_history, train_loss, val_loss, total_time = run_training_run(
            acc_steps=acc_steps,
            num_steps=STEPS_PER_RUN,
            val_interval=VALIDATION_INTERVAL,
            train_corpus=train_corpus,
            val_corpus=val_corpus,
            vocab_size=vocab_size,
            vocab_list=vocab_list
        )
        
        # Store results
        all_plot_data[acc_steps] = (step_history, train_loss, val_loss)
        all_runtimes.append({
            "Accum. Steps": acc_steps,
            "Effective Batch Size": config.BATCH_SIZE * acc_steps,
            f"Total Time for {STEPS_PER_RUN} steps (s)": total_time,
            f"Time per 'Epoch' ({STEPS_PER_RUN//2} steps) (s)": total_time / (STEPS_PER_RUN / (STEPS_PER_RUN//2))
        })

    # --- 3. Plot All Loss Curves on One Graph ---
    print("\nPlotting comparison graph...")
    plt.figure(figsize=(12, 8))
    for acc_steps, (steps, train_loss, val_loss) in all_plot_data.items():
        plt.plot(steps, train_loss, label=f'Train Loss (Acc={acc_steps})')
        # You can also plot validation loss if you want
        # plt.plot(steps, val_loss, '--', label=f'Val Loss (Acc={acc_steps})')

    plt.title('Training Loss vs. Gradient Accumulation Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plot_filename = "grad_accumulation_comparison.png"
    plt.savefig(plot_filename)
    print(f"✅ Comparison plot saved to {plot_filename}")

    # --- 4. Report Runtimes in a Table ---
    print("\n\n--- Experiment Results (Part 2.3) ---")
    results_df = pd.DataFrame(all_runtimes)
    print(results_df.to_string(index=False))
    
    print("\nExperiment complete.")

if __name__ == "__main__":
    main()