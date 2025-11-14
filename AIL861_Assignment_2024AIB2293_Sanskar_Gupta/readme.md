Decoder-Only Transformer from Scratch (AIL861)

0. Download the encoding 
Download the embedding file from this link and keep it in the same directory as the other files without changing its name 
Link - https://drive.google.com/file/d/1xPeesJ9WTW59SvA2sVau72h4t0ssCpLf/view?usp=sharing
You can download the weights for my experiment from here 
Link - https://huggingface.co/sanskar1144/AIL861_Tinystories_Model

1. Setup and Installation

To run this project, you must first create the Conda environment which contains all necessary dependencies.
Clone the repository (if applicable) and navigate to the project directory.
Create the Conda environment using the provided environment.yml file:

conda env create -f environment.yml

Activate the environment:
conda activate llm_assignment


2. How to Run the Code

Step 1: Train the Model (Part 1.1)
The first step is to run the main training script. This will train the model, save the final model checkpoint (transformer_tinystories.pth), and save the vocabulary (vocab.json).

python train.py

What it does: Trains the model based on the settings in config.py.

Outputs:
transformer_tinystories.pth: The trained model weights.
vocab.json: The vocabulary (wtoi, itow) file.
training_metrics.png: A plot of the training and validation loss curves.

Step 2: Run Main Evaluation (Part 1.2)

After the model is trained, run this script to evaluate its performance on the official validation set.

python run_evaluation.py
What it does: Loads the trained model and runs the Part 1.2 evaluation.

Outputs:
Prints the final Average Perplexity and Average BLEU Score to the console.
Prints 5 sample generations (Prompt vs. Model vs. Reference).
Saves attention visualizations to the attn_plots/ directory.

Step 3: Run Part 2 Experiments

The following scripts are standalone experiments for Part 2. They must be run after train.py (as they all load the saved model), but can be run in any order.

Part 2.1: Beam Search Decoding
This script runs a comparison of Stochastic, Greedy (k=1), and Beam Search (k=5, k=10) decoding.
python beam.py
What it does: Runs 5 prompts through each decoding strategy.

Outputs: Prints a table comparing BLEU Score, Tokens/Second, and Avg. Time/Prompt for each strategy.

Part 2.2: KV Caching
This script measures the speedup (tokens/sec) gained from using a KV Cache.
python kv.py
What it does: Runs 20 prompts through the stochastic generator with and without the KV cache.
Outputs: Prints a table comparing the Tokens/Second and Total Time for both methods, along with the final speedup.

Part 2.3: Gradient Accumulation

This script compares training loss curves for different gradient accumulation settings.
python gacc.py
What it does: Runs 4 separate training loops (for 2,000 steps each) with accumulation steps of 1, 2, 4, and 8.

Outputs:
grad_accumulation_comparison.png: A plot comparing the training loss curves.
Prints a table comparing the Total Time for each configuration.