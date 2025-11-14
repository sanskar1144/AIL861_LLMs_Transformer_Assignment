import torch

# --- Data ---
DATASET_NAME = "roneneldan/TinyStories"
DATA_SUBSET = 50000 # Use a subset of data (e.g., 10000) or -1 for the full dataset
VAL_SET_SIZE = 0.10 # 10% of the training data for validation
VOCAB_MIN_FREQ = 1  # Minimum frequency for a word to be in vocab

# --- Model ---
EMBED_DIM = 300       # Must match FastText
NUM_LAYERS = 4
NUM_HEADS = 6
HEAD_DIM = EMBED_DIM // NUM_HEADS # Head dim must be embed_dim // num_heads - 50 in our case
FF_HIDDEN_DIM = 4 * EMBED_DIM # Fully connected hidden layer dimension
CONTEXT_LENGTH = 64   # Max sequence length
VOCAB_SIZE = None     # Will be set dynamically in train.py

# --- Training ---
LEARNING_RATE = 1e-3
N_STEPS = 20000
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Uncomment this to run on GPU
DEVICE = "cpu"  # I ran on the  CPU itself in my MAC. Prefer to use CPU

# --- Evaluation & Generation ---
EVAL_SAMPLES = 50        # Number of samples for BLEU/Perplexity
EVAL_PROMPT_LENGTH = 5
EVAL_MAX_TOKENS = 40     # Tokens to generate
BEAM_WIDTH = 5
TEMPERATURE = 1.0 # Just in case you want to experiment with temperature sampling, change Temperature
TOP_K = 10

# --- File Paths ---
MODEL_SAVE_PATH = "transformer_tinystories.pth" # Path to save/load the trained model
METRICS_PLOT_PATH = "training_metrics.png" # Path to save training/validation loss plot
ATTN_PLOT_DIR = "attn_plots" # Folder to save attention visualizations
VOCAB_SAVE_PATH = "vocab.json" # Path to save/load vocabulary