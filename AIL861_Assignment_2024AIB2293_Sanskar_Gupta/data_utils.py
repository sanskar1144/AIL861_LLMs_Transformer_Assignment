import torch
import random
import re
from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import config

"""
This file handles all data preparation. 
Its job is to take the raw text from TinyStories and turn it into clean, numerical batches of tensors
that the model can directly consume for training.
"""


# --- Text Preprocessing ---

def preprocess_text(text):
    """Basic text preprocessing: lowercase, remove special chars, split."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Apart from numbers, letters, and spaces, remove everything
    tokens = text.split()
    return tokens

# --- Vocabulary ---

def build_vocabulary(stories, min_freq):
    """Builds vocab from a list of stories."""
    # Input is 'train texts' which is actually a list of strings/ stories
    all_tokens = []
    for story in stories:
        all_tokens.extend(preprocess_text(story)) # store all tokens in a list
    
    counter = Counter(all_tokens)
    # Vocab includes words and special tokens
    vocab = [word for word, freq in counter.items() if freq >= min_freq]
    # Add special tokens
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab = special_tokens + vocab
    # print (vocab[:20]) # Print first 20 tokens in vocab for verification
    wtoi = {word: i for i, word in enumerate(vocab)}
    itow = {i: word for i, word in enumerate(vocab)}
    
    return wtoi, itow, vocab

# --- Data Loading and Splitting ---
def load_and_split_data(dataset_name, subset, val_size):
    """Loads dataset, splits it into train/val, and processes text."""
    ds = load_dataset(dataset_name)
    # print(type(ds))
    # print(ds)
    # Use full train set or a subset
    if subset > 0:
        train_stories = ds['train']['text'][:subset] # Creates a sub data to train on
    else:
        train_stories = ds['train']['text']
        
    # print(type(train_stories), len(train_stories))
    # print(train_stories[0])
    # Split train data into train and validation
    train_texts, val_texts = train_test_split(
        train_stories, 
        test_size=val_size, 
        random_state=42
    )
    # print(type(train_texts), len(train_texts), type(val_texts), len(val_texts))
    # The official validation set for final evaluation
    official_val_texts = ds['validation']['text']
    
    # return [],  [], []
    return train_texts, val_texts, official_val_texts

# --- Encoding/Decoding ---

def encode(text, wtoi):
    """Encodes a single string into a tensor of indices."""
    # The input to each function is a single story/ string
    tokens = preprocess_text(text)
    # Add <sos> and <eos>
    tokens = ['<sos>'] + tokens + ['<eos>']
    unk_idx = wtoi.get('<unk>', 0) # # Get the index for the <unk> token, defaulting to 0 if it's somehow missing
    # Convert each word to its index.
    # If a word is not in our vocab (wtoi), use the <unk> index.
    indices = [wtoi.get(w, unk_idx) for w in tokens]
    
    # Return as a PyTorch tensor
    return torch.tensor(indices, dtype=torch.long)
    
def decode(indices, itow):
    """Decodes a tensor of indices into a string."""
    text = [itow.get(int(i), '<unk>') for i in indices]
    return " ".join(text)

# --- Batching ---

def get_data_for_batching(texts, wtoi):
    """Converts all texts into a single flat list of token indices."""
    # Input to this is train texts or val texts which is a list of strings/ stories
    corpus_indices = []
    for text in texts:
        corpus_indices.extend(encode(text, wtoi).tolist())
    # print(f"First story encoded length: {len(corpus_indices)} tokens")
    # print(corpus_indices)
    return corpus_indices

def get_batch(corpus_indices, batch_size, context_length, device):
    """Generates a batch of data from the flat corpus."""
    idx_list = [random.randint(0, len(corpus_indices) - context_length - 2) for _ in range(batch_size)]
    
    x = torch.stack([torch.tensor(corpus_indices[idx : idx + context_length]) for idx in idx_list])
    y = torch.stack([torch.tensor(corpus_indices[idx + 1 : idx + 1 + context_length]) for idx in idx_list])
    
    return x.to(device), y.to(device)