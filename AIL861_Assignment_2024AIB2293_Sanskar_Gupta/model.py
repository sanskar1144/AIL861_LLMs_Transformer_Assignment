import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fasttext
import fasttext.util
import config

# ================================================
# 1. Multi-Head Attention Head
# ================================================
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.head_dim = head_dim
        self.last_attn = None # For visualization

    def forward(self, x, kv_cache=None, record_attn=False):
        # x: The input tensor, with shape (Batch, SequenceLength, embed_dim).
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # If cache exists, append new keys/values
        if kv_cache is not None:
            # We only get the *newest* Q, K, V (seq_len=1)
            # So we concat K, V with the cached K, V
            K = torch.cat([kv_cache['k'], K], dim=1)
            V = torch.cat([kv_cache['v'], V], dim=1)
            # x in this case represents only the newest token and not the entire sequence

        # Save updated cache
        new_cache = {'k': K, 'v': V}

        # Compute attention scores
        # Q shape: (B, T_q, C)
        # K.T shape: (B, C, T_k)
        # Result: (B, T_q, T_k)
        # Take a transpose on the last two dims of K and also scale the dot product further down
        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Causal mask for autoregressive attention
        T_q, T_k = Q.shape[1], K.shape[1]
        mask = torch.tril(torch.ones(T_q, T_k, device=x.device))
        
        # # When using KV cache, T_q is 1 but T_k grows.
        # # We only need to mask the last row of the attention matrix.
        # # But a full tril mask is still correct and simpler.
        if T_q == 1 and T_k > 1:
            # More efficient mask for generation: no need to mask
            pass
        else:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        if record_attn:
            self.last_attn = attn_weights.detach().cpu()
        
        out = attn_weights @ V
        return out, new_cache

# ================================================
# 2. Transformer Block
# ================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.linear_proj = nn.Linear(num_heads * head_dim, embed_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, kv_caches=None):
        residual = x
        x = self.ln1(x)

        new_caches = []
        attention_outputs = []

        for i, head in enumerate(self.attention_heads):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            attn_out, new_cache = head(x, kv_cache)
            attention_outputs.append(attn_out)
            new_caches.append(new_cache)

        concatenated = torch.cat(attention_outputs, dim=-1)
        mha_out = self.linear_proj(concatenated)
        x = residual + mha_out

        residual2 = x
        x = self.ln2(x)
        x = self.feedforward(x)
        x = residual2 + x
        return x, new_caches

# ================================================
# 3. Transformer Model
# ================================================
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, head_dim, ff_hidden_dim, context_size, vocab_list):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize with FastText
        self.initialize_fasttext_embeddings(vocab_list, embed_dim)

        # Positional encoding
        pe = torch.zeros(context_size, embed_dim)
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, head_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim) # Final LayerNorm before output
        self.unembedding = nn.Linear(embed_dim, vocab_size)

        # Weight tying - forcing the output embedding to be the same as input embedding
        self.unembedding.weight = self.embedding.weight

    def forward(self, x, kv_caches=None):
        seq_len = x.size(1)
        
        # Handle KV Caching: x is only the newest token
        if kv_caches is not None:
            # pos_offset is the length of the cache
            pos_offset = kv_caches[0][0]['k'].shape[1] 
            pos_emb = self.pe[pos_offset : pos_offset + seq_len, :]
            embeddings = self.embedding(x) + pos_emb
        else:
            # Full forward pass
            pos_emb = self.pe[:seq_len, :]
            embeddings = self.embedding(x) + pos_emb

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            embeddings, new_cache = layer(embeddings, layer_cache)
            new_caches.append(new_cache)

        embeddings = self.ln_final(embeddings)
        logits = self.unembedding(embeddings)
        return logits, new_caches
    
    def build_embedding_matrix_fasttext(self, ft_model, vocab, embed_dim):
        matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
        wtoi = {word: i for i, word in enumerate(vocab)}
        # Loops through every word in our vocabulary.\
        # Gets its pre-trained vector from the ft_model.
        # Assigns that vector to the correct idx (row) in the matrix.
        for word, idx in wtoi.items():
            vec = ft_model.get_word_vector(word)
            matrix[idx] = vec
        return matrix

    def initialize_fasttext_embeddings(self, vocab, embed_dim):
        print("Initializing embeddings with FastText...")
        fasttext.util.download_model('en', if_exists='ignore')
        ft = fasttext.load_model("cc.en.300.bin")
        
        # Check dim
        if ft.get_dimension() != embed_dim:
            raise ValueError(f"FastText dim ({ft.get_dimension()}) doesn't match config EMBED_DIM ({embed_dim})")
            
        embedding_matrix = self.build_embedding_matrix_fasttext(ft, vocab, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # Freeze embeddings as per assignment suggestion [cite: 37]
        self.embedding.weight.requires_grad = False
        print(" Embeddings initialized and frozen.")
        
    def forward_with_attn_capture(self, x, num_heads_to_visualize=5):
        """
        Runs a forward pass while recording attention maps from the first Transformer block.
        Returns logits and the captured attention maps.
        """
        seq_len = x.size(1)
        embeddings = self.embedding(x) + self.pe[:seq_len, :]
        
        captured_attns = []

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = None
            
            # Monkey-patch the forward pass of the first layer's heads to record attention
            if i == 0:
                # Clear previous attention maps
                for head in layer.attention_heads:
                    head.last_attn = None
                
                # Create a list of heads to record
                heads_to_record = layer.attention_heads[:num_heads_to_visualize]
                
                # Define a new forward function that sets record_attn=True
                def make_recording_forward(head_obj):
                    def recording_forward(x_in, kv_cache=None):
                        # Call the original AttentionHead.forward
                        # return AttentionHead.forward(self, x_in, kv_cache=kv_cache, record_attn=True)
                        return AttentionHead.forward(head_obj, x_in, kv_cache=kv_cache, record_attn=True)
                    return recording_forward
                
                # Temporarily replace forward methods
                original_forwards = {}
                for head in heads_to_record:
                    original_forwards[head] = head.forward
                    head.forward = make_recording_forward(head)

            # Run the layer
            embeddings, new_cache = layer(embeddings, layer_cache)
            new_caches.append(new_cache)
            
            # After the first layer, restore original forward methods
            if i == 0:
                captured_attns = [h.last_attn for h in heads_to_record]
                for head, orig_forward in original_forwards.items():
                    head.forward = orig_forward # Restore

        embeddings = self.ln_final(embeddings)
        logits = self.unembedding(embeddings)
        return logits, captured_attns