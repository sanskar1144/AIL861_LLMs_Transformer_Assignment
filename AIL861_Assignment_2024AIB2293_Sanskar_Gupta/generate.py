import torch
import torch.nn.functional as F
import config
# ---------------------------------------------
# Stochastic Sampling (with KV Caching)
# ---------------------------------------------
def generate_stochastic(model, x, max_tokens, eos_idx, temperature=1.0, top_k=None):
    """
    Autoregressive generation with KV Caching and stochastic sampling.
    """
    model.eval()
    kv_caches = None
    x_input = x

    for _ in range(max_tokens):
        # On the first pass, x_input is the full prompt
        # On subsequent passes, it's just the last token
        if kv_caches is not None:
            x_input = x_input[:, -1:]

        # Get logits and new caches
        logits, kv_caches = model(x_input, kv_caches)
        
        # Get logits for the last token
        logits = logits[:, -1, :] / temperature

        # Optional top-k filtering
        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k)
            min_vals = top_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_vals, torch.full_like(logits, float('-inf')), logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to the sequence
        x = torch.cat([x, next_token], dim=-1)
        x_input = x # The full sequence is used for the next token input

        # --- ADD THIS CHECK ---
        # If the model generated the <eos> token, stop.
        if next_token.item() == eos_idx:
            break
        # --- END OF ADDED BLOCK ---

    return x

# # ---------------------------------------------
# # Greedy Inference (No KV Caching)
# # ---------------------------------------------
# def generate_greedy(model, x, max_tokens, context_size):
#     """
#     Simple greedy generation *without* KV Caching.
#     """
#     model.eval()
#     for _ in range(max_tokens):
#         # Crop context to avoid exceeding positional encoding
#         x_cond = x[:, -context_size:]
#         logits, _ = model(x_cond, kv_caches=None)
        
#         # Get the last token
#         next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
#         x = torch.cat((x, next_token), dim=-1)
#     return x

# ---------------------------------------------
# Batch Beam Search
# ---------------------------------------------
"""
def generate_beam_search(model, x_batch, beam_width, max_tokens, context_size):
    
    # Beam search generation for a batch.
    
    model.eval()
    batch_results = []

    for b in range(x_batch.size(0)):
        x = x_batch[b].unsqueeze(0) # Process one prompt at a time
        
        # [beam_width, seq_len]
        beams = x.repeat(beam_width, 1) 
        # [beam_width]
        scores = torch.zeros(beam_width, device=x.device)

        for _ in range(max_tokens):
            candidates = []
            candidate_scores = []
            
            # Get logits for all beams in parallel
            beam_inputs = beams[:, -context_size:]
            logits, _ = model(beam_inputs) # [beam_width, seq_len, vocab_size]
            
            # Get probs for the last token
            probs = F.softmax(logits[:, -1, :], dim=-1) # [beam_width, vocab_size]
            
            # Get top k probabilities for *each* beam
            # topk_probs: [beam_width, beam_width]
            # topk_indices: [beam_width, beam_width]
            topk_probs, topk_indices = probs.topk(beam_width, dim=-1)

            for i in range(beam_width): # For each current beam
                for k in range(beam_width): # For each new candidate
                    new_beam = torch.cat([beams[i], topk_indices[i, k].unsqueeze(0)])
                    new_score = scores[i] + torch.log(topk_probs[i, k])
                    
                    candidates.append(new_beam)
                    candidate_scores.append(new_score)

            # Select the top 'beam_width' candidates from all possibilities
            # candidate_scores will be [beam_width * beam_width]
            candidate_scores = torch.stack(candidate_scores)
            top_scores, top_indices = candidate_scores.topk(beam_width)
            
            # Update beams and scores
            beams = torch.stack([candidates[i] for i in top_indices])
            scores = top_scores

        # Select the best beam for this batch item
        best_idx = torch.argmax(scores).item()
        batch_results.append(beams[best_idx])

    return torch.stack(batch_results)
"""

def generate_beam_search(model, x_batch, beam_width, max_tokens, context_size, eos_idx):
    """
    Beam search generation for a batch.
    This version is not KV-cached
    It now uses log-probabilities and correctly handles <eos> stopping.
    """
    model.eval()
    batch_results = []
    
    for b in range(x_batch.size(0)):
        x = x_batch[b].unsqueeze(0) # Process one prompt at a time
        
        # Start with k identical beams
        beams = [x.clone() for _ in range(beam_width)]
        scores = torch.zeros(beam_width, device=x.device)
        
        # List to store finished beams
        finished_beams = []
        finished_scores = []

        for _ in range(max_tokens):
            all_candidates = []
            all_scores = []
            
            # # Keep track of beams that are still running
            # new_beams = []
            # new_scores = []
            
            for i in range(len(beams)): # For each active beam
                beam = beams[i]
                beam_score = scores[i]

                # Get logits for the last token
                beam_input = beam[:, -context_size:]
                logits, _ = model(beam_input)
                
                # Use log_softmax for numerical stability
                probs = F.log_softmax(logits[:, -1, :], dim=-1) # [1, vocab_size]
                
                # Get top k candidates for *this* beam
                topk_probs, topk_indices = probs.topk(beam_width, dim=-1)
                
                for k in range(beam_width):
                    new_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0) # Shape [1, 1]
                    new_beam = torch.cat([beam, new_token], dim=-1)
                    new_score = beam_score + topk_probs[0, k]

                    if new_token.item() == eos_idx:
                        # This beam is finished, add to finished list
                        finished_beams.append(new_beam)
                        finished_scores.append(new_score)
                    else:
                        # This beam is still active
                        all_candidates.append(new_beam)
                        all_scores.append(new_score)

            if not all_candidates: # All beams finished
                break
                
            # Select the top k from all *active* candidates
            all_scores = torch.stack(all_scores)
            top_scores, top_indices = all_scores.topk(min(beam_width, len(all_scores)))
            
            beams = [all_candidates[i] for i in top_indices]
            scores = top_scores

        # After max_tokens, if we have finished beams, choose the best one
        if finished_beams:
            best_finished_score = torch.stack(finished_scores).max()
            # If the best active beam is worse than the best finished beam, use the finished one
            if not beams or scores.max() < best_finished_score:
                best_idx = torch.stack(finished_scores).argmax().item()
                batch_results.append(finished_beams[best_idx].squeeze(0))
                continue # Go to the next item in batch

        # If no beams finished, just take the best one from the active list
        best_idx = torch.argmax(scores).item()
        batch_results.append(beams[best_idx].squeeze(0))

    return torch.stack(batch_results)


# ---------------------------------------------
# Stochastic Sampling (WITHOUT KV Caching)
# ---------------------------------------------
def generate_stochastic_no_cache(model, x, max_tokens, eos_idx, temperature=1.0, top_k=None):
    """
    Autoregressive generation WITHOUT KV Caching.
    It re-computes the entire sequence at every step.
    """
    model.eval()
    # No cache variables.

    for _ in range(max_tokens):
        # 1. Always crop input to the max context size
        # We must re-process the entire sequence every time.
        x_input = x[:, -config.CONTEXT_LENGTH:]
        
        # 2. Always call the model with no cache
        logits, _ = model(x_input, kv_caches=None) 
        
        # 3. Get logits for the very last token
        logits = logits[:, -1, :] / temperature

        # 4. Optional top-k filtering (Identical to the other function)
        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k)
            min_vals = top_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_vals, torch.full_like(logits, float('-inf')), logits)

        # 5. Sample (Identical to the other function)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 6. Append the new token
        x = torch.cat([x, next_token], dim=-1)

        # 7. Check for <eos>
        if next_token.item() == eos_idx:
            break

    return x
