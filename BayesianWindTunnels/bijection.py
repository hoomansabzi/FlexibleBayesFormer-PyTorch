import torch
import random 

class BijectionWindTunnel:
    """
    Generates data and exact Bayesian posteriors for the Bijection task.
    Optimized for GPU training via Bulk Tensor Transfer.
    """
    def __init__(self, V=20, device='cuda', shared_vocab=False):
        self.V = V
        self.device = device 
        self.sep_token = 0
        self.shared_vocab = shared_vocab
        
        self.x_offset = 1
        
        if self.shared_vocab:
            # Both x and y draw from the exact same token pool (1 to V)
            self.y_offset = 1
            self.vocab_size = V + 1 
        else:
            # Disjoint vocabulary: x is (1 to V), y is (V+1 to 2V)
            self.y_offset = V + 1
            self.vocab_size = 2 * V + 1

    def generate_batch(self, batch_size, sep=True, repeat_prob=0.0, seq_pairs=None, include_trivial_pair=False, repeat_mode='last_only'):
        """
        sep (bool): If True, pairs are [x, y, SEP]. If False, sequences are dense [x, y, x, y].
        repeat_prob (float): The probability of repeating a previously seen pair.
        repeat_mode (str): 'last_only' applies repeat_prob ONLY to the final pair.
                           'anywhere' applies repeat_prob to ANY pair after the first one.
        """
        if seq_pairs is None:
            seq_pairs = self.V if include_trivial_pair else self.V - 1
            
        tokens_per_pair = 3 if sep else 2
        seq_len = seq_pairs * tokens_per_pair 
        
        # Validate repeat_mode
        assert repeat_mode in ['last_only', 'anywhere'], "repeat_mode must be 'last_only' or 'anywhere'"
        
        batch_sequences = []
        
        for b in range(batch_size):
            # Both pools will be [1, 2, ..., V] if shared_vocab is True
            x_pool = list(range(self.x_offset, self.x_offset + self.V))
            y_pool = list(range(self.y_offset, self.y_offset + self.V))
            
            random.shuffle(x_pool)
            random.shuffle(y_pool)
            
            seen_x_to_y = {}
            pool_idx = 0
            seq = []
            
            for step in range(seq_pairs):
                # If we've exhausted our vocabulary size, we must force a repeat
                force_repeat = (pool_idx >= self.V)
                
                # Determine if we should repeat a pair based on the chosen mode
                do_repeat = force_repeat
                if len(seen_x_to_y) > 0 and not force_repeat:
                    if repeat_mode == 'last_only':
                        is_last_pair = (step == seq_pairs - 1)
                        if is_last_pair and random.random() < repeat_prob:
                            do_repeat = True
                    elif repeat_mode == 'anywhere':
                        if random.random() < repeat_prob:
                            do_repeat = True
                
                if do_repeat:
                    # Draw a previously seen pair
                    chosen_x = random.choice(list(seen_x_to_y.keys()))
                    chosen_y = seen_x_to_y[chosen_x]
                else:
                    # Draw a novel pair
                    chosen_x = x_pool[pool_idx]
                    chosen_y = y_pool[pool_idx]
                    seen_x_to_y[chosen_x] = chosen_y
                    pool_idx += 1
                        
                if sep:
                    seq.extend([chosen_x, chosen_y, self.sep_token])
                else:
                    seq.extend([chosen_x, chosen_y])
            
            batch_sequences.append(seq)
        
        sequences = torch.tensor(batch_sequences, dtype=torch.long, device=self.device)
        y_predict_indices = torch.arange(1, seq_len, tokens_per_pair, device=self.device)
        
        return sequences, y_predict_indices

    def get_bayesian_posterior(self, sequences, y_predict_indices):
        batch_size = sequences.shape[0]
        
        seq_lists = sequences.cpu().tolist()
        y_indices_list = y_predict_indices.cpu().tolist()
        
        batch_posteriors = []
        
        for b in range(batch_size):
            seq = seq_lists[b]
            seen_x_to_y = {}
            seen_y_set = set()
            seq_posteriors = []
            
            for step, idx in enumerate(y_indices_list):
                current_x = seq[idx - 1]
                true_y = seq[idx]
                
                step_probs = [0.0] * self.vocab_size
                
                if current_x in seen_x_to_y:
                    step_probs[seen_x_to_y[current_x]] = 1.0
                else:
                    remaining_options = max(1, self.V - len(seen_y_set))
                    prob = 1.0 / remaining_options
                    
                    # This loop naturally adapts to the shared offset (1 to V)
                    for y_val in range(self.y_offset, self.y_offset + self.V):
                        if y_val not in seen_y_set:
                            step_probs[y_val] = prob
                            
                seq_posteriors.append(step_probs)
                seen_x_to_y[current_x] = true_y
                seen_y_set.add(true_y)
                
            batch_posteriors.append(seq_posteriors)
            
        return torch.tensor(batch_posteriors, dtype=torch.float32, device=self.device)
