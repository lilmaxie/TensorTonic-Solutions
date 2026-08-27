import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    tokens = np.asarray(token_ids)
    mask_pos = np.asarray(mask_positions, dtype=bool)
    probs = np.asarray(replace_probs, dtype=float)
    rand_toks = np.asarray(random_tokens)

    # create a copy of `masked_ids` and initialize a `labels` array filled entirely with -100
    masked_ids = tokens.copy()
    labels = np.full(tokens.shape, -100, dtype=np.int64)

    # assign the original label to all positions selected as the mask
    labels[mask_pos] = tokens[mask_pos]

    # 80% group: replace_probs < 0.8 --> replace with mask_token_id (103)
    cond_mask = mask_pos & (probs < 0.8)
    masked_ids[cond_mask] = mask_token_id

    # 10% group: 0.8 <= replace_probs < 0.9 --> replace with random token
    cond_random = mask_pos & (probs >= 0.8) & (probs < 0.9)
    masked_ids[cond_random] = rand_toks[cond_random]

    # last 10%: replace_probs >= 0.9 --> keep original token
    return masked_ids, labels
    

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        hs = np.asarray(hidden_states, dtype=np.float64)
        W_mat = np.asarray(self.W, dtype=np.float64)
        b_vec = np.asarray(self.b, dtype=np.float64)

        # (batch, seq_len, hidden_size) x (hidden_size, vocab_size) --> (batch, seq_len, vocab_size)
        logits = hs @ W_mat + b_vec
        return logits
