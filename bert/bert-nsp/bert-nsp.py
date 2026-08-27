import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    """
    Returns: list of (sentence_A, sentence_B, is_next_label) tuples
    """
    pairs = []

    # iterate through each pairing specification in pair_specs
    for spec in pair_specs:
        doc_a = spec["doc_a"]
        sent_a = spec["sent_a"]
        doc_b = spec["doc_b"]
        sent_b = spec["sent_b"]

        # extract sentence A and sentence B from the document
        sentence_a = documents[doc_a][sent_a]
        sentence_b = documents[doc_b][sent_b]

        # check for consecutive conditions: same document and `sent_b` immediately following `sent_a`
        if doc_a == doc_b and sent_b == sent_a + 1:
            is_next_label = 1
        else:
            is_next_label = 0

        pairs.append((sentence_a, sentence_b, is_next_label))

    return pairs
        

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext logits: cls_hidden @ W + b
        """
        h = np.asarray(cls_hidden, dtype=np.float64)
        W_mat = np.asarray(self.W, dtype=np.float64)
        b_vec = np.asarray(self.b, dtype=np.float64)

        logits = h@W_mat + b_vec

        return logits

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
