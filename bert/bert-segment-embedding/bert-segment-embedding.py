import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        token_ids = np.asarray(token_ids)
        segment_ids = np.asarray(segment_ids)

        # extract batch, seq_len size
        batch_size, seq_len = token_ids.shape

        # look up Token Embeddings using NumPy Advanced Indexing --> shape (batch_size, seq_len, hidden_size)
        e_token = self.token_embeddings[token_ids]

        # create position indices [0, 1, ..., seq_len - 1] and look up position embeddings --> shape (seq_len, hidden_size)
        positions = np.arange(seq_len)
        e_pos = self.position_embeddings[positions]

        # look up segment embeddings (use 0 or 1) --> shape (batch_size, seq_len, hidden_size)
        e_seg = self.segment_embeddings[segment_ids]

        combine_embeddings = e_token + e_pos + e_seg

        return combine_embeddings