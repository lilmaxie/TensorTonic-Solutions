import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    
    seq_length: int, number of positions (sequence length)
    d_model: int, embedding dimension
    Returns: np.ndarray of shape (seq_length, d_model)
    """
    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    # create a positional vector `pos` with shape (seq_length, 1) for broadcasting.
    pos = np.arange(seq_length, dtype=np.float64).reshape(-1, 1)

    even_indices = np.arange(0, d_model, 2, dtype=np.float64)
    div_term = np.exp(even_indices * (-np.log(10000.0) / d_model))

    # calculate the phase angle: theta = pos * div_term, which has the shape (seq_length, d_model // 2).
    angles = pos * div_term

    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe