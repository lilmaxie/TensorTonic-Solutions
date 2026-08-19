import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # init PE matrix with shape (seq_len, d_model)
    pe = np.zeros((seq_len, d_model), dtype=np.float64)

    # positional vector pos has shape (seq_len, 1)
    pos = np.arange(seq_len, dtype=np.float64)[:, np.newaxis]

    # calculate even column (sin): 2i = 0, 2, 4,...
    even_indices = np.arange(0, d_model, 2)
    div_term_sin = np.power(base, even_indices/d_model)
    pe[:, 0::2] = np.sin(pos/div_term_sin)

    # calculate odd column (sin): 2i+1 with 2i = 0, 2, 4,...
    if d_model > 1:
        odd_indices = np.arange(0, d_model-1, 2)
        div_term_cos = np.power(base, odd_indices/d_model)
        pe[:, 1::2] = np.cos(pos/div_term_cos)
        
    return pe
    