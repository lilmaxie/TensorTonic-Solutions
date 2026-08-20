import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.

    Q, K, V: shape (batch_size, seq_len, d_model)
    W_q, W_k, W_v, W_o: shape (d_model, d_model)
    num_heads: int (h)
    Returns: ndarray of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len_q, d_model = Q.shape
    _, seq_len_k, _ = K.shape
    d_k = d_model // num_heads

    # linearly project Q, K, V using weight matrices
    # shape after dot product: (batch_size, seq_len, d_model)
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    # split and permute axes to move the heads dimension to the second position
    # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)

    # compute the parallel attention score matrix for all heads
    # transpose the last two dimensions of K_heads to (batch, num_heads, d_k, seq_len_k)
    # scores has shape: (batch, num_heads, seq_len_q, seq_len_k)
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    # normalizing attention probabilities using the Softmax function
    attn_weights = softmax(scores, axis=-1)

    # multiply Attention Weights by V_heads 
    # attn_out has shape: (batch, num_heads, seq_len_q, d_k)
    attn_out = np.matmul(attn_weights, V_heads)

    # concatenate the heads back into the original d_model space
    # (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, num_heads, d_k) -> (batch, seq_len_q, d_model)
    concat_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, d_model)

    output = np.dot(concat_out, W_o)

    return output