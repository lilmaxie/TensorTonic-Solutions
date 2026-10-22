import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    x  = np.asarray(x , dtype=np.float64)
    wq = np.asarray(Wq, dtype=np.float64)
    wk = np.asarray(Wk, dtype=np.float64)
    wv = np.asarray(Wv, dtype=np.float64)
    wo = np.asarray(Wo, dtype=np.float64)
    w1 = np.asarray(W1, dtype=np.float64)
    w2 = np.asarray(W2, dtype=np.float64)

    B, N, D = x.shape
    H = num_heads
    d_h = D // H

    # LayerNorm operation with population variance and epsilon = 1e-6
    def _layer_norm(u: np.ndarray) -> np.ndarray:
        mean = np.mean(u, axis=-1, keepdims=True)
        var = np.var(u, axis=-1, keepdims=True)

        return (u - mean) / np.sqrt(var + 1e-6)

    # GELU activation
    def _gelu(val: np.ndarray) -> np.ndarray:
        return 0.5 * val * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (val + 0.044715 * (val ** 3))))

    # STEP 1: MULTI-HEAD SELF-ATTENTION (MSA) + RESIDUAL 1
    x_norm1 = _layer_norm(x)

    # design Q, K, V: size (B, N, D)
    Q = x_norm1@wq
    K = x_norm1@wk
    V = x_norm1@wv

    # reshape and transpose heads: (B, H, N, d_h)
    Q = Q.reshape(B, N, H, d_h).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, H, d_h).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, H, d_h).transpose(0, 2, 1, 3)

    # calculate attention scores: (B, H, N, N)
    scores = (Q@K.swapaxes(-1, -2)) / np.sqrt(d_h)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # attention context: (B, H, N, d_h)
    context = attn_weights@V

    # connect the ends back to (B, N, D)
    context_concat = context.transpose(0, 2, 1, 3).reshape(B, N, D)

    # work with Wo and add the first remainder
    attn_out = context_concat@wo
    y = x + attn_out

    # STEP w: TWO-LAYER MLP + RESIDUAL 2
    y_norm2 = _layer_norm(y)
    mlp_hidden = _gelu(y_norm2@w1)
    mlp_out = mlp_hidden@w2

    # Add the second remainder
    z = y + mlp_out

    return z