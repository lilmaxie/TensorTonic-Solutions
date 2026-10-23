import numpy as np

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 Vision Transformer logits with shape (B, C).
    """
    img = np.asarray(image, dtype=np.float64)
    w_p = np.asarray(W_patch, dtype=np.float64)
    b_p = np.asarray(patch_bias, dtype=np.float64)
    cls_tok = np.asarray(cls_token, dtype=np.float64)
    pos_emb = np.asarray(pos_embed, dtype=np.float64)
    w_h = np.asarray(W_head, dtype=np.float64)

    B, H, W, C_in = img.shape
    P = patch_size

    # helper function
    def _layer_norm(u: np.ndarray) -> np.ndarray:
        mean = np.mean(u, axis=-1, keepdims=True)
        var = np.var(u, axis=-1, keepdims=True)
        return (u - mean) / np.sqrt(var + 1e-6)

    def _gelu(val: np.ndarray) -> np.ndarray:
        return 0.5 * val * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (val + 0.044715 * (val ** 3))))

    # ==========================================
    # STAGE 1: PATCH EXTRACTION & LINEAR PROJECTION
    # ==========================================
    n_h = H // P
    n_w = W // P
    N = n_h * n_w

    # remove excess edge pixels that do not form complete image segments.
    H_valid = n_h * P
    W_valid = n_w * P
    img_valid = img[:, :H_valid, :W_valid, :]

    # decompose the image grid and permute the axes to ensure a row-major scanning order.
    patches = img_valid.reshape(B, n_h, P, n_w, P, C_in).transpose(0, 1, 3, 2, 4, 5)
    patches_flat = patches.reshape(B, N, P * P * C_in)

    # projection the hidden embedded space D
    tokens = patches_flat @ w_p + b_p

    # ==========================================
    # STAGE 2: APPEND [CLS] TOKEN AND ADD POS EMBEDDING
    # ==========================================
    D = tokens.shape[-1]
    cls_broadcast = np.broadcast_to(cls_tok, (B, 1, D))
    tokens = np.concatenate([cls_broadcast, tokens], axis=1)
    tokens = tokens + pos_emb

    # ==========================================
    # STAGE 3: THROUGH THE ENCODER BLOCKS
    # ==========================================
    d_h = D // num_heads

    for block in encoder_weights:
        wq = np.asarray(block["Wq"], dtype=np.float64)
        wk = np.asarray(block["Wk"], dtype=np.float64)
        wv = np.asarray(block["Wv"], dtype=np.float64)
        wo = np.asarray(block["Wo"], dtype=np.float64)
        w1 = np.asarray(block["W1"], dtype=np.float64)
        w2 = np.asarray(block["W2"], dtype=np.float64)

        # Multi-Head Attention (Pre-LN)
        seq_len = tokens.shape[1]
        norm1 = _layer_norm(tokens)

        Q = (norm1 @ wq).reshape(B, seq_len, num_heads, d_h).transpose(0, 2, 1, 3)
        K = (norm1 @ wk).reshape(B, seq_len, num_heads, d_h).transpose(0, 2, 1, 3)
        V = (norm1 @ wv).reshape(B, seq_len, num_heads, d_h).transpose(0, 2, 1, 3)

        scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_h)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        attn_out = (attn_weights @ V).transpose(0, 2, 1, 3).reshape(B, seq_len, D)
        tokens = tokens + (attn_out @ wo)

        # MLP (Pre-LN)
        norm2 = _layer_norm(tokens)
        mlp_out = _gelu(norm2 @ w1) @ w2
        tokens = tokens + mlp_out

    # ==========================================
    # STAGE 4: CLASSIFICATION HEAD
    # ==========================================
    # extract the state of the [CLS] token at position 0
    cls_final = tokens[:, 0, :]
    cls_norm = _layer_norm(cls_final)
    logits = cls_norm @ w_h

    return logits