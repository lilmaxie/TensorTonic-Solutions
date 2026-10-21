import numpy as np

def unet_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    """
    Returns float64 per-pixel segmentation logits in NHWC layout.
    """
    # Ensure the input tensor is in float64 format
    x_arr = np.asarray(x, dtype=np.float64)

    # 2D same-padding convolution helper function with ReLU (reused across all blocks)
    def _conv2d_same_relu(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        B, H, W, _ = inp.shape
        K_h, K_w, _, C_out = kernel.shape
        pad_h = K_h // 2
        pad_w = K_w // 2

        # Symmetric zero padding for height and width
        inp_padded = np.pad(
            inp,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0.0
        )

        out = np.zeros((B, H, W, C_out), dtype=np.float64)

        # Space scanning and multidimensional contraction computation
        for i in range(H):
            for j in range(W):
                patch = inp_padded[:, i : i + K_h, j : j + K_w, :]
                out[:, i, j, :] = np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias

        return np.maximum(0.0, out)

    # 1: ENCODER BLOCK
    enc1 = _conv2d_same_relu(
        x_arr,
        np.asarray(weights["enc_kernel1"], dtype=np.float64),
        np.asarray(weights["enc_bias1"], dtype=np.float64)
    )
    skip = _conv2d_same_relu(
        enc1,
        np.asarray(weights["enc_kernel2"], dtype=np.float64),
        np.asarray(weights["enc_bias2"], dtype=np.float64)
    )

    # Non-overlapping 2x2 max pooling via vectorized reshaping
    B, H, W, C = skip.shape
    pooled = skip.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))

    # 2: BOTTLENECK (BRIDGE)
    bridge1 = _conv2d_same_relu(
        pooled,
        np.asarray(weights["bridge_kernel1"], dtype=np.float64),
        np.asarray(weights["bridge_bias1"], dtype=np.float64)
    )
    bridge2 = _conv2d_same_relu(
        bridge1,
        np.asarray(weights["bridge_kernel2"], dtype=np.float64),
        np.asarray(weights["bridge_bias2"], dtype=np.float64)
    )

    # 3: UPSAMPLING & GHÉP KÊNH SKIP
    # Scale up the space by 2x using pixel repetition
    bridge_up = np.repeat(np.repeat(bridge2, 2, axis=1), 2, axis=2)

    # Linear projection and ReLU activation
    W_up = np.asarray(weights["W_up"], dtype=np.float64)
    b_up = np.asarray(weights["b_up"], dtype=np.float64)
    u = np.maximum(0.0, bridge_up @ W_up + b_up)

    # Center Crop tensor skip according to the spatial dimensions of u
    _, h_u, w_u, _ = u.shape
    _, h_s, w_s, _ = skip.shape
    top = (h_s - h_u) // 2
    left = (w_s - w_u) // 2
    skip_cropped = skip[:, top : top + h_u, left : left + w_u, :]

    # Multiplexing: Skip placed before, Decoder placed after
    concat = np.concatenate([skip_cropped, u], axis=-1)

    # 4: DECODER BLOCK
    dec1 = _conv2d_same_relu(
        concat,
        np.asarray(weights["dec_kernel1"], dtype=np.float64),
        np.asarray(weights["dec_bias1"], dtype=np.float64)
    )
    dec2 = _conv2d_same_relu(
        dec1,
        np.asarray(weights["dec_kernel2"], dtype=np.float64),
        np.asarray(weights["dec_bias2"], dtype=np.float64)
    )

    # 5: OUTPUT LAYER (RAW LOGITS)
    W_out = np.asarray(weights["W_out"], dtype=np.float64)
    b_out = np.asarray(weights["b_out"], dtype=np.float64)
    logits = dec2 @ W_out + b_out

    return logits