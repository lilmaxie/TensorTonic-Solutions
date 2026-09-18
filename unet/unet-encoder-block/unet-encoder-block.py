import numpy as np

def unet_encoder_block(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> dict:
    """
    Returns pooled and skip as float64 arrays in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    k1 = np.asarray(kernel1, dtype=np.float64)
    b1 = np.asarray(bias1, dtype=np.float64)
    k2 = np.asarray(kernel2, dtype=np.float64)
    b2 = np.asarray(bias2, dtype=np.float64)

    def _conv2d_same_relu(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        B, H, W, _ = inp.shape
        K_h, K_w, _, C_out = kernel.shape
        pad_h = K_h//2
        pad_w = K_w//2

        # zero padding
        inp_padded = np.pad(
            inp,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode = 'constant',
            constant_values = 0.0
        )

        out = np.zeros((B, H, W, C_out), dtype=np.float64)

        # scan the spatial window and compute the tensor contraction
        for i in range(H):
            for j in range(W):
                # patch shape (B, K_h, K_w, C_in)
                patch = inp_padded[:, i : i+K_h, j : j+K_w, :]
                # np.tensordot contracts axes (1, 2, 3) of the patch with axes (0, 1, 2) of the kernel
                out[:, i, j, :] = np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias

        return np.maximum(0.0, out)

    # 2 consecutive convolutions with ReLU are performed to obtain Skip Tensor S.
    conv1 = _conv2d_same_relu(x, k1, b1)
    skip = _conv2d_same_relu(conv1, k2, b2)

    # non-overlapping 2x2 maxpooling using reshape vectorize
    B, H, W, C = skip.shape
    # Separate pixel pairs (2x2) along the height and width dimensions: (B, H//2, 2, W//2, 2, C) and take the maximum along window axes 2 and 4
    pooled = skip.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

    return {
        "pooled": pooled,
        "skip": skip
    }
    