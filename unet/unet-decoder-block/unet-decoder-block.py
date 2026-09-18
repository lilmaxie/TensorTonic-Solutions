import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray,
                       W_up: np.ndarray, b_up: np.ndarray,
                       kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 decoder features in NHWC layout.
    """
    x    = np.asarray(x,       dtype=np.float64)
    skip = np.asarray(skip,    dtype=np.float64)
    w_up = np.asarray(W_up,    dtype=np.float64)
    b_up = np.asarray(b_up,    dtype=np.float64)
    k1   = np.asarray(kernel1, dtype=np.float64)
    b1   = np.asarray(bias1,   dtype=np.float64)
    k2   = np.asarray(kernel2, dtype=np.float64)
    b2   = np.asarray(bias2,   dtype=np.float64)

    # upsampling by repeating pixels twice along the height (1) and width (2) axes
    x_up = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)

    # linear proj and ReLU act
    u = np.maximum(0.0, x_up@w_up + b_up)

    # center crop tensor skip base on u shape
    _, h_up, w_up_dim, _ = u.shape
    _, h_skip, w_skip, _ = skip.shape

    top = (h_skip - h_up) // 2
    left = (w_skip - w_up_dim) // 2
    skip_cropped = skip[:, top : top+h_up, left: left+w_up_dim, :]

    # concat channel
    concat = np.concatenate([skip_cropped, u], axis=-1)

    # 2D same-padding convolution with ReLU
    def _conv2d_same_relu(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        B, H, W, _ = inp.shape
        K_h, K_w, _, C_out = kernel.shape
        pad_h = K_h // 2
        pad_w = K_w // 2

        inp_padded = np.pad(
            inp,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0.0
        )

        out = np.zeros((B, H, W, C_out), dtype=np.float64)
        for i in range(H):
            for j in range(W):
                patch = inp_padded[:, i : i + K_h, j : j + K_w, :]
                out[:, i, j, :] = np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias

        return np.maximum(0.0, out)

    # passed through two consecutive convolutional layers
    conv1 = _conv2d_same_relu(concat, k1, b1)
    y = _conv2d_same_relu(conv1, k2, b2)

    return y