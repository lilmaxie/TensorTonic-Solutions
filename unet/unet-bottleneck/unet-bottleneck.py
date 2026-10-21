import numpy as np

def unet_bottleneck(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                    kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 bottleneck features in NHWC layout.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    k1 = np.asarray(kernel1, dtype=np.float64)
    b1 = np.asarray(bias1, dtype=np.float64)
    k2 = np.asarray(kernel2, dtype=np.float64)
    b2 = np.asarray(bias2, dtype=np.float64)

    def _conv2d_same_relu(inp: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
        B, H, W, _ = inp.shape
        K_h, K_w, _, C_out = kernel.shape
        pad_h = K_h//2
        pad_w = K_w//2

        # apply zero padding along the two spatial axes, H and W
        inp_padded = np.pad(
            inp,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant',
            constant_values=0.0
        )

        out = np.zeros((B, H, W, C_out), dtype=np.float64)

        # sliding window scanning and multidimensional convolution
        for i in range(H):
            for j in range(W):
                patch = inp_padded[:, i : i+K_h, j : j+K_w, :]
                out[:, i, j, :] = np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias

        return np.maximum(0.0, out)

    # Conv layers, each followed by ReLU
    conv1 = _conv2d_same_relu(x_arr, k1, b1)
    y = _conv2d_same_relu(conv1, k2, b2)

    return y