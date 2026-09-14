import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    # extract the dimensions of the input tensor and the filter
    N, C_in, H, W_in = x.shape
    C_out, _, K_H, K_W = W.shape

    # calculate output spatial dimensions (valid padding, stride = 1)
    H_out = H - K_H + 1
    W_out = W_in - K_W + 1

    # init the result array with shape (N, C_out, H_out, W_out)
    out = np.zeros((N, C_out, H_out, W_out), dtype=float)

    # iterate through each sample in the batch and each output filter
    for n in range(N):
        for c in range(C_out):
            kernel = W[c] # shape (C_in, K_H, K_W)
            bias = b[c]

            # slide the window across the H_out and W_out space
            for i in range(H_out):
                for j in range(W_out):
                    # slice the receptive field across all C_in channels: shape (C_in, K_H, K_W)
                    patch = x[n, :, i:i+K_H, j:j+K_W]
                    # perform element-wise multiplication, aggregate across all channels, and add the bias
                    out[n, c, i, j] = np.sum(patch * kernel) + bias

    return out