import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: dict with keys "output" and "mode"
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    gamma1 = np.asarray(gamma1, dtype=np.float64)
    beta1 = np.asarray(beta1, dtype=np.float64)
    gamma2 = np.asarray(gamma2, dtype=np.float64)
    beta2 = np.asarray(beta2, dtype=np.float64)

    eps = 1e-5

    def bn(data, gamma, beta):
        mean = np.mean(data, axis=0)
        var = np.var(data, axis=0)
        data_norm = (data - mean) / np.sqrt(var + eps)
        return gamma * data_norm + beta

    if mode == "post": # Conv -> BN -> ReLU -> Conv -> BN -> Add Skip -> ReLU
        h1 = np.dot(x, W1)
        h1 = bn(h1, gamma1, beta1)
        h1 = np.maximum(0, h1)

        h2 = np.dot(h1, W2)
        h2 = bn(h2, gamma2, beta2)

        output = np.maximum(0, h2 + x)

    elif mode == "pre": # BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Add Skip
        h1 = bn(x, gamma1, beta1)
        h1 = np.maximum(0, h1)
        h1 = np.dot(h1, W1)

        h2 = bn(h1, gamma2, beta2)
        h2 = np.maximum(0, h2)
        h2 = np.dot(h2, W2)

        output = h2 + x

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return {
        "output": output,
        "mode": mode
    }