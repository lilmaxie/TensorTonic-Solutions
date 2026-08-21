import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.asarray(x, dtype=np.float64)
    conv1 = np.asarray(conv1, dtype=np.float64)
    W1_b1 = np.asarray(W1_b1, dtype=np.float64)
    W2_b1 = np.asarray(W2_b1, dtype=np.float64)
    W1_b2 = np.asarray(W1_b2, dtype=np.float64)
    W2_b2 = np.asarray(W2_b2, dtype=np.float64)
    Ws_b2 = np.asarray(Ws_b2, dtype=np.float64)
    fc = np.asarray(fc, dtype=np.float64)

    # init conv --> ReLU
    out = np.maximum(0, np.dot(x, conv1))

    # block 1 (identity shortcut)
    # flow: conv(W1) -> ReLU -> conv(W2) -> add(Identity) -> ReLU
    h1 = np.maximum(0, np.dot(out, W1_b1))
    h1 = np.dot(h1, W2_b1)
    out = np.maximum(0, h1 + out)

    # block 2 (projection shortcut)
    # flow: conv(W1) -> ReLU -> conv(W2) -> add(Proj(Ws)) -> ReLU
    shortcut2 = np.dot(out, Ws_b2)
    h2 = np.maximum(0, np.dot(out, W1_b2))
    h2 = np.dot(h2, W2_b2)
    out = np.maximum(0, h2 + shortcut2)

    # FC layer
    # flow: linear (FC) -> logits
    logits = np.dot(out, fc)
    
    return logits