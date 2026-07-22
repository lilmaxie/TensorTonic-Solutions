def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    W = np.asarray(W)

    L = np.sqrt(6/(fan_in + fan_out))

    W_scaled = W * (2*L) - L

    return W_scaled.astype(float).tolist()