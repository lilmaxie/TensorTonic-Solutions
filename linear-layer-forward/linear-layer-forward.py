def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X_arr = np.asarray(X)
    W_arr = np.asarray(W)
    b_arr = np.asarray(b)

    Y = (X_arr @ W_arr + b_arr).astype(float).tolist()
    
    return Y 