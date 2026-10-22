import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    img = np.asarray(image, dtype=np.float64)
    w = np.asarray(W_proj, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)

    B, H, W, C = img.shape
    P = patch_size

    # determine the number of complete patches along each dimension
    n_h = H//P
    n_w = W//P
    N = n_h*n_w

    # remove any excess material that doesn't create enough patches at the bottom and right edges
    H_valid = n_h*P
    W_valid = n_w*P
    img_valid = img[:, :H_valid, :W_valid, :]

    # extract patches and permute axes to row-major order
    # split dimensions: (B, n_h, P, n_w, P, C) and permute axes: (B, n_h, n_w, P, P, C)
    patches = img_valid.reshape(B, n_h, P, n_w, P, C).transpose(0, 1, 3, 2, 4, 5)

    # flatten each patch into a 1D vector: shape (B, N, P*P*C)
    patches_flat = patches.reshape(B, N, P*P*C)

    # linear projection to D-dimensional embedding space
    # (B, N, P*P*C) @ (P*P*C, D) + (D,) -> (B, N, D)
    embeddings = patches_flat@w + b

    return embeddings
    