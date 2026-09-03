import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)

    # init an index array [0, 1, ..., n_samples - 1] and a seeded random number generator
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)

    # shuffle only once
    rng.shuffle(indices)

    # iterate through the index array in steps of `batch_size`
    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx: start_idx + batch_size]

        # if drop_last=True and the last batch is smaller than batch_size, discard it
        if drop_last and len(batch_idx) < batch_size:
            continue

        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        yield (X_batch, y_batch)