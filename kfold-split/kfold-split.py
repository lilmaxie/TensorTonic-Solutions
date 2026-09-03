import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(N)
    else:
        indices = np.arange(N)

    # split the array of indices into k balanced folds (distributing any remainder to the initial folds)
    folds = np.array_split(indices, k)

    # create a list of k pairs (train_idx, val_idx)
    splits = []
    for i in range(k):
        val_idx = folds[i]

        # combine all folds other than the i-th fold to form the training set
        train_folds = folds[:i] + folds[i+1:]
        train_idx = np.concatenate(train_folds)

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx
        })

    return splits