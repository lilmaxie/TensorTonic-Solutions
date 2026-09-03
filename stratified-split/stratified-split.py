import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # init a seeded random number generator
    rng = np.random.default_rng(seed)

    # get a list of unique classes (sorted)
    unique_classes = np.unique(y)

    train_indices = []
    test_indices = []

    # iterate through the layers to perform stratification
    for c in unique_classes:
        # get all sample indices belonging to class c
        c_indices = np.flatnonzero(y==c)
        n_c = len(c_indices)

        # calculate the number of test samples: round(n_c * test_size)
        n_test = int(round(n_c * test_size))

        # if the class has more than one sample, cap n_test at n_c - 1.
        if n_c > 1:
            n_test = min(n_test, n_c-1)

        # randomly shuffle the indices of class c
        shuffled_c_indices = rng.permutation(c_indices)

        # train/test split
        test_indices.append(shuffled_c_indices[:n_test])
        train_indices.append(shuffled_c_indices[n_test:])

    # combine the indices from all classes and sort them in ascending order
    test_idx = np.sort(np.concatenate(test_indices))
    train_idx = np.sort(np.concatenate(train_indices))

    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx]
    }