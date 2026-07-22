import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes, counts = np.unique(y_train, return_counts=True)

    majority_class = classes[np.argmax(counts)]

    n_samples = len(X_test) if X_test.ndim > 0 else 0

    predictions = np.full(n_samples, fill_value=majority_class, dtype=int)

    return predictions