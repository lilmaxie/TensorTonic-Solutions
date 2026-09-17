import math
import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test, dtype=float)

    n_samples, n_features = X_tr.shape
    unique_classes = np.unique(y_tr)
    unique_classes.sort()

    eps = 1e-9

    # training Step (Fitting): Estimating Prior, Mean, and Population Variance
    model_params = []
    for c in unique_classes:
        X_c = X_tr[y_tr == c]
        n_c = len(X_c)

        # log prior
        log_prior = math.log(n_c/n_samples)

        # mean and population var (ddof=0)
        mean_c = np.mean(X_c, axis=0)
        var_c = np.var(X_c, axis=0) + eps

        model_params.append({
            "class": int(c),
            "log_prior": log_prior,
            "mean": mean_c,
            "var": var_c
        })

    # inference Step: Calculate the log-posterior for each test sample
    predictions = []
    two_pi = 2.0 * math.pi

    for x in X_te:
        best_class = None
        best_log_posterior = -float("inf")

        for param in model_params:
            mean = param["mean"]
            var = param["var"]

            # Log Gaussian Likelihood per feature
            log_likelihood = -0.5 * np.log(two_pi*var) - ((x-mean)**2) / (2.0*var)

            # aggregate log-posterior
            total_log_posterior = param["log_prior"] + np.sum(log_likelihood)

            # update the class with the highest log-posterior
            if total_log_posterior > best_log_posterior:
                best_log_posterior = total_log_posterior
                best_class = param["class"]

        predictions.append(int(best_class))

    return predictions