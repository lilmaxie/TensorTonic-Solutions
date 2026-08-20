import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)

    if n == 0 or n_bins <= 0:
        return 0.0

    # determine the bin index for each sample: floor(p * n_bins)
    # p = 1.0 is assigned to the last bin (n_bins - 1)
    bin_indices = np.floor(y_pred*n_bins).astype(int)
    bin_indices = np.minimum(bin_indices, n_bins-1)

    # calculate the deviation between accuracy and confidence for each bin
    ece = 0.0
    for m in range(n_bins):
        # take the masks of the bin m samples
        mask = (bin_indices == m)
        bin_size = np.sum(mask)

        if bin_size == 0:
            continue

        # calculate the average accuracy and confidence of bin m
        acc_m = np.mean(y_true[mask])
        conf_m = np.mean(y_pred[mask])

        # contribution of bin m to the total ECE
        ece += (bin_size/n) * abs(acc_m - conf_m)

    return float(ece)