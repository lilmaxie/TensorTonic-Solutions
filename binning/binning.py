def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    if min(values) == max(values):
        return [0] * len(values)

    bin_width = (max(values) - min(values)) / num_bins

    result = []
    for v in values:
        bin_idx = int((v - min(values))/ bin_width)
        bin_idx = min(bin_idx, num_bins - 1)
        result.append(bin_idx)

    return result