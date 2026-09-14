import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x, dtype=float)

    # generate an array of random real numbers in the range [0.0, 1.0) with the same shape as x
    if rng is not None:
        rand_vals = rng.random(x.shape)
    else:
        rand_vals = np.random.random(x.shape)

    # calculate the compensation scaling factor 1 / (1 - p)
    scale = 1.0 / (1.0-p)

    # retain positions with values ​​< (1 - p); drop the remaining positions
    # 0 to the dropped element and 1 / (1 - p) to the retained element
    keep_mask = rand_vals < (1.0 - p)
    dropout_pattern = keep_mask.astype(float) * scale

    # element-wise between the input and the dropout_pattern
    output = x * dropout_pattern

    return output, dropout_pattern