def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    H = len(X)
    W = len(X[0])
    p = pool_size
    s = stride

    # output size
    H_out = (H-p)//s + 1
    W_out = (W-p)//s + 1

    pooled_output = []

    # slide the window across the output rows and columns
    for i in range(H_out):
        row_res = []
        r_start = i * s
        r_end = r_start + p

        for j in range(W_out):
            c_start = j * s
            c_end = c_start + p

            # max value in pxp window
            max_val = max(
                X[r][c]
                for r in range(r_start, r_end)
                for c in range(c_start, c_end)
            )
            row_res.append(max_val)

        pooled_output.append(row_res)

    return pooled_output