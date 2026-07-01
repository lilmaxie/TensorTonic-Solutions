def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    # we give x initialize
    x = x0

    # basic training steps from scratch
    for _ in range(steps):
        gradient = 2*a*x + b

        x = x - lr*gradient

    return x