def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    m = len(s1)
    n = len(s2)

    dp = [
        [0] * (n+1)
        for _ in range(m+1)
    ]

    for i in range(m+1):
        dp[i][0] = i

    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1

            delete = dp[i-1][j] + 1
            insert = dp[i][j-1] + 1
            replace = dp[i-1][j-1] + cost

            dp[i][j] = min(delete, insert, replace)

    return dp[m][n]