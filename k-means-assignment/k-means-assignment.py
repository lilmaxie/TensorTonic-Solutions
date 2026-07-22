import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.asarray(points)
    centroids = np.asarray(centroids)

    # points[:, None, :] --> (N, 1, D)
    # centroids[None, :, :] --> (1, K, D)
    # the broadcasting subtract will create (N, K, D) dimension matrix
    distance = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    assignments = np.argmin(distance, axis=1)

    return assignments.tolist()