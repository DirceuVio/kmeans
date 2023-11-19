import numpy as np
from scipy.spatial.distance import euclidean


class KMeans:
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    k : int
        The number of clusters.

    Attributes
    ----------
    k : int
        The number of clusters.
    centroids : dict
        A dictionary where keys are cluster indices and values are centroid arrays.
    n_iter : int
        The number of iterations performed during fitting.

    Methods
    -------
    fit(X, threshold=0.001, max_iter=100)
        Fit the K-Means algorithm to the input data.

    predict(X)
        Predict the closest cluster for each data point.

    Properties
    ----------
    centroids : dict
        Get the centroids of the clusters.
    n_iter : int
        Get the number of iterations performed during fitting.
    """

    def __init__(self, k: int) -> None:
        """
        Initialize KMeans instance.

        Parameters
        ----------
        k : int
            The number of clusters.
        """
        self.k = k
        self._centroids = {}

    def fit(self, X: np.ndarray, threshold: float = 0.001, max_iter: int = 100) -> None:
        """
        Fit the K-Means algorithm to the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        threshold : float, optional
            The convergence threshold, by default 0.001.
        max_iter : int, optional
            The maximum number of iterations, by default 100.
        """
        self._centroids = dict(
            enumerate(X[np.random.choice(X.shape[0], size=self.k, replace=False), :])
        )
        self._n_iter = 0
        start_error = np.inf

        while True:
            cluster_labels = [
                np.argmin([euclidean(row, i) for i in self.centroids.values()])
                for row in X
            ]
            error = np.sum(
                [
                    np.sum(
                        [
                            euclidean(row, self._centroids[label])
                            for row in X[cluster_labels == label]
                        ]
                    )
                    for label in set(cluster_labels)
                ]
            )

            if abs(error / start_error - 1) < threshold or self._n_iter == max_iter:
                break
            else:
                start_error = error

            self._centroids = dict(
                enumerate(
                    [X[cluster_labels == i].mean(axis=0) for i in set(cluster_labels)]
                )
            )
            self._n_iter += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each data point.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            Array of cluster labels.
        """
        return [
            np.argmin([euclidean(row, i) for i in self._centroids.values()])
            for row in X
        ]

    @property
    def centroids(self) -> dict:
        """
        Get the centroids of the clusters.

        Returns
        -------
        dict
            A dictionary where keys are cluster indices and values are centroid arrays.
        """
        return self._centroids

    @property
    def n_iter(self) -> int:
        """
        Get the number of iterations performed during fitting.

        Returns
        -------
        int
            The number of iterations.
        """
        return self._n_iter
