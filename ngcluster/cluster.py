"""
Clustering functions
"""

from random import randint

import numpy as np
from scipy.cluster import vq

def random_clusters(data, k):
    """
    Randomly assign the data to `k` clusters.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    k : int
        The number of clusters to generate.

    Returns
    -------
    ndarray
        A 1-dimensional array of length n, where each element is the cluster ID
        (from 0 to k-1) of the corresponding row in the input data.
    """
    return np.array([randint(0, k-1) for i in range(data.shape[0])])

def kmeans(data, k):
    """
    Perform k-means clustering on a set of expression vectors forming k clusters.

    The initial centroids are chosen at random. k-means is iterated until the
    distortion changes by at most 0.00001 between iterations. The entire
    algorithm is run 30 times, and the result with the lowest distortion is
    returned.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    k : int
        The number of clusters to generate.

    Returns
    -------
    ndarray
        A 1-dimensional array of length n, where each element is the cluster ID
        (from 0 to k-1) of the corresponding row in the input data.
    """

    centroids, _ = vq.kmeans(data, k, iter=30, thresh=1e-05)
    clusters, _ = vq.vq(data, centroids)

    return clusters

def graph_clusters(data, fn, fn_args=[], fn_kwargs={}, threshold=5):
    """
    Perform local-peak clustering on the data based on the given graph function.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    fn : function
        The graph function to use. It is expected to take the expression data as
        its first argument and return an adjacency matrix.

    fn_args : list, optional
        A list of arguments to supply to `fn` following the data argument.

    fn_kwargs : dict, optional
        A list of keyword arguments to supply to `fn`.

    threshold : int, optional
        The degree threshold for identifying local peaks.

    Returns
    -------
    ndarray
        A 1-dimensional array of length n, where each element is the cluster ID
        (from 0 to k-1) of the corresponding row in the input data.
    """

    # Compute the graph
    adj = fn(data, *fn_args, **fn_kwargs)

    # Compute the degree of each node
    deg = adj.sum(1) # sum(1) counts the number of True elements in each row

    # Find the local peaks
    # Each peak is stored as its index into the data array
    # TODO: Find a faster numpy-based way to do this
    # - find indices where deg > threshold: (degree > threshold).nonzero()[0]
    peaks = []
    for i, degree in enumerate(deg):
        if degree <= threshold:
            continue
        for j, adjacent in enumerate(adj[i]):
            if not adjacent:
                continue
            if deg[j] >= degree:
                break
        else:
            peaks.append(i)

    # Find the clusters (peaks and neighbors)
    # Clusters is a list of lists of integers, each of which is an index into
    # the data array
    # TODO: Find a faster numpy-based way to do this
    clusters = np.empty(data.shape[0], dtype=int)
    clusters.fill(-1)
    cluster_id = 0
    for i in peaks:
        clusters[i] = cluster_id
        for j, adjacent in enumerate(adj[i]):
            if adjacent:
                clusters[j] = cluster_id
        cluster_id += 1

    return clusters
