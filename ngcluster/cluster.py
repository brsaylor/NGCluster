"""
Clustering functions
"""

from random import randint

import numpy as np
from scipy.cluster import vq
from numba import jit

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

def graph_clusters_expanding(data, fn, fn_args=[], fn_kwargs={}, threshold=5,
        max_clusters=1000, iterations=1):
    """
    Compute a graph-based clustering of the data. Clusters are initialized to
    include high-degree nodes (with degree above `threshold`, and without
    neighbors of higher degree) and their neighbors. In each subsequent
    iteration, clusters are expanded outward to include neighbors of neighbors,
    etc.

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
        The degree threshold for above which cluster initialization nodes are
        chosen. (Default: 5)

    max_clusters : int, optional
        The maximum number of clusters to create. (Default: 1000)

    iterations : int, optional
        The number of times to expand outward to include additional neighbors of
        clustered nodes. A value of 1 (default) results in clusters that include
        a central node and its immediate neighbors. A value of 2 results in
        clusters that also include neighbors of neighbors, and so on.

    Returns
    -------
    ndarray
        A 1-dimensional array of length n, where each element is the cluster ID
        (from 0 to k-1) of the corresponding row in the input data.
    """

    # TODO: This implementation is not very efficient: it iterates through more
    # nodes than need to be considered. An adjacency list representation of the
    # graph would be more efficient, and maybe also some way of separating
    # clustered from unclustered nodes.
    
    # Compute the graph
    adj = fn(data, *fn_args, **fn_kwargs)

    # Compute the degree of each node
    deg = adj.sum(1) # sum(1) counts the number of True elements in each row

    clusters = np.empty(data.shape[0], dtype=int)
    clusters.fill(-1)

    # Array of gene indices, sorted by node degree non-increasing
    nodes = np.argsort(-deg)

    @jit(nopython=True)
    def do_clustering(
            nodes, adj, deg, clusters, threshold, max_clusters, iterations):

        cluster_id = -1

        # Used for marking nodes for later assignment to a particular cluster.
        # The marker value is subtracted from the cluster id to be assigned,
        # resulting in a negative value that causes the node to be treated as
        # unclustered.
        marker = 9999

        # Create clusters from high-degree nodes and their immediate neighbors
        for i in range(len(nodes)):
            p = nodes[i]

            # If we've reached the degree threshold or the maximum number of
            # clusters, stop creating clusters
            if deg[p] <= threshold or cluster_id + 1 == max_clusters:
                break

            # If node p is not in a cluster...
            if clusters[p] < 0:

                # ...and has no neighbors of higher degree...
                for j in range(i):
                    q = nodes[j]
                    has_higher_degree_neighbor = False
                    if adj[p, q] and deg[q] > deg[p]:
                        has_higher_degree_neighbor = True
                        break
                if has_higher_degree_neighbor:
                    continue

                # ...assign it to a new cluster...
                cluster_id += 1
                clusters[p] = cluster_id

                # ...and assign all of its unclustered neighbors to the cluster.
                for j in range(len(nodes)):
                    q = nodes[j]
                    if adj[p, q] and clusters[q] < 0:
                        clusters[q] = cluster_id

        # Expand the clusters outward (neighbors of neighbors, etc.)
        for iteration in range(1, iterations):

            # For each unclustered node p...
            for i in range(len(nodes)):
                p = nodes[i]
                if clusters[p] >= 0:
                    continue

                # ...if p is adjacent to a clustered node q...
                for j in range(len(nodes)):
                    q = nodes[j]
                    if adj[p, q] and clusters[q] >= 0:
                        # ...mark p for assignment to the same cluster as q.
                        clusters[p] = clusters[q] - marker
                        break

            # Assign the nodes marked for cluster assignment
            for i in range(len(nodes)):
                p = nodes[i]
                if clusters[p] < -1:
                    clusters[p] += marker

    do_clustering(nodes, adj, deg, clusters, threshold, max_clusters, iterations)

    return clusters
