"""
Functions for creating neighborhood graphs from expression data
"""

from math import isfinite

import numpy as np
from numpy import inf
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
from numba import jit

def distance_matrix(data, metric):
    """
    Create a square, symmetric distance matrix using the given distance metric
    over the rows of data.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.
    
    metric : string
        The distance metric to use if `dmatrix` is not supplied. Any metric
        accepted by scipy.spatial.distance.pdist can be used - for example,
        'euclidean', 'correlation', 'cosine'.
    
    Returns
    -------
    ndarray
        An n*n matrix of pairwise distances.
    """
    with np.errstate(invalid='ignore'):
        dist = pdist(data, metric)
    dist[np.isnan(dist)] = inf
    dist = squareform(dist)
    return dist

@jit(nopython=True)
def count_edges(adj):
    """
    Count the number of edges in the given graph.

    Parameters
    ----------
    adj : ndarray
        An adjacency matrix represented by an n*n array of boolean values.

    Returns
    -------
    int
        The number of edges in the graph.
    """
    count = 0
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[0]):
            count += adj[i, j]
    return count

def threshold_graph(data, threshold, metric='correlation'):
    """
    Create a threshold neighbor graph from the given expression data.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    threshold : float
        Two genes will be connected by an edge if their distance is less than
        the threshold (or, if metric is 'correlation', their correlation
        coefficient is greater than the threshold).

    metric : string, optional
        The distance metric to be used. Any metric accepted by
        scipy.spatial.distance.pdist can be used - for example, 'euclidean'
        'correlation' (default), 'cosine'.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    if metric == 'correlation':
        threshold = 1 - threshold

    dist = distance_matrix(data, metric)

    # Compute the adjacency matrix
    adj = (dist < threshold)
    np.fill_diagonal(adj, 0)

    return adj

def nearest_neighbor_graph(data, metric='correlation'):
    """
    Create a nearest neighbor graph from the given expression data.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    metric : string, optional
        The distance metric to be used. Any metric accepted by
        scipy.spatial.distance.pdist can be used - for example, 'euclidean'
        'correlation' (default), 'cosine'.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    dist = distance_matrix(data, metric)

    # Prevent any vertex from being its own nearest neighbor
    np.fill_diagonal(dist, inf)

    nearest_neighbors = dist.argmin(1)

    adj = np.zeros(dist.shape, dtype=bool)

    # TODO: @jit the loop
    for i, j in enumerate(nearest_neighbors):
        if dist[i, j] == inf:
            continue
        adj[i, j] = True
        adj[j, i] = True

    return adj

def relative_neighborhood_graph(data, metric='correlation'):
    """
    Create a relative neighborhood graph from the given expression data. The
    correlation coefficient is used as a similarity measure.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    metric : string, optional
        The distance metric to be used. Any metric accepted by
        scipy.spatial.distance.pdist can be used - for example, 'euclidean'
        'correlation' (default), 'cosine'.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    dist = distance_matrix(data, metric)

    adj = np.zeros(dist.shape, dtype=bool)

    # TODO: Improvement over this basic algorithm is possible (Supowit 1983)
    @jit(nopython=True)
    def build_graph(dist, adj):
        # For each pair of points (p, q)...
        for p in range(dist.shape[0] - 1):
            for q in range(p + 1, dist.shape[0]):

                if not isfinite(dist[p, q]):
                    continue

                # ...if there is no other point z closer to both p and q than
                # they are to each other...
                for z in range(dist.shape[0]):
                    if (z == p or z == q or
                            not isfinite(dist[p, z]) or not isfinite(dist[q, z])):
                        continue
                    if dist[p, z] < dist[p, q] and dist[q, z] < dist[p, q]:
                        # (don't add an edge)
                        break
                else:
                    # ...then add edge (p, q).
                    adj[p, q] = adj[q, p] = True
        return adj

    return build_graph(dist, adj)

def gabriel_graph(data):
    """
    Create a Gabriel graph from the given expression data. Euclidean distance is
    used to compute the graph.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    # FIXME: This is slow! More efficient algorithms exist.

    rows, cols = data.shape
    midpoint = np.empty(cols)
    adj = np.zeros((rows, rows), dtype=bool)

    @jit(nopython=True)
    def build_graph(data, rows, cols, midpoint, adj):

        # For each pair of points (p, q)...
        for p in range(rows - 1):
            for q in range(p + 1, rows):

                # ...calculate their midpoint and their distance 'radius' to it.
                radius = 0
                for i in range(cols):
                    midpoint[i] = (data[p, i] + data[q, i]) * 0.5
                    radius += (data[p, i] - midpoint[i]) ** 2
                radius **= 0.5

                # For each other point z...
                for z in range(rows):
                    if z == p or z == q:
                        continue

                    # ...calculate its distance from the midpoint.
                    dist = 0
                    for i in range(cols):
                        dist += (data[z, i] - midpoint[i]) ** 2
                    dist **= 0.5

                    # If that distance is less than 'radius', then p and q are not
                    # Gabriel neighbors.
                    if dist <= radius:
                        break
                else:
                    # Since there is no other point z is within 'radius' distance of
                    # the midpoint, p and q are Gabriel neighbors.
                    adj[p, q] = adj[q, p] = True
        return adj

    return build_graph(data, rows, cols, midpoint, adj)

def gabriel_graph_delaunay(data):
    """
    An attempt at a more efficient algorithm for building the Gabriel graph by
    taking advantage of the fact that it is a subgraph of the Delaunay
    tesselation. It passes all unit tests except the first (which has too few
    data points). Unfortunately, it consumes a huge amount of memory when run
    on the real data.
    """

    rows, cols = data.shape
    midpoint = np.empty(cols)
    adj = np.zeros((rows, rows), dtype=bool)

    delaunay = Delaunay(data)
    indices, indptr = delaunay.vertex_neighbor_vertices

    @jit(nopython=True)
    def build_graph(data, rows, cols, midpoint, adj, indices, indptr):

        # For each pair of points (p, q) that are neighbors in the Delaunay
        # triangulation...
        for p in range(rows - 1):
            for i in range(indices[p], indices[p+1]):
                q = indptr[i]

                # ...calculate their midpoint and their distance 'radius' to it.
                radius = 0
                for i in range(cols):
                    midpoint[i] = (data[p, i] + data[q, i]) * 0.5
                    radius += (data[p, i] - midpoint[i]) ** 2
                radius **= 0.5

                # For each other point z...
                for z in range(rows):
                    if z == p or z == q:
                        continue

                    # ...calculate its distance from the midpoint.
                    dist = 0
                    for i in range(cols):
                        dist += (data[z, i] - midpoint[i]) ** 2
                    dist **= 0.5

                    # If that distance is less than 'radius', then p and q are not
                    # Gabriel neighbors.
                    if dist <= radius:
                        break
                else:
                    # Since there is no other point z is within 'radius' distance of
                    # the midpoint, p and q are Gabriel neighbors.
                    adj[p, q] = adj[q, p] = True
        return adj

    return build_graph(data, rows, cols, midpoint, adj, indices, indptr)
