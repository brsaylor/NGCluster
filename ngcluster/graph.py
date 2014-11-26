"""
Functions for creating neighborhood graphs from expression data
"""

import numpy as np
from numba import jit

def threshold_graph(data, threshold):
    """
    Create a threshold neighbor graph from the given expression data. The
    correlation coefficient is used as a similarity measure.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    threshold : float
        Two genes will be connected by an edge if their correlation coefficient
        is greater than the threshold.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    # Compute correlation matrix, replacing NaN (due to all-zero rows) with 0
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(data))

    # Compute the adjacency matrix
    adj = (corr > threshold)
    np.fill_diagonal(adj, 0)

    return adj

def nearest_neighbor_graph(data):
    """
    Create a nearest neighbor graph from the given expression data. The
    correlation coefficient is used as a similarity measure.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    # Compute correlation matrix, replacing NaN (due to all-zero rows) with 0
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(data))

    # Prevent any vertex from being its own nearest neighbor
    np.fill_diagonal(corr, -np.inf)

    nearest_neighbors = corr.argmax(1)

    # TODO: Find a fast numpy way to do this
    adj = np.zeros(corr.shape, dtype=bool)
    for i, j in enumerate(nearest_neighbors):
        adj[i][j] = True
        adj[j][i] = True

    return adj

def relative_neighborhood_graph(data):
    """
    Create a relative neighborhood graph from the given expression data. The
    correlation coefficient is used as a similarity measure.

    Parameters
    ----------
    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    Returns
    -------
    ndarray
        An adjacency matrix represented by an n*n array of boolean values.
    """

    # Compute correlation matrix, replacing NaN (due to all-zero rows) with 0
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(data))

    adj = np.zeros(corr.shape, dtype=bool)

    # TODO: Improvement over this basic algorithm is possible (Supowit 1983)
    @jit(nopython=True)
    def build_graph(corr, adj):
        # For each pair of points (p, q)...
        for p in range(corr.shape[0] - 1):
            for q in range(p + 1, corr.shape[0]):
                # ...if there is no other point z closer to both p and q than
                # they are to each other...
                for z in range(corr.shape[0]):
                    if z == p or z == q:
                        continue
                    if corr[p, z] > corr[p, q] and corr[q, z] > corr[p, q]:
                        # (don't add an edge)
                        break
                else:
                    # ...then add edge (p, q).
                    adj[p, q] = adj[q, p] = True
        return adj

    return build_graph(corr, adj)
