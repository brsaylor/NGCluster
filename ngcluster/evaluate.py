"""
Functions for evaluating clusterings
"""

from math import sqrt, isnan

import numpy as np
from scipy.spatial.distance import pdist, squareform
from numba import jit

class ClusterEvaluationError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def fom(clusters, hidden_data, adjust=True):
    """
    Calculate the 2-norm figure of merit as defined in Yeung et al., 2000.

    Parameters
    ----------
    clusters : ndarray
        A 1-dimensional array of length n, where n is the number of genes in the
        clustered expression data, and each element is the cluster ID of the
        corresponding gene, negative if that gene is not in any cluster.

    hidden_data : ndarray
        A 1-dimensional array containing the expression data for the condition
        that was excluded from the clustering.

    adjust : bool, optional
        If True (default), the adjusted 2-norm FOM will be returned; otherwise,
        the figure will not be adjusted for cluster count bias.

    Returns
    -------
    float
        The 2-norm figure of merit.

    Notes
    -----
    In this implementation, the number of genes *n* is set to the number of
    *clustered* genes, allowing for clusterings in which some genes are not in
    any cluster (which should be indicated by negative cluster IDs).
    """

    ssd = 0.  # Sum of squared differences
    num_clustered_genes = 0
    num_clusters = clusters.max() + 1

    if num_clusters <= 0:
        raise ClusterEvaluationError("There are no clusters")

    # Calculate the sum of squared deviations from cluster means
    for i in range(num_clusters):
        cluster_data = hidden_data[clusters == i]
        cluster_mean = cluster_data.mean()
        ssd += ((cluster_data - cluster_mean) ** 2).sum()
        num_clustered_genes += len(cluster_data)

    result = sqrt(ssd / float(num_clustered_genes))

    if adjust:
        if num_clustered_genes == num_clusters:
            raise ClusterEvaluationError(
                    "Adjusted FOM is undefined for 1 gene per cluster")
        result /= sqrt(float(num_clustered_genes - num_clusters) /
                float(num_clustered_genes))

    return result

def aggregate_fom(data, fn, fn_args=[], fn_kwargs={}, adjust=True):
    """
    Calculate the aggregate 2-norm figure of merit as defined in Yeung et al.,
    2000.

    For each condition, the corresponding column of data is removed, the
    remaining data is clustered using `fn`, and the resulting clusters are
    evaluated against the hidden data by `fom()`.

    Parameters
    ----------
    data : ndarray
        An n*m array of expression data for n genes under m conditions.

    fn : function
        The clustering function to use. It is expected to take the expression
        data as its first argument.

    fn_args : list, optional
        A list of arguments to supply to `fn` following the data argument.

    fn_kwargs : dict, optional
        A list of keyword arguments to supply to `fn`.

    adjust : bool, optional
        If True (default), the adjusted figure of merit will be used.
        
    Returns
    -------
    float
        The aggregate 2-norm figure of merit.
    """

    result = 0.

    for e in range(data.shape[1]):

        # Remove column e from the data to cluster
        data_to_cluster = data.compress(
                np.array([col != e for col in range(data.shape[1])]),
                axis=1)

        # Get the removed column of data
        hidden_data = data.take(e, axis=1)

        # Do the clustering
        clusters = fn(data_to_cluster, *fn_args, **fn_kwargs)

        # Add the FOM based on the clustering to the result
        result += fom(clusters, hidden_data, adjust)

    return result

@jit(nopython=True)
def rand_index(X, Y):
    """
    Calculate the Rand index of the two given clusterings X and Y.

    One or both of the clusterings can contain unclustered objects. If an object
    is unclustered in both X and Y, it is removed from consideration. If an
    object is clustered in X but not in Y, or vice versa, it will be included in
    the calculations and its (negative) cluster ID will be treated like any
    other.

    To compare X and Y based only on objects clustered in both, first set all
    cluster ID's in X to negative where they are negative in Y, and vice versa.

    Parameters
    ----------
    X : ndarray
        A 1-dimensional array of length n, where n is the number of genes in the
        clustered expression data, and each element is the cluster ID of the
        corresponding gene, negative if that gene is not in any cluster.

    Y : ndarray
        A different clustering of the same genes, represented similarly to X.

    Returns
    -------
    float
        The non-adjusted Rand index of X and Y.

    """

    agreements = 0.
    total_pairs = 0.
    n = len(X)

    for i in range(n - 1):

        # Skip gene i if both X and Y do not include it in a cluster
        if X[i] < 0 and Y[i] < 0:
            continue

        for j in range(i + 1, n):

            # Skip gene j if both X and Y do not include it in a cluster
            if X[j] < 0 and Y[j] < 0:
                continue

            if (    (X[i] == X[j] and Y[i] == Y[j]) or
                    (X[i] != X[j] and Y[i] != Y[j])):
                agreements += 1

            total_pairs += 1

    return agreements / total_pairs

def silhouette_widths(clusters, data, metric='euclidean', dmatrix=None):
    """
    Calculate the silhouette widths for the given clustering.

    Parameters
    ----------
    clusters : ndarray
        A 1-dimensional array of length n, where n is the number of genes in the
        clustered expression data, and each element is the cluster ID of the
        corresponding gene, negative if that gene is not in any cluster.

    data : ndarray, optional
        The expression data, required to calculate pairwise distances if
        if `dmatrix` is not supplied.

    metric : string, optional
        The distance metric to use if `dmatrix` is not supplied. Any metric
        accepted by scipy.spatial.distance.pdist can be used - for example,
        'euclidean' (default), 'correlation', 'cosine'.

    dmatrix : ndarray, optional
        A n by n distance matrix, required if `data` is not supplied.

    Returns
    -------
    ndarray
        A 1-dimensional array of length n, each element of which is the
        silhouette width of the corresponding gene.
    """

    if dmatrix is None:
        # Calculate distance matrix
        dmatrix = squareform(pdist(data, metric))

    widths = np.empty(len(clusters))  # Silhouette widths
    k = int(clusters.max() + 1)       # Number of clusters
    dsum = np.empty(k)                # Sum of distances by cluster
    dcount = np.empty(k)              # Count of objects by cluster

    @jit(nopython=True)
    def compute_widths(dmatrix, clusters, k, widths, dsum, dcount):
        n = len(dmatrix)

        # For each object 'i'...
        for i in range(n):

            # (if 'i' is not in a cluster, set its width to 0)
            if clusters[i] < 0:
                widths[i] = 0.
                continue

            for c in range(k):
                dsum[c] = 0.
                dcount[c] = 0.

            # ...calculate the average distance 'a' to all other objects 'j' in
            # the same cluster, and the average distance 'd[c]' to the objects
            # 'j' in each other cluster 'c'. Let 'b' be the minimum value of
            # 'd[c]' for object i.
            for j in range(n):
                if i == j:
                    continue
                c = clusters[j]

                # Skip 'j' if it is not in a cluster,
                # or if its distance to i is undefined.
                if c < 0 or isnan(dmatrix[i, j]):
                    continue

                dsum[c] += dmatrix[i, j]
                dcount[c] += 1

            # If 'i' is the only object in its cluster, then its silhouette
            # width is 0.
            if dcount[clusters[i]] == 0:
                widths[i] = 0.
                continue

            # Calculate 'a' and 'b'
            b = np.inf
            for c in range(k):
                if c == clusters[i]:
                    a = dsum[c] / dcount[c]
                else:
                    d = dsum[c] / dcount[c]
                    if d < b:
                        b = d

            # Silhouette width for object i
            widths[i] = (b - a) / max(a, b)

        return widths

    return compute_widths(dmatrix, clusters, k, widths, dsum, dcount)

def silhouette_stats(clusters, widths):
    """
    Compute per-cluster statistics on silhouette widths.

    Parameters
    ----------
    clusters : ndarray
        A 1-dimensional array of length n, where n is the number of genes in the
        clustered expression data, and each element is the cluster ID of the
        corresponding gene, negative if that gene is not in any cluster.

    widths : ndarray
        An array returned by silhouette_widths which was passed the same
        clusters array.

    Returns
    -------
    ndarray
        A structured array with length equal to the number of clusters, and the
        following fields with statistics for silhouette widths by cluster:
        cluster, count, mean, min, max
        The records are sorted by mean silhouette width.
    """

    k = int(clusters.max() + 1)
    stats = np.empty(k, dtype=[
        ('cluster', 'i4'),
        ('count', 'i4'),
        ('mean', 'f8'),
        ('min', 'f8'),
        ('max', 'f8'),
        ])

    for i in range(k):
        cwidths = widths[clusters == i]
        stats['cluster'][i] = i
        stats['count'][i] = cwidths.size
        stats['mean'][i] = cwidths.mean()
        stats['min'][i] = cwidths.min()
        stats['max'][i] = cwidths.max()

    stats.sort(order=['mean'])

    return stats
