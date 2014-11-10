"""
Functions for evaluating clusterings
"""

from math import sqrt

import numpy as np

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
