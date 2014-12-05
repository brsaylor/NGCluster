"""
Functions for plotting clustering results
"""

import matplotlib.pyplot as plt

def plot_cluster_expression(names, data, clusters):
    """
    Generate plots of the expression data of the genes in each cluster.

    Parameters
    ----------
    names : ndarray
        The names of the genes.

    data : ndarray
        An n*m array of of expression data for n genes under m conditions.

    clusters : ndarray
        A 1-dimensional array of length n, where n is the number of genes in the
        clustered expression data, and each element is the cluster ID of the
        corresponding gene, negative if that gene is not in any cluster.

    Returns
    -------
    list of matplotlib.figure.Figure
        A list of Figures, each of which contains one or more subplots. Each
        subplot is a plot of the expression data for the genes in one cluster.
    """

    plots_per_fig = 2

    num_clusters = clusters.max() + 1
    figs = []
    for cluster_id in range(num_clusters):
        if cluster_id % plots_per_fig == 0:
            figs.append(plt.figure())
        cluster_data = data.compress(clusters == cluster_id, axis=0)
        plt.subplot(plots_per_fig, 1, cluster_id % plots_per_fig + 1)
        plt.title("Cluster {0}".format(cluster_id))
        for row in cluster_data:
            plt.plot(range(1, len(row) + 1), row)

    return figs
