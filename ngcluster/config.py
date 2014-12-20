"""
Named sets of clustering parameters
"""

from collections import OrderedDict

from ngcluster.cluster import (random_clusters, kmeans, graph_clusters,
        graph_clusters_expanding)
from ngcluster.graph import (threshold_graph, nearest_neighbor_graph,
        relative_neighborhood_graph, gabriel_graph)

configurations = OrderedDict([
    
    # Control configurations
    ('random10', {
        'description': "10 random clusters",
        'cluster': (random_clusters, {'k': 10})
        }),
    ('kmeans10', {
        'description': "k-means with 10 clusters",
        'cluster': (kmeans, {'k': 10})
        }),

    # The class assignment from Lonardi
    ('threshold_graph_default', {
        'description': "Default threshold graph clustering",
        'cluster': (graph_clusters, {'threshold': 5}),
        'graph': (threshold_graph, {'metric': 'correlation', 'threshold': 0.85})
        }),

    # Vary the following parameters:
    # graph: nearest, relative, gabriel
    # metric: correlation, cosine, euclidean
    # threshold: 5, 10
    # iterations: 1, 2
    # max_clusters: none, 10
    # Use graph_clusters_expanding only.

    ('nearest_cor_thresh5_iter1_nomax', {
        'description': "Nearest neighbor graph",
        'graph': (nearest_neighbor_graph, {'metric': 'correlation'}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),
    ('relative_cor_thresh5_iter1_nomax', {
        'description': "Relative neighborhood graph",
        'graph': (relative_neighborhood_graph, {'metric': 'correlation'}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),

    ('nearest_cos_thresh5_iter1_nomax', {
        'description': "Nearest neighborhood graph",
        'graph': (nearest_neighbor_graph, {'metric': 'cosine'}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),
    ('relative_cos_thresh5_iter1_nomax', {
        'description': "Relative neighborhood graph",
        'graph': (relative_neighborhood_graph, {'metric': 'cosine'}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),

    ('nearest_euc_thresh5_iter1_nomax', {
        'description': "Nearest neighbor graph",
        'graph': (nearest_neighbor_graph, {'metric': 'euclidean'}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),
    ('relative_euc_thresh5_iter1_nomax', {
        'description': "Relative neighborhood graph",
        'graph': (relative_neighborhood_graph, {'metric': 'euclidean'}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),
    ('gabriel_euc_thresh5_iter1_nomax', {
        'description': "Gabriel graph",
        'graph': (gabriel_graph, {}),
        'cluster': (graph_clusters_expanding, {'threshold': 5, 'iterations': 1}),
        }),

    ])

# Files with external cluster assignments to be used in calculating the Rand
# index. The main program will look for these files in the data directory.
external_cluster_files = [
        'threshold_graph_default-clusters.txt',
        'gene_association.txt',
        'Ranked_TFs.txt',
        ]
