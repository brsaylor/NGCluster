"""
Named sets of clustering parameters
"""

from collections import OrderedDict

from ngcluster.cluster import random_clusters, kmeans, graph_clusters
from ngcluster.graph import (threshold_graph, nearest_neighbor_graph,
        relative_neighborhood_graph, gabriel_graph)

configurations = OrderedDict([
    ('random10', {
        'description': "10 random clusters",
        'cluster': (random_clusters, {'k': 10})
        }),
    ('kmeans10', {
        'description': "k-means with 10 clusters",
        'cluster': (kmeans, {'k': 10})
        }),
    ('threshold_graph_default', {
        'description': "Default threshold graph clustering",
        'cluster': (graph_clusters, {'threshold': 5}),
        'graph': (threshold_graph, {'threshold': 0.85})
        }),
    ('nearest_neighbor_default', {
        'description': "Default nearest neighbor graph clustering",
        'cluster': (graph_clusters, {'threshold': 5}),
        'graph': (nearest_neighbor_graph, {})
        }),
    ('relative_neighbor_default', {
        'description': "Default relative neighbor graph clustering",
        'cluster': (graph_clusters, {'threshold': 5}),
        'graph': (relative_neighborhood_graph, {})
        }),
    ('gabriel_default', {
        'description': "Default Gabriel graph clustering",
        'cluster': (graph_clusters, {'threshold': 5}),
        'graph': (gabriel_graph, {})
        }),
    ])
