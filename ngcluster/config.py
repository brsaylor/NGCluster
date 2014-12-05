"""
Named sets of clustering parameters
"""

from collections import OrderedDict

from ngcluster.cluster import random_clusters, graph_clusters
from ngcluster.graph import threshold_graph

configurations = OrderedDict([
    ('random5', {
        'description': "5 random clusters",
        'cluster': (random_clusters, {'k': 5})
        }),
    ('threshold_graph_default', {
        'description': "Default threshold graph clustering",
        'cluster': (graph_clusters, {'threshold': 5}),
        'graph': (threshold_graph, {'threshold': 0.85})
        })
    ])
