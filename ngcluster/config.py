"""
Named sets of clustering parameters
"""

from collections import OrderedDict

from ngcluster.cluster import random_clusters, graph_clusters
from ngcluster.graph import threshold_graph

configurations = OrderedDict([
    ('random5', {
        'description': "5 random clusters",
        'fn': random_clusters,
        'fn_args': [5],
        'fn_kwargs': {},
        }),
    ('threshold_graph_default', {
        'description': "Default threshold graph clustering",
        'fn': graph_clusters,
        'fn_args': [threshold_graph],
        'fn_kwargs': {
            'fn_args': [0.85],
            'threshold': 5,
            },
        })
    ])
