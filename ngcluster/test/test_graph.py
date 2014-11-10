"""
Unit tests for the ngcluster.graph module
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ngcluster import graph

class TestThresholdGraph(unittest.TestCase):
    """ Tests for threshold_graph """

    def testThresholdGraph(self):
        threshold = 0.5
        data = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [12, 11, 10],
            [9, 8, 7]
            ])

        # The correct adjacency matrix
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
            ])
        assert_array_equal(graph.threshold_graph(data, threshold), adj)
