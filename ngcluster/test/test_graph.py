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

class TestNearestNeighborGraph(unittest.TestCase):
    """ Tests for nearest_neighbor_graph """

    def testNearestNeighbor4(self):
        data = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [3, 2, 1],
            [3, 2, 1],
            ])
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            ])
        assert_array_equal(graph.nearest_neighbor_graph(data), adj)

    def testNearestNeighbor4WithZeros(self):
        data = np.array([
            [1, 2, 3],
            [0, 0, 0],
            [3, 2, 1],
            [3, 2, 1],
            ])
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            ])
        assert_array_equal(graph.nearest_neighbor_graph(data), adj)
