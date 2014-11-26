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

class TestRelativeNeighborhoodGraph(unittest.TestCase):
    """ Tests for relative_neighborhood_graph """

    def testRelNeighbor1(self):
        data = np.array([
            [1, 2, 3], # a
            [1, 2, 3], # b
            [3, 2, 1], # c
            [3, 2, 1], # d
            ])

        # (a, b) and (c, d) have correlation 1.
        # Other pairs have correlation -1.
        # For all pairs, no other point is more highly correlated with both than
        # they are to each other. Therefore, all pairs are relative neighbors.

        adj = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            ])
        assert_array_equal(graph.relative_neighborhood_graph(data), adj)

    def testRelNeighbor2(self):

        data = np.array([
            [1, 1, 3], # a
            [1, 2, 3], # b
            [1, 3, 3], # c
            ])

        # p  q  corr  RN?  why
        # a  b  0.87   Y   c is closer to a, but not to b
        # a  c  0.5    N   b is closer to both
        # b  c  0.87   Y   a is closer to c, but not to b

        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            ])
        assert_array_equal(graph.relative_neighborhood_graph(data), adj)

    def testRelNeighbor3(self):

        data = np.array([
            [1, 1, 3], # a
            [1, 2, 3], # b
            [1, 2, 3], # c
            [1, 3, 3], # d
            ])

        # p  q  corr  RN?  why
        # a  b  0.87   Y   c is closer to b, but not to a
        # a  c  0.87   Y   b is closer to c, but not to a
        # a  d  0.5    N   c is closer to both
        # b  c  1      Y   no point closer to either
        # b  d  0.87   Y   c is closer to b, but not to d
        # c  d  0.87   Y   b is closer to c, but not to d

        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            ])
        assert_array_equal(graph.relative_neighborhood_graph(data), adj)
