"""
Unit tests for the ngcluster.graph module
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ngcluster import graph

class TestThresholdGraph(unittest.TestCase):
    """ Tests for threshold_graph """

    def testCorrelation(self):
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

    def testCorrelationZeros(self):
        # Test for when data has all-zero rows
        threshold = 0.5
        data = np.array([
            [0, 1],
            [0, 2],
            [0, 0],
            [0, 0],
            ])
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            ])
        assert_array_equal(graph.threshold_graph(data, threshold), adj)

    def testEuclidean(self):
        threshold = 1
        data = np.array([
            [0, 0],
            [0.9, 0],
            [0, 0.9],
            [0, 1]
            ])
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            ])
        assert_array_equal(graph.threshold_graph(
            data, threshold, metric='euclidean'), adj)

    def testEuclideanZeros(self):
        # Test for when data is all zero
        threshold = 1
        data = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            ])
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            ])
        assert_array_equal(graph.threshold_graph(
            data, threshold, metric='euclidean'), adj)

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
            [1, 2, 3],
            [3, 2, 1],
            [3, 2, 1],
            [0, 0, 0],
            ])
        adj = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
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

class TestGabrielGraph(unittest.TestCase):

    def testGabrielPair(self):

        # There are only two points, so they are Gabriel neighbors

        data = np.array([
            [0, 0],
            [1, 1],
            ])
        adj = np.array([
            [0, 1],
            [1, 0],
            ])
        assert_array_equal(graph.gabriel_graph(data), adj)

    def testGabrielSquare(self):

        # Each pair defining a side of a square are Gabriel neighbors

        data = np.array([
            [0, 0], # a
            [1, 0], # b
            [1, 1], # c
            [0, 1], # d
            ])
        adj = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            ])
        assert_array_equal(graph.gabriel_graph(data), adj)

    def testGabrielSquare2(self):

        # Point z is in the middle of the square, so the corners of the square
        # are Gabriel neighbors with z instead of each other

        data = np.array([
            [0, 0], # a
            [1, 0], # b
            [1, 1], # c
            [0, 1], # d
            [0.5, 0.5], # z
            ])
        adj = np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            ])
        assert_array_equal(graph.gabriel_graph(data), adj)

    def testGabrielSquare3(self):

        # From testGabrielSquare2, move z a little to the right, out of the
        # discs between the pairs defining the top (c,d), left (a,d), and bottom
        # (a, b) sides, making those pairs neighbors as well.

        data = np.array([
            [0, 0], # a
            [1, 0], # b
            [1, 1], # c
            [0, 1], # d
            [0.51, 0.5], # z
            ])
        adj = np.array([
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0],
            ])
        assert_array_equal(graph.gabriel_graph(data), adj)

    def testGabrielCube(self):

        # Each pair defining an edge of the cube are Gabriel neighbors,
        # so each point has 3 neighbors.

        data = np.array([
            [0, 0, 0], # a
            [1, 0, 0], # b
            [1, 1, 0], # c
            [0, 1, 0], # d
            [0, 0, 1], # e
            [1, 0, 1], # f
            [1, 1, 1], # g
            [0, 1, 1], # h
            ])
        adj = np.array([
            #a  b  c  d  e  f  g  h
            [0, 1, 0, 1, 1, 0, 0, 0], # a
            [1, 0, 1, 0, 0, 1, 0, 0], # b
            [0, 1, 0, 1, 0, 0, 1, 0], # c
            [1, 0, 1, 0, 0, 0, 0, 1], # d
            [1, 0, 0, 0, 0, 1, 0, 1], # e
            [0, 1, 0, 0, 1, 0, 1, 0], # f
            [0, 0, 1, 0, 0, 1, 0, 1], # g
            [0, 0, 0, 1, 1, 0, 1, 0], # h
            ])
        assert_array_equal(graph.gabriel_graph(data), adj)
