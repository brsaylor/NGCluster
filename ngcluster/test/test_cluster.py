"""
Unit tests for the ngcluster.cluster module
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from ngcluster.cluster import (random_clusters, kmeans, graph_clusters,
        graph_clusters_expanding)

class TestRandomClusters(unittest.TestCase):
    """ Tests for random_clusters """

    def testLength(self):
        # Simply tests that the length of the returned array is correct
        self.assertEqual(len(random_clusters(np.empty(100), 5)), 100)

class TestKMeans(unittest.TestCase):
    """ Tests for kmeans """

    def testKmeans(self):
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [101, 102, 103],
            [104, 105, 106],
            ])
        clusters = kmeans(data, 2)
        self.assertEqual(clusters[0], clusters[1])
        self.assertEqual(clusters[2], clusters[3])

class TestGraphClusters(unittest.TestCase):
    """ Tests for graph_clusters """

    # In each test, a graph function is defined that returns a manually created
    # adjacency matrix. This is to remove the actual graph creation functions
    # from the testing equation.

    def test3(self):

        # 0--1--2

        def graph(data):
            return np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
                ])

        data = np.empty(3)
        
        assert_array_equal(
                graph_clusters(data, graph, threshold=1),
                np.array([0, 0, 0]))
        assert_array_equal(
                graph_clusters(data, graph, threshold=2),
                np.array([-1, -1, -1]))

    def test4(self):

        #    1
        #    |
        # 4--0--2
        #    |
        #    3

        def graph(data):
            return np.array([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                ])

        data = np.empty(4)

        assert_array_equal(
                graph_clusters(data, graph, threshold=1),
                np.array([0, 0, 0, 0]))
        assert_array_equal(
                graph_clusters(data, graph, threshold=2),
                np.array([0, 0, 0, 0]))
        assert_array_equal(
                graph_clusters(data, graph, threshold=3),
                np.array([-1, -1, -1, -1]))

    def test6(self):

        # 0--1--2  3--4--5

        def graph(data):
            return np.array([
                [0, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                ])

        data = np.empty(6)

        assert_array_equal(
                graph_clusters(data, graph, threshold=1),
                np.array([0, 0, 0, 1, 1, 1]))
        assert_array_equal(
                graph_clusters(data, graph, threshold=2),
                np.array([-1, -1, -1, -1, -1, -1]))


    def test9(self):

        #    1
        #    |
        # 4--0--2--5--6--7
        #    |        |
        #    3        8

        def graph(data):
            return np.array([
                #0  1  2  3  4  5  6  7  8
                [0, 1, 1, 1, 1, 0, 0, 0, 0], #0
                [1, 0, 0, 0, 0, 0, 0, 0, 0], #1
                [1, 0, 0, 0, 0, 1, 0, 0, 0], #2
                [1, 0, 0, 0, 0, 0, 0, 0, 0], #3
                [1, 0, 0, 0, 0, 0, 0, 0, 0], #4
                [0, 0, 1, 0, 0, 0, 1, 0, 0], #5
                [0, 0, 0, 0, 0, 1, 0, 1, 1], #6
                [0, 0, 0, 0, 0, 0, 1, 0, 0], #7
                [0, 0, 0, 0, 0, 0, 1, 0, 0], #8
                ])

        data = np.empty(9)

        assert_array_equal(
                graph_clusters(data, graph, threshold=1),
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))
        assert_array_equal(
                graph_clusters(data, graph, threshold=3),
                np.array([0, 0, 0, 0, 0, -1, -1, -1, -1]))
        assert_array_equal(
                graph_clusters(data, graph, threshold=4),
                np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]))

class TestGraphClustersExpanding(unittest.TestCase):
    """ Tests for graph_clusters_expanding """

    # In each test, a graph function is defined that returns a manually created
    # adjacency matrix. This is to remove the actual graph creation functions
    # from the testing equation.

    def test3(self):

        # 0--1--2

        def graph(data):
            return np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
                ])

        data = np.empty(3)
        
        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=1, iterations=1),
                np.array([0, 0, 0]))
        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=2, iterations=1),
                np.array([-1, -1, -1]))

    def test4(self):

        #    1
        #    |
        # 4--0--2
        #    |
        #    3

        def graph(data):
            return np.array([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                ])

        data = np.empty(4)

        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=1, iterations=1),
                np.array([0, 0, 0, 0]))
        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=2, iterations=1),
                np.array([0, 0, 0, 0]))
        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=3, iterations=1),
                np.array([-1, -1, -1, -1]))

    def test6(self):

        # 0--1--2  3--4--5

        def graph(data):
            return np.array([
                [0, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                ])

        data = np.empty(6)

        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=1, iterations=1),
                np.array([0, 0, 0, 1, 1, 1]))
        assert_array_equal(
                graph_clusters_expanding(data, graph, threshold=2, iterations=1),
                np.array([-1, -1, -1, -1, -1, -1]))

    def graph9(self, data):

        #    1
        #    |
        # 4--0--2--5--6--7
        #    |        |
        #    3        8

        return np.array([
            #0  1  2  3  4  5  6  7  8
            [0, 1, 1, 1, 1, 0, 0, 0, 0], #0
            [1, 0, 0, 0, 0, 0, 0, 0, 0], #1
            [1, 0, 0, 0, 0, 1, 0, 0, 0], #2
            [1, 0, 0, 0, 0, 0, 0, 0, 0], #3
            [1, 0, 0, 0, 0, 0, 0, 0, 0], #4
            [0, 0, 1, 0, 0, 0, 1, 0, 0], #5
            [0, 0, 0, 0, 0, 1, 0, 1, 1], #6
            [0, 0, 0, 0, 0, 0, 1, 0, 0], #7
            [0, 0, 0, 0, 0, 0, 1, 0, 0], #8
            ])


    def test9(self):

        data = np.empty(9)

        assert_array_equal(
                graph_clusters_expanding(
                    data, self.graph9, threshold=1, iterations=1),
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))
        assert_array_equal(
                graph_clusters_expanding(
                    data, self.graph9, threshold=3, iterations=1),
                np.array([0, 0, 0, 0, 0, -1, -1, -1, -1]))
        assert_array_equal(
                graph_clusters_expanding(
                    data, self.graph9, threshold=4, iterations=1),
                np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]))

    def test9expanding2(self):

        assert_array_equal(
                graph_clusters_expanding(
                    np.empty(9), self.graph9, threshold=3, iterations=2),
                np.array([0, 0, 0, 0, 0, 0, -1, -1, -1]))

    def test9expanding3(self):

        # first iteration clusters 0,1,2,3,4
        # second iteration adds 5
        # third iteration adds 6

        assert_array_equal(
                graph_clusters_expanding(
                    np.empty(9), self.graph9, threshold=3, iterations=3),
                np.array([0, 0, 0, 0, 0, 0, 0, -1, -1]))

    def test9expanding4(self):

        assert_array_equal(
                graph_clusters_expanding(
                    np.empty(9), self.graph9, threshold=3, iterations=4),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
