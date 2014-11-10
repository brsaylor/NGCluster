"""
Unit tests for the ngcluster.evaluate module
"""

import unittest

import numpy as np

from ngcluster.evaluate import fom, aggregate_fom

def dummy_clusters(data, k):
    """ Perform a dummy clustering of the data by alternately assigning rows
    into k clusters. """
    return np.array([i % k for i in range(data.shape[0])])

class TestFOM(unittest.TestCase):
    """ Tests for fom """

    def testNonAdjustedOnePerCluster(self):
        clusters = np.array([0, 1, 2, 3])
        hidden_data = np.array([0, 1, 2, 3])
        self.assertEqual(fom(clusters, hidden_data, adjust=False), 0.)

    def testNonAdjustedTwoPerCluster(self):
        clusters = np.array([0, 0, 1, 1])
        hidden_data = np.array([0, 4, 8, 12])
        self.assertEqual(fom(clusters, hidden_data, adjust=False), 2.)

    def testTwoPerCluster(self):
        clusters = np.array([0, 0, 1, 1])
        hidden_data = np.array([0, 4, 8, 12])
        self.assertAlmostEqual(fom(clusters, hidden_data), 2.828427125)

    def testUnclustered(self):
        clusters = np.array([-1, -1, 0, 0, 1, 1])
        hidden_data = np.array([999, 999, 0, 4, 8, 12])
        self.assertAlmostEqual(fom(clusters, hidden_data), 2.828427125)

class TestAggregateFOM(unittest.TestCase):
    """ Tests for aggregate_fom """

    def testZero(self):
        # If all rows are zero, the aggregate FOM should be zero.
        numrows = 10
        k = 5
        data = np.array([[0,0,0] for i in range(numrows)])
        self.assertEqual(aggregate_fom(data, dummy_clusters, [k]), 0.)

    def testZero2(self):
        # If all rows have equal values, the aggregate FOM should be zero.
        numrows = 10
        k = 5
        data = np.array([[0,1,2] for i in range(numrows)])
        self.assertEqual(aggregate_fom(data, dummy_clusters, [k]), 0.)

    def testTwoPerCluster(self):
        # If all columns have equal values, the aggregate FOM should be numcols
        # times the FOM with one column as the hidden data.
        k = 2
        data = np.array([
            [0, 0, 0, 0], 
            [8, 8, 8, 8], 
            [4, 4, 4, 4], 
            [12, 12, 12, 12], 
            ])
        self.assertAlmostEqual(aggregate_fom(data, dummy_clusters, [k]),
                4*2.828427125)

    def testNonAdjustedTwoPerCluster(self):
        # Same as testTwoPerCluster, but with adjust=False
        k = 2
        data = np.array([
            [0, 0, 0, 0], 
            [8, 8, 8, 8], 
            [4, 4, 4, 4], 
            [12, 12, 12, 12], 
            ])
        self.assertAlmostEqual(
                aggregate_fom(data, dummy_clusters, [k], adjust=False), 8)

    def testFibonacci(self):
        # Yields a FOM that's different for each cluster and each condition
        k = 2
        data = np.array([
            [1, 1, 2],
            [3, 5, 8],
            [13, 21, 34],
            [55, 89, 144],
            ])
        self.assertAlmostEqual(aggregate_fom(data, dummy_clusters, [k]),
                139.7143912044)
