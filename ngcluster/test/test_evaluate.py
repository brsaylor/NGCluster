"""
Unit tests for the ngcluster.evaluate module
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ngcluster.evaluate import (fom, aggregate_fom, rand_index,
        silhouette_widths, silhouette_stats)

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

class TestRandIndex(unittest.TestCase):
    """ Tests for rand_index """

    def testZeroAgreement(self):
        X = np.array([0, 1, 2]) # All in different clusters
        Y = np.array([0, 0, 0]) # All in same cluster
        self.assertEqual(rand_index(X, Y), 0.0)

    def testPerfectAgreementDifferent(self):
        X = np.array([0, 1, 2]) # All in different clusters
        Y = np.array([0, 1, 2]) # All in different clusters
        self.assertEqual(rand_index(X, Y), 1.0)

    def testPerfectAgreementSame(self):
        X = np.array([0, 0, 0]) # All in same cluster
        Y = np.array([0, 0, 0]) # All in same cluster
        self.assertEqual(rand_index(X, Y), 1.0)

    def testHalfAgreement(self):
        X = np.array([0, 0, 0, 0]) # All in same cluster
        Y = np.array([0, 0, 0, 1]) # Half of the pairs are split
        self.assertEqual(rand_index(X, Y), 0.5)

    def testRand1971Example(self):
        # This is from William Rand's 1971 paper that defined the index
        X = np.array([0, 0, 0, 1, 1, 1])
        Y = np.array([0, 0, 1, 1, 1, 2])
        self.assertEqual(rand_index(X, Y), 0.6)

    def testZeroAgreementWithUnclustered(self):
        X = np.array([0, 1, 2, -1]) # All in different clusters
        Y = np.array([0, 0, 0, -1]) # All in same cluster
        self.assertEqual(rand_index(X, Y), 0.0)

    def testHalfAgreementWithUnclustered(self):
        X = np.array([0, 0, 0, 0, -1]) # All in same cluster
        Y = np.array([0, 0, 0, 1, -1]) # Half of the pairs are split
        self.assertEqual(rand_index(X, Y), 0.5)

class TestSilhouetteWidths(unittest.TestCase):
    """ Tests for silhouette_widths """

    # Check the output of silhouette_widths against figure 2 or 3 in Rousseeuw's
    # 1987 paper that originally described the silhouette technique.
    # cluster_data is a tuple of dictionaries, one for each cluster, in which
    # the keys are the country abbreviations and the values are the silhouette
    # widths for the corresponding countries, as given in the figure.
    def _testRousseeuwFig(self, cluster_data):

        # Figures 2 and 3 are based on the following distance matrix of 12
        # countries
        countries = {k: v for v, k in enumerate([
            'BEL', 'BRA', 'CHI', 'CUB', 'EGY', 'FRA', 'IND', 'ISR', 'USA',
            'USS', 'YUG', 'ZAI'])}
        countryDistances = [
            [],
            [5.58],
            [7.00, 6.50],
            [7.08, 7.00, 3.83],
            [4.83, 5.08, 8.17, 5.83],
            [2.17, 5.75, 6.67, 6.92, 4.92],
            [6.42, 5.00, 5.58, 6.00, 4.67, 6.42],
            [3.42, 5.50, 6.42, 6.42, 5.00, 3.92, 6.17],
            [2.50, 4.92, 6.25, 7.33, 4.50, 2.25, 6.33, 2.75],
            [6.08, 6.67, 4.25, 2.67, 6.00, 6.17, 6.17, 6.92, 6.17],
            [5.25, 6.83, 4.50, 3.75, 5.75, 5.42, 6.08, 5.83, 6.67, 3.67],
            [4.75, 3.00, 6.08, 6.67, 5.00, 5.58, 4.83, 6.17, 5.67, 6.50, 6.92]]

        # Create a square, symmetric distance matrix from the above data
        countryMatrix = np.empty((12, 12))
        for i, row in enumerate(countryDistances):
            for j, val in enumerate(row):
                countryMatrix[i,j] = val
                countryMatrix[j,i] = val

        # Set cluster assignments and correct silhouette widths from the figure
        # (cluster_data)
        clusters = np.empty(12)
        correct_widths = np.empty(12)
        for i, cluster in enumerate(cluster_data):
            for abbr, width in cluster.items():
                clusters[countries[abbr]] = i
                correct_widths[countries[abbr]] = width

        # Compute the widths using our function. After rounding to the precision
        # displayed in the figure, the results should match.
        rounded_widths = np.round(
                silhouette_widths(clusters, None, dmatrix=countryMatrix),
                2)
        assert_array_equal(rounded_widths, correct_widths)

    def testRousseeuwFig2(self):
        self._testRousseeuwFig(({
                'USA': 0.43,
                'BEL': 0.39,
                'FRA': 0.35,
                'ISR': 0.30,
                'BRA': 0.22,
                'EGY': 0.20,
                'ZAI': 0.19,
                }, {
                'CUB': 0.40,
                'USS': 0.34,
                'CHI': 0.33,
                'YUG': 0.26,
                'IND': -0.04,
                }))

    def testRousseeuwFig3(self):
        self._testRousseeuwFig(({
                'USA': 0.47,
                'FRA': 0.44,
                'BEL': 0.42,
                'ISR': 0.37,
                'EGY': 0.02,
                }, {
                'ZAI': 0.28,
                'BRA': 0.25,
                'IND': 0.17,
                }, {
                'CUB': 0.48,
                'USS': 0.44,
                'YUG': 0.31,
                'CHI': 0.31,
                }))

    def testOneObject(self):
        data = np.array([[1, 2]])
        clusters = np.array([0])
        correct = np.array([0])
        assert_array_equal(silhouette_widths(clusters, data), correct)

    def testTwoSingletons(self):
        data = np.array([[1, 2], [3,4]])
        clusters = np.array([0, 1])
        correct = np.array([0, 0])
        assert_array_equal(silhouette_widths(clusters, data), correct)

    def testUnclustered(self):
        data = np.array([[1, 2], [3,4]])
        clusters = np.array([0, -1])
        correct = np.array([0, 0])
        assert_array_equal(silhouette_widths(clusters, data), correct)

    def testEuclidean(self):
        data = np.array([[0, 0], [0, 0], [3, 4], [5, 12]])
        clusters = np.array([0, 0, 1, 1])
        # cluster 0: a[0] = a[1] = 0
        # cluster 1: a[2] = a[3] = 8.246211251235321
        # distance from (0, 0) to (3, 4) is 5
        # distance from (0, 0) to (5, 12) is 13
        # so:
        # cluster 0: b[0] = b[1] = 9
        # cluster 1: b[2] = 5, b[3] = 13
        # width[0] = (b[0] - a[0]) / max(a[0], b[0]) = 9/9 = 1
        #  = width[1]
        # width[2] = (b[2] - a[2]) / max(a[2], b[2])
        #          = (5 - 8.246211251235321) / 8.246211251235321
        #          = -0.3936609374091676
        # width[3] = (b[3] - a[3]) / max(a[3], b[3])
        #          = (13 - 8.246211251235321) / 13
        #          = 0.365676057597283
        correct = np.array([1, 1, -0.3936609374091676, 0.365676057597283])
        assert_array_almost_equal(silhouette_widths(clusters, data), correct)

class TestSilhouetteStats(unittest.TestCase):
    """ Tests for silhouette_stats """

    def test1(self):
        clusters = np.array([0, 0, 1, 1, 2, 2, -1])
        widths =   np.array([6, 5, 4, 3, 2, 1, 100])
        stats, summary = silhouette_stats(clusters, widths)
        correct_stats = np.array([
            (2, 2, 1.5, 1, 2),
            (1, 2, 3.5, 3, 4),
            (0, 2, 5.5, 5, 6),
            ], dtype=stats.dtype)
        assert_array_equal(stats, correct_stats)
