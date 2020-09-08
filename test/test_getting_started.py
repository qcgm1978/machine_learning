import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest, math
from datatype import DataTypes
import numpy
class TDD_GETTING_STARTED(unittest.TestCase):
    def test_getting_started(self):
        l = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
        self.assertIsInstance(l, list)
    def test_datatypes(self):
        d = DataTypes(5)
        self.assertTrue(d.Numerical())
        self.assertTrue(d.Discrete())
        self.assertFalse(d.Continuous())
        d = DataTypes(5.0)
        self.assertTrue(d.Numerical())
        self.assertFalse(d.Discrete())
        self.assertTrue(d.Continuous())
        d = DataTypes({"speed": [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]})
        d1 = DataTypes({"speed": [99, 86, 87, 88, 86, 103, 87, 94, 78, 77, 85, 86]})
        m = d.getMean()
        self.assertAlmostEqual(m, 89.77, 1)
        median = d.getMedian()
        median1 = d1.getMedian()
        self.assertEqual(median, 87)
        self.assertEqual(median1, 86.5)
        mode = d.getMode()
        # print(mode)
        self.assertEqual(mode[0], 86)
        self.assertEqual(mode.mode, 86)
        self.assertEqual(mode[1], 3)
        self.assertEqual(mode.count, 3)
    def test_standard_deviation(self):
        d = DataTypes({"speed": [86, 87, 88, 86, 87, 85, 86]})
        d1 = DataTypes({"speed": [32, 111, 138, 28, 59, 77, 97]})
        s = d.getPSD()
        s1 = d1.getPSD()
        self.assertAlmostEqual(s, 0.9, 2)
        self.assertAlmostEqual(s1, 37.85, 2)
        v = d1.getVariance()
        self.assertAlmostEqual(v, 1432.2, 1)
        # the formula to find the standard deviation is the square root of the variance:
        self.assertEqual(s1, math.sqrt(v))
        self.assertEqual(s1 ** 2, (v))
    def test_uniform(self):
        d = DataTypes({"speed": [86, 87.7, 88, 86, 87, 85, 86]})
        d1 = DataTypes({"speed": [91.6, 87.7, 88, 86, 87, 85, 86]})
        s = d.getPSD()
        s1 = d1.getPSD()
        self.assertAlmostEqual(s, 1.00, 2)
        self.assertAlmostEqual(s1, 2.00, 2)
        self.assertEqual(d.getProbability(), 0.683)
        self.assertEqual(d1.getProbability(), 0.954)
    def test_NCEE(self):
        d = DataTypes({"points": [580, 600, 680, 620], "expectation": 690})
        m = d.getMean()
        self.assertEqual(m, 620)
        p = d.get1stdProbability()
        self.assertAlmostEqual(p, 37.4, 1)
        distance = d.getDistance1std()
        self.assertAlmostEqual(distance, 1.87, 2)
        probability = d.getProbability()
        self.assertEqual(probability, 0.015)
    def test_percentile(self):
        ages = [
            5,
            31,
            43,
            48,
            50,
            41,
            7,
            11,
            15,
            39,
            80,
            82,
            32,
            2,
            8,
            6,
            25,
            36,
            27,
            61,
            31,
        ]
        d = DataTypes({"ages": ages})
        p = d.getPercentile(0.75)
        p1 = d.getPercentile(0.9)
        self.assertEqual(p, 43)
        self.assertEqual(p1, 61.0)
    def test_data_distribution(self):
        x = numpy.random.uniform(0.0, 5.0, 250)
        isfloat = all(isinstance(v, float) for v in x)
        self.assertTrue(isfloat)
    def test_histogram(self):
        x = numpy.random.normal(5.0, 100.0, 100000)
        d = DataTypes({"x": x})
        # d.pyplot(100)
    def test_scatter(self):
        x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
        y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
        d = DataTypes({"x": x, "y": y})
        # d.scatterLine()
        r = d.getR()
        self.assertAlmostEqual(r, -0.76, 2)
        p = d.predict(10)
        self.assertEqual(p, 85.59308314937454)
    def test_bad_fit(self):
        x = [
            89,
            43,
            36,
            36,
            95,
            10,
            66,
            34,
            38,
            20,
            26,
            29,
            48,
            64,
            6,
            5,
            36,
            66,
            72,
            40,
        ]
        y = [
            21,
            46,
            3,
            35,
            67,
            95,
            53,
            72,
            58,
            10,
            26,
            34,
            90,
            33,
            38,
            20,
            56,
            2,
            47,
            15,
        ]
        d = DataTypes({"x": x, "y": y})
        # d.scatterLine()
        r = d.getR()
        self.assertAlmostEqual(r, 0.01, 2)
    def test_random_data(self):
        x = numpy.random.normal(5.0, 1.0, 1000)
        y = numpy.random.normal(10.0, 5.0, 1000)
        d = DataTypes({'x': x, 'y': y})
        d.scatter()
if __name__ == "__main__":
    unittest.main()
