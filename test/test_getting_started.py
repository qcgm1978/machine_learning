import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest, math,matplotlib
from data.handle_data import HandleData
from graphic.plot import Plot
from AI import DoAI
from predict import Predict
import numpy
from mysql_data.decorators_func import singleton
from utilities import getPath,parseNumber


class TDD_GETTING_STARTED(unittest.TestCase):
    @singleton
    def setUp(self):
        class PlotAI(HandleData, Plot, DoAI, Predict):
            def __init__(self,arg=None):
                HandleData.__init__(self,arg)
                Plot.__init__(self)

        self.__class__.PlotAI = PlotAI
        self.__class__.p = PlotAI()

    def test_getting_started(self):
        l = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
        self.assertIsInstance(l, list)

    def test_datatypes(self):
        d = self.PlotAI(5)
        self.assertTrue(d.Numerical())
        self.assertTrue(d.Discrete())
        self.assertFalse(d.Continuous())
        d = self.PlotAI(5.0)
        self.assertTrue(d.Numerical())
        self.assertFalse(d.Discrete())
        self.assertTrue(d.Continuous())
        d = self.PlotAI({"speed": [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]})
        d1 = self.PlotAI({"speed": [99, 86, 87, 88, 86, 103, 87, 94, 78, 77, 85, 86]})
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
        d = self.PlotAI({"speed": [86, 87, 88, 86, 87, 85, 86]})
        d1 = self.PlotAI({"speed": [32, 111, 138, 28, 59, 77, 97]})
        s = d.getPSD()
        s1 = d1.getPSD()
        self.assertAlmostEqual(s, 0.9, 2)
        m = d.getMean()
        m1 = d1.getMean()
        self.assertAlmostEqual(m, 86.4, 1)
        self.assertAlmostEqual(m1, 77.4, 1)
        self.assertAlmostEqual(s1, 37.85, 2)
        v = d1.getVariance()
        self.assertAlmostEqual(v, 1432.2, 1)
        # the formula to find the standard deviation is the square root of the variance:
        self.assertEqual(s1, math.sqrt(v))
        self.assertEqual(s1 ** 2, (v))

    def test_uniform(self):
        d = self.PlotAI({"speed": [86, 87.7, 88, 86, 87, 85, 86]})
        d1 = self.PlotAI({"speed": [91.6, 87.7, 88, 86, 87, 85, 86]})
        s = d.getPSD()
        s1 = d1.getPSD()
        self.assertAlmostEqual(s, 1.00, 2)
        self.assertAlmostEqual(s1, 2.00, 2)
        self.assertEqual(d.getProbability(), 0.6827)
        self.assertEqual(d1.getProbability(), 0.9545)

    def test_NCEE(self):
        d = self.PlotAI({"points": [580, 600, 680, 620], "expectation": 690})
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
            5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31
        ]
        d = self.PlotAI({"ages": ages})
        p = d.getPercentile(0.75)
        p1 = d.getPercentile(0.9)
        self.assertEqual(p, 43)
        self.assertAlmostEqual(p1, 61.0)
    def test_grouped_percentile(self):
#         When the data is grouped:
#
# Add up all percentages below the score,
# plus half the percentage at the score.
        l=({'D':.12},{'C':.5},{'B':.3},{'A':.08})
        p=self.p.getPercentile('B',l=l)
        self.assertEqual(p,.77)
    def test_decile(self):
        ages = [
            5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31
        ]
        d=self.p.getDecile(ages,5)
        d1=self.p.getDecile(ages,31)
        self.assertEqual(d,0)
        self.assertAlmostEqual(d1,.048,3)

    def test_Quartile(self):
        l=1, 3, 3, 4, 5, 6, 6, 7, 8, 8
        q=self.p.getQuartile(l,2)
        q1=self.p.getQuartile(l,1)
        q3=self.p.getQuartile(l,3)
        q4=self.p.getQuartile(l,percentile=75)
        self.assertEqual(q,5.5)
        self.assertEqual(q1,3)
        self.assertEqual(q3,7)
        self.assertEqual(q4,7)
    def test_Estimating_Percentiles(self):
        filePath = "data/shopping.csv"
        path = getPath(filePath)
        df=self.p.readCsv(path)
        p=df['People']
        t=df['Time (hours)']
        self.assertEqual(len(p),7)
        l = list(map(parseNumber,p.tolist()))
        poly=self.p.polyfit(t.tolist(),l)
        # print(poly)
        self.assertEqual(sum(l)*.3,8760)
        self.assertEqual(sum(l[:4]),3850)
        self.assertEqual(sum(l[:5]),10350)
        self.assertAlmostEqual((8760-3850)/l[4],.755,3)
        per=self.p.getPercentile(l=l,percent='30%')
        self.assertAlmostEqual(per,950)
    def test_data_distribution(self):
        valCount=250
        x = numpy.random.uniform(0.0, 5.0, valCount)
        isfloat = all(isinstance(v, float) for v in x)
        self.assertTrue(isfloat)
        fig,ax=self.p.getFigAx()
        bars=5
        n, bins, patches=self.p.plotHist(x,bars,ax=ax,insertBar=False,barCol='#2877b4')
        self.p.saveAndShow()
        print(n,bins,patches)
        self.assertEqual(len(n),bars)
        self.assertEqual(sum(n),valCount)
        self.assertEqual(len(n)+1,len(bins))
        self.assertIsInstance(patches,matplotlib.container.BarContainer)
    def test_histogram(self):
        d = self.PlotAI()
        x = numpy.random.normal(5.0, 100.0, 100000)
        x = d.getND(.4, 4, size=10000)
        # d = self.PlotAI({"x": x})
        d.plotND(100, l=x)

    def test_scatter(self):
        x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
        y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
        d = self.PlotAI({"x": x, "y": y})
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
        d = self.PlotAI({"x": x, "y": y})
        # d.scatterLine()
        r = d.getR()
        self.assertAlmostEqual(r, 0.01, 2)

    def test_random_data(self):
        x = numpy.random.normal(5.0, 1.0, 1000)
        y = numpy.random.normal(10.0, 5.0, 1000)
        d = self.PlotAI({'x': x, 'y': y})
        d.scatter()


if __name__ == "__main__":
    unittest.main()
