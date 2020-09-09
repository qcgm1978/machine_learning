import sys, os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datatype import DataTypes
import unittest,pandas
from mysql_data.decorators_func import singleton
from unum.units import * # Load a number of common units.
import pint,math
from pint import UnitRegistry
ureg = UnitRegistry()
class TDD_TEST_SD(unittest.TestCase):
    @singleton
    def setUp(self):
        file = "data/metabolic.csv"
        X = ["Sex", "Metabolic rate"]
        y = "CO2"
        predictVals = [2300, 1300]
        self.__class__.d = DataTypes({"file": file, "x": X, "y": y})
        self.__class__.l1 = self.__class__.d.queryDf('Sex == "Male"')
        self.__class__.l2 = self.__class__.d.queryDf('Sex == "Female"')
        self.__class__.s1 = self.__class__.d.getSD(self.__class__.l1)
        self.__class__.s2 = self.__class__.d.getSD(self.__class__.l2)
        self.__class__.s3 = self.__class__.d.getSD(self.__class__.l1,expected=None)
    def test_test_SD(self):
        self.assertIsInstance(self.d.df, pandas.core.frame.DataFrame)
        mr = self.d.getDfCol()
        self.assertIsInstance(mr, pandas.core.series.Series)
        self.assertAlmostEqual(self.d.getPSD(mr), 694.4, 1)
        self.assertAlmostEqual(self.s1, 894.37, 2)
        self.assertAlmostEqual(self.s2, 420.96, 2)
        self.assertIsNone(self.s3)
        p=[[1,.2],[2,.4]]
        sd = self.d.getSD(p, isEqlProb=False)
        mean=1*.2+2*.4
        self.assertEqual(sd,math.sqrt(.2*(1-mean)**2+.4*(2-mean)**2))
    def test_plot(self):
        # self.d.plotGroupedBar(
        #     l1=self.l1,
        #     l2=self.l2,
        #     l1txt="Male",
        #     l2txt="Female",
        #     title='Furness data set on metabolic rates of northern fulmars',
        #     prop='Metabolic rate'
        # )
        y1 = self.l1
        y2 = self.l2
        mMean = self.d.getMean(self.l1)
        fMean = self.d.getMean(self.l2)
        t1 = ['Female', 'Std. Dev.', int(round(self.s1))]
        t2 = ['Male', "Std. Dev", int(round(self.s2))]
        # self.d.scatterGrouped(
        #     [
        #         ("Female", y2.values.tolist()+[fMean],t1),
        #         ("Male", y1.values.tolist() + [mMean],t2),
        #         ("Female Mean", [fMean]),
        #         ("Male Mean", [mMean]),
        #     ],
        #     title=['Sample standard deviation of','metabolic rate in male and female fulmars'],
        #     yTxt="Matabolic rate",
        #     xTxt="Sex",
        # )
    def test_PSD(self):
        l = [2, 4, 4, 4, 5, 5, 7, 9]
        p = self.d.getPSD(l)
        self.assertEqual(p, 2)
    def test_average_height(self):
        m=self.d.to('inch','m')
        self.assertEqual(m,0.0254*ureg.m)
        self.assertEqual(m.magnitude,0.0254)
        self.assertEqual(m.units,'meter')
        self.assertIsInstance(m.dimensionality,pint.util.UnitsContainer)
        self.assertEqual(m.dimensionality,pint.util.UnitsContainer({'[length]':1}))
        m=self.d.to('inch','cm',num=70)
        m1=self.d.to('inch','cm',num=3)
        self.assertEqual(m.magnitude,177.8)
        self.assertEqual(m1.magnitude, 7.62)
        p0 = self.d.getProbability(0)
        p1 = self.d.getProbability(1)
        p2 = self.d.getProbability(2)
        p3 = self.d.getProbability(3)
        self.assertEqual(p0,1)
        self.assertEqual(p1,.6827)
        self.assertEqual(p2,.9545)
        self.assertEqual(p3, .9973)

if __name__ == "__main__":
    unittest.main()
