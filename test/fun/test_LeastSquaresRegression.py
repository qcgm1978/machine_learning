import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest
from graphic.plot import Plot
from predict import Predict
from data.handle_data import HandleData
from mysql_data.decorators_func import singleton
from utilities import getPath,parseNumber
import numpy as np
class TDD_TEST_LEASTSQUARESREGRESSION(unittest.TestCase):
    @singleton
    def setUp(self):
        class PlotAI(Plot,Predict,HandleData):
            def __init__(self,arg=None):
                Plot.__init__(self)
                HandleData.__init__(self)
        self.__class__.p = PlotAI()
    def test_test_LeastSquaresRegression(self):
        x, y = self.p.getSortedXyInt((11,25,12),(130,620,12))
        self.p\
            .scatter({"x": x, "y": y},s=100)\
            .plotFitLine(color='#FF9700')\
            .grid(x=True,y=True,color='#9F9F9F')\
            .setXyLabel(xLabel='Temperature C',yLabel='Sales')\
            .setXyLimits((10,26),(0,700))\
            .setXyFormat(
                yFormat=lambda v,_:'$'+str(int(v))
            )\
            .activate()\
            .saveAndShow()
        r = self.p.getR()
        self.assertAlmostEqual(r, .96, 2)
        x=19
        p = self.p.predict(x)
        self.assertAlmostEqual(p, 418.7,1)
        self.p.scatter(x,p).saveAndShow()
        m,b,lineEquation=self.p.leastSquaresRegression()
        self.assertAlmostEqual(m,37.6,1)
        self.assertAlmostEqual(b,-296.1,1)
    def test_m_b(self):
        filePath = "data/sun_ice.csv"
        path = getPath(filePath)
        df=self.p.readCsv(path)
        print(df)
        x=df['"x"\nHours of Sunshine']
        y=df['"y"\nIce Creams Sold']
        self.assertEqual(len(x),5)
        m,b,lineEquation=self.p.leastSquaresRegression(x,y)
        self.assertAlmostEqual(m,1.5,1)
        self.assertAlmostEqual(b,.3,1)
        self.assertEqual(lineEquation,'y=1.5x+0.3')
if __name__ == '__main__':
    unittest.main()
