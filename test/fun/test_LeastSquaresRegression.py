import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest
from functools import reduce
from graphic.plot import Plot
from predict import Predict
from data.handle_data import HandleData
from utilities import getPath,parseNumber
import numpy as np
class TDD_TEST_LEASTSQUARESREGRESSION(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class PlotAI(Plot,Predict,HandleData):
            def __init__(self,arg=None):
                Plot.__init__(self)
                HandleData.__init__(self)
        self.p = PlotAI()
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
            .saveAndShow()
        r = self.p.getR()
        self.assertAlmostEqual(r, .96, 2)
        x=19
        p = self.p.predict(x)
        self.assertAlmostEqual(p, 418.7,1)
        self.p.scatter(x,p).saveAndShow()
        m,b,lineEquation,fitVals,error=self.p.leastSquaresRegression()
        self.assertAlmostEqual(m,37.6,1)
        self.assertAlmostEqual(b,-296.1,1)
    def test_m_b(self):
        filePath = "data/sun_ice.csv"
        path = getPath(filePath)
        df=self.p.readCsv(path)
        # print(df)
        x=df['x']
        y=df['y']
        predicted=df.iloc[:, 2]
        err=df.iloc[:, 3]
        equation=predicted.name
        self.assertEqual(len(x),5)
        m,b,lineEquation,fitVals,error=self.p.leastSquaresRegression(x,y,roundTo=3)
        self.assertAlmostEqual(m,1.518,3)
        self.assertAlmostEqual(b,.305,3)
        self.assertEqual(lineEquation,'y = 1.518x + 0.305')
        self.assertEqual(lineEquation,equation)
        self.assertEqual(predicted.tolist(),list(map(lambda y:round(y,2),fitVals)))
        self.assertEqual(err.tolist(),list(map(lambda e:round(e,2),error)))
        self.p\
            .scatter({"x": x, "y": y},s=100)\
            .plotFitLine(color='#FF9700')\
            .grid(x=True,y=True,color='#9F9F9F')\
            .setXyLabel(xLabel='Temperature C',yLabel='Sales')\
            .setXyLimits((0,10),(0,15))\
            .setXyFormat(
                yFormat=lambda v,_:None if int(v) % 5 else int(v),
                xFormat=lambda v,_:None if int(v) % 5 else int(v)
            )\
            .saveAndShow()\
            .freeze()
        p=self.p.predict(8)
        self.assertAlmostEqual(p,12.45,2)
    def test_least_squares_definition(self):
        # A way of finding a "line of best fit" by making the total of the square of the errors as small as possible (which is why it is called "least squares").
        x, y = self.p.getSortedXyInt((0,10,10),(2.5,8,10))
        lineX=reduce(lambda acc,item:acc+[list(item)],zip(x,x),[])
        self.assertRaises(ValueError,lambda:self.p.predict(x))
        self.p.setInfo({'x':x,'y':y})
        predictY=self.p.predict(x)
        lineY=reduce(lambda acc,item:acc+[list(item)],zip(y,predictY),[])
        annos=self.p.getAnnos(zip(x,y),predictY)
        self.p\
            .clear()\
            .scatter(s=100)\
            .plotFitLine(color='#FF9700')\
            .drawTxt(annos)\
            .plotLine(lineX,lineY,isSeparate=True)\
            .saveAndShow()\
            .freeze()
        x=self.p.getLinspaceData(0,10,10)
        y=self.p.getYByFunc(x,[1,2.5])
        self.p\
            .plotLine(x,y)\
            .saveAndShow()
if __name__ == '__main__':
    unittest.main()
