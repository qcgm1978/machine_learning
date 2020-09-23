import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datatype import DataTypes
import numpy as np
import unittest
class TDD_REGRESSION(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X = ['Weight', 'Volume']
        y = 'CO2'
        """ to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TDD_REGRESSION, cls).setUpClass()
        cls.d = DataTypes({'x': X, 'y': y})
        # cls.the_resource = get_some_resource()
    def test_regression(self):
        x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
        y = [100, 90, 80, 60, 60, 55, 60, 65, 70,
             70, 75, 76, 78, 79, 90, 99, 99, 100]
        d = DataTypes({'x': x, 'y': y})
        d.scatter().polynomialRegressionLine().saveAndShow()
        r = d.getRSquared()
        self.assertAlmostEqual(r, .94, 2)
        x1=17
        speed = d.predictPolynomialRegression(x1)
        self.assertAlmostEqual(speed, 88.87, 2)
        d\
            .scatter(x1,speed)\
            .plotLine([0,x1,x1,x1],[speed,speed,0,speed])\
            .setXyLimits((0,20),(0,100))\
            .saveAndShow('polynomialRegressionLine')
    def test_bad_fit(self):
        x = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20,
             26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
        y = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10,
             26, 34, 90, 33, 38, 20, 56, 2, 47, 15]
        d = DataTypes({'x': x, 'y': y})
        d.scatter().polynomialRegressionLine().saveAndShow().freeze()
        r = d.getRSquared()
        self.assertAlmostEqual(r, .00995, 5)
    def test_multiple_regression(self):
        file = "data/cars.csv"
        # predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
        predictVals = [2300, 1300]
        predict = self.d.predictMultipleRegression(file, predictVals)
        self.assertAlmostEqual(predict[0], 107.209, 3)
        self.assertEqual(
            list(map(lambda x: round(x, 3), predict[1])), [0.008, 0.008])
        predictVals1 = [v if i else v + 1 for (i, v) in enumerate(predictVals)]
        self.assertEqual(predictVals1, [2301, 1300])
        predict1 = self.d.predictMultipleRegression(file, predictVals1)
        # These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
        self.assertAlmostEqual(predict1[0], predict[0]+predict[1][0], 3)
        # And if the engine size (Volume) increases by 1 ccm, the CO2 emission increases by 0.00780526 g.
        predictVals2 = [v+1 if i else v for (i, v) in enumerate(predictVals)]
        self.assertEqual(predictVals2, [2300, 1301])
        predict2 = self.d.predictMultipleRegression(file, predictVals2)
        self.assertAlmostEqual(predict2[0], predict[0] + predict[1][1], 3)
        predictVals3 = [v+1 for (i, v) in enumerate(predictVals)]
        self.assertEqual(predictVals3, [2301, 1301])
        predict3 = self.d.predictMultipleRegression(file, predictVals3)
        self.assertAlmostEqual(predict3[0], predict[0] + sum(predict[1]), 3)
        predictVals4 = [v if i else v +
                        1000 for (i, v) in enumerate(predictVals)]
        self.assertEqual(predictVals4, [3300, 1300])
        predict4 = self.d.predictMultipleRegression(file, predictVals4)
        self.assertAlmostEqual(
            predict4[0], predict[0] + predict[1][0] * 1000, 3)
    def test_scale(self):
        file = "data/cars.csv"
        scaleCols = ['Weight', 'Volume']
        d = DataTypes()
        scale = d.scale(file, scaleCols)
        # print(scale)
        self.assertIsInstance(scale, np.ndarray)
        # To get the number of dimensions, shape (size of each dimension) and size (number of all elements) of NumPy array, use attributes ndim , shape , and size of numpy. ndarray .
        self.assertEqual(scale.ndim, 2)
        self.assertEqual(scale.shape, (36, 2))
        self.assertEqual(scale.size, scale.shape[0]*scale.shape[1])
        l = list(scale)
        e = list(map(lambda x: round(x, 2), l[0]))
        self.assertEqual(e, [-2.1, -1.59])
    def test_scale_predict(self):
        file = "data/cars2.csv"
        X = ['Weight', 'Volume']
        y = 'CO2'
        d = DataTypes({'x': X, 'y': y})
        p = d.predictScale(file, [2300, 1.3])
        self.assertAlmostEqual(p[0], 107.209, 3)
    def test_test_model(self):
        np.random.seed(2)
        x = np.random.normal(3, 1, 100)
        y = np.random.normal(150, 40, 100) / x
        d = DataTypes({'x': x, 'y': y})
        # d.plotScatter('train')
        # d.plotScatter('test')
    def test_polynormial_line(self):
        np.random.seed(2)
        x = np.random.normal(3, 1, 100)
        y = np.random.normal(150, 40, 100) / x
        d = DataTypes({'x': x, 'y': y})
        # d.polynomialRegressionLine()
        p = d.predict4PolynomialRegression(6)
        p1 = d.predict4PolynomialRegression(60)
        self.assertAlmostEqual(p, 181, 0)
        self.assertAlmostEqual(p1, 55100190, 0)
        r2 = d.getRSquared('train')
        self.assertAlmostEqual(r2, .797, 3)
        r2 = d.getRSquared('test')
        self.assertAlmostEqual(r2, .838, 3)
        p = d.predict4PolynomialRegression(5)
        self.assertAlmostEqual(p, 24.88, 2)
if __name__ == '__main__':
    unittest.main()
