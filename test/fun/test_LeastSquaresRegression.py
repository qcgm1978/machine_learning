import unittest
from graphic.plot import Plot
from mysql_data.decorators_func import singleton
from utilities import getPath,parseNumber
class TDD_TEST_LEASTSQUARESREGRESSION(unittest.TestCase):
    @singleton
    def setUp(self):
        class PlotAI(Plot):
            def __init__(self,arg=None):
                Plot.__init__(self)
        self.__class__.p = Plot()
    def test_test_LeastSquaresRegression(self):
        self.assertEqual(True,1)
if __name__ == '__main__':
    unittest.main()

                