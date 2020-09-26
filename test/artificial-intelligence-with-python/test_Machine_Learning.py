import sys
import os

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# print(SCRIPT_DIR)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from utilities import getPath,parseNumber
from ml import Panda
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class PlotAI(Plot,Panda):
            def __init__(self,arg=None):
                Plot.__init__(self)
        cls.p = PlotAI()
        cols='''Index Date         Location   MinTemp ... RainToday  RISK_MM  RainTomorrow'''
        s='''0   2008-12-01   Albury           13.4 ...         No             0.0                   No
        1   2008-12-02   Albury            7.4 ...          No             0.0                   No
        2   2008-12-03   Albury          12.9 ...          No             0.0                   No
        3   2008-12-04   Albury            9.2 ...          No             1.0                   No
        4   2008-12-05   Albury          17.5 ...          No             0.2                   No'''
        cls.df=cls.p.DataFrame(s,cols)
    def test_to_csv(self):
        p = '/Users/zhanghongliang/Documents/machine_learning/test/artificial-intelligence-with-python/data/weatherAUS.csv'
        self.p.to_csv(self.df,p)
        self.p.read_csv(p)
    def test_Machine_Learning(self):
        self.assertEqual(self.df.shape,(5,8))
        self.assertIsInstance(self.df.count().sort_values()[0],np.int64)

if __name__ == '__main__':
    unittest.main()
