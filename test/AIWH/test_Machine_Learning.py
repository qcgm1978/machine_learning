import sys
import os
PACKAGE_PARENT = '../..'
PACKAGE_CURRENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# print(SCRIPT_DIR)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_CURRENT)))
sys.path.append(os.path.dirname(__file__))
import unittest,numpy as np
from graphic.plot import Plot
from utilities import getPath,parseNumber
from AIWH.ml import Panda
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class PlotAI(Panda):
            def __init__(self,arg=None):
                pass
                # Plot.__init__(self)
        cls.p = PlotAI()
        cls.path = '/Users/zhanghongliang/Documents/machine_learning/test/AIWH/data/weatherAUS.csv'
        cls.p.loadDataSet(cls.path)
    def test_to_csv(self):
        pass
        # self.p.to_csv(self.df,p)
    def test_Machine_Learning(self):
        self.assertEqual(self.p.df.shape,(142193, 24))
        vals = self.p.df.count().sort_values()
        self.assertIsInstance(vals[0],np.int64)
        self.assertAlmostEqual((vals[4]-vals[0])/vals[4],.42,2)
    def test_data_preprocessing(self):
        rows,cols=self.p.df.shape
        columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date','WindGustDir','WindDir9am','WindSpeed9am','WindDir3pm']
        self.p\
            .loadDataSet(self.path)\
            .preprocessingData(columns)\
            .explore()
        cols=cols-len(columns)
        self.assertEqual(cols,13)
        rows = 115160
        self.assertEqual(self.p.df.shape[0],rows)
if __name__ == '__main__':
    unittest.main()
