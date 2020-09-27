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
from .ml import ML
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml = ML()
        cls.path = '/Users/zhanghongliang/Documents/machine_learning/test/AIWH/data/weatherAUS.csv'
        cls.columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date','WindGustDir','WindDir9am','WindSpeed9am','WindDir3pm']
    def test_to_csv(self):
        pass
        # self.p.to_csv(self.df,p)
    def test_data_preprocessing(self):
        self.assertRaises(AttributeError,lambda:self.ml.__loadDataSet(self.path))
        self.ml.preprocessingData(self.path)
        self.assertEqual(self.ml.df.shape,(142193, 24))
        self.assertEqual(self.ml.getFeatures(),24)
        self.assertAlmostEqual(self.ml.getObservations(),145e3,-5)
        vals = self.ml.df.count().sort_values()
        self.assertIsInstance(vals[0],np.int64)
        self.assertAlmostEqual((vals[4]-vals[0])/vals[4],.42,2)
    def test_ml(self):
        return
        rows,cols=self.p.df.shape
        objective = 'possibility of rain'
        score,time=\
        self.p\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessingData(self.path,self.columns)\
            .explore()\
            .buildModel()\
            .ModelEvaluationOptimization()\
            .predict()
        self.assertEqual(self.p.objective,objective)
        cols=cols-len(columns)
        self.assertEqual(cols,13)
        rows = 115160
        self.assertEqual(self.p.df.shape[0],rows)
        self.assertEqual(self.p.getXcolumns().tolist(),['Rainfall', 'Humidity3pm', 'RainToday'])
        self.assertAlmostEqual(score[1],.8,1)
        self.assertLess(time[1],1)
        # score,time=self.p\
        #     .buildModel(1)\
        #     .getScoreTime()
        # self.assertAlmostEqual(score[1],.8,1)
        # self.assertLess(time[1],7)
        # score,time=self.p\
        #     .buildModel(2)\
        #     .getScoreTime()
        # self.assertAlmostEqual(score[1],.8,1)
        # self.assertLess(time[1],6)
        # score,time=self.p\
        #     .buildModel(3)\
        #     .getScoreTime()
        # self.assertGreater(score[1],.8)
        # self.assertLess(time[1],6)
if __name__ == '__main__':
    unittest.main()
