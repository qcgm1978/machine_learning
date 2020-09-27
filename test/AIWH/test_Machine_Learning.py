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
        cls.target='RainTomorrow'
    def test_to_csv(self):
        pass
        # self.ml.to_csv(self.df,p)
    def test_data_preprocessing(self):
        self.ml.preprocessLoadData(self.path)
        self.assertEqual(self.ml.shape,(142193, 24))
        self.assertEqual(self.ml.features,24)
        self.assertAlmostEqual(self.ml.observations,145e3,-5)
        vals = self.ml.df.count().sort_values()
        self.assertIsInstance(vals[0],np.int64)
        self.assertAlmostEqual((vals[4]-vals[0])/vals[4],.42,2)
    def test_ml(self):
        rows,cols=self.ml.df.shape
        objective = 'possibility of rain'
        #Change yes and no to 1 and 0 respectvely for RainToday and RainTomorrow variable
        rep=(('RainToday',{'No': 0, 'Yes': 1}),('RainTomorrow',{'No': 0, 'Yes': 1}))
        score,time=\
        self.ml\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessLoadData(self.path)\
            .preprocessReplace(rep)\
            .preprocessDrop(self.columns)\
            .preprocessNormalize()\
            .explorePredictor(self.target)\
            .exploreSimplify('Humidity3pm')\
            .buildModel()\
            .ModelEvaluationOptimization()\
            .predict()
        self.assertEqual(self.ml.objective,objective)
        self.assertEqual(self.ml.target,self.target)
        self.assertRaises(AttributeError,lambda: self.ml.__target)  
        cols=cols-len(self.columns)
        self.assertEqual(cols,13)
        rows = 115160
        self.assertEqual(self.ml.df.shape[0],rows)
        self.assertEqual(self.ml.xColumns.tolist(),['Rainfall', 'Humidity3pm', 'RainToday'])
        self.assertAlmostEqual(score[1],.8,1)
        self.assertLess(time[1],1)
        # score,time=self.ml\
        #     .buildModel(1)\
        #     .getScoreTime()
        # self.assertAlmostEqual(score[1],.8,1)
        # self.assertLess(time[1],7)
        # score,time=self.ml\
        #     .buildModel(2)\
        #     .getScoreTime()
        # self.assertAlmostEqual(score[1],.8,1)
        # self.assertLess(time[1],6)
        # score,time=self.ml\
        #     .buildModel(3)\
        #     .getScoreTime()
        # self.assertGreater(score[1],.8)
        # self.assertLess(time[1],6)
if __name__ == '__main__':
    unittest.main()
