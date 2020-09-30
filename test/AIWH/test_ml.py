import unittest,numpy as np
from graphic.plot import Plot
from utilities import getPath,parseNumber
from .ml import ML
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml = ML({'topFeaturesNum':3})
        cls.path = getPath('test/AIWH/data/weatherAUS.csv')
        cls.path = 'data/weatherAUS.csv'
        cls.columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date','WindGustDir','WindDir9am','WindSpeed9am','WindDir3pm']
        cls.target='RainTomorrow'
    def test_to_csv(self):
        pass
        # self.ml.to_csv(self.df,p)
    def test_data_preprocessing(self):
        self.ml.preprocessLoadData(self.path)
        self.assertEqual(self.ml.shape,(142193, 24))
        self.assertEqual(self.ml.dataCols,24)
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
            .EDApredictor(target=self.target)\
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
        score,time=self.ml\
            .buildModel(1)\
            .ModelEvaluationOptimization()\
            .predict()
        self.assertAlmostEqual(score[1],.8,1)
        self.assertLess(time[1],10)
        score,time=self.ml\
            .buildModel(2)\
            .ModelEvaluationOptimization()\
            .predict()
        self.assertAlmostEqual(score[1],.8,1)
        self.assertLess(time[1],6)
        score,time=self.ml\
            .buildModel(3)\
            .ModelEvaluationOptimization()\
            .predict()
        self.assertGreater(score[1],.8)
        self.assertLess(time[1],6)
    def test_decision_tree(self):
        d1 = {"UK": 0, "USA": 1, "N": 2}
        d2 = {"YES": 1, "NO": 0}
        mapData = (("Nationality", d1), ("Go", d2))
        file = "data/shows.csv"
        X = ["Age", "Experience", "Rank", "Nationality"]
        target = "Go"
        d = ML()
        objective = 'go to a comedy show or not'
        p=d\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessLoadData(file)\
            .preprocessReplace(mapData)\
            .EDApredictor(X, target)\
            .buildModel(2)\
            .ModelEvaluationOptimization(enableGraph=False)\
            .predict(val=[40, 10, 6, 1],custom=True)
        self.assertEqual(p,(0, 'NO'))
        features=d.extractFeatures()
        self.assertEqual(features.tolist(),['Age', 'Experience', 'Rank','Nationality'])
if __name__ == '__main__':
    unittest.main()
