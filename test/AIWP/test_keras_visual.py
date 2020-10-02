import re,inspect, unittest,numpy as np
import pandas
from pandas.core.indexes import category
from utilities import getPath,parseNumber
from .dl import DL
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class AI(DL):
            pass
        cls.ai = AI({'topFeaturesNum':3})
        cls.path = 'data/pima-indians-diabetes.data.csv'
        cls.columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date','WindGustDir','WindDir9am','WindSpeed9am','WindDir3pm']
        cls.target='RainTomorrow'
        cls.fraud = 492
    def test_cols(self):
        ai = self.ai
        ai\
            .sequential()\
            .interpretPlotModel()
        self.assertEqual(ai.summary()[0],'Model: "sequential"')
    def test_history(self):
        ai = self.ai
        dim = 8
        objective = 'onset of diabetes as 1 or not as 0'
        ai\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessLoadData(self.path,isPD=True)\
            .preprocessSplit(dim)\
            .preprocessNormalize()\
            .buildModel()\
            .ModelEvaluationOptimization()
        ai.interpretPlotCurve()
        # history=ai.getHistory()
        # self.assertCountEqual(list(history),['accuracy', 'loss', 'val_accuracy', 'val_loss'])
        # print(ai.train_history.history['accuracy'][:5],ai.train_history.history['val_accuracy'][:5])
if __name__ == '__main__':
    unittest.main()
