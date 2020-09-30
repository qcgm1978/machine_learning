import inspect, unittest,numpy as np
import pandas
from tensorflow.python.keras.callbacks import History
from utilities import getPath,parseNumber
from .dl import DL
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class AI(DL):
            pass
        cls.ai = AI({'topFeaturesNum':3})
        cls.path = 'data/creditcard.csv'
        cls.columns = ['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date','WindGustDir','WindDir9am','WindSpeed9am','WindDir3pm']
        cls.target='RainTomorrow'
        cls.fraud = 492
    def test_data_preprocessing(self):
        self.ai.preprocessLoadData(self.path)
        self.assertEqual(self.ai.shape,(284807, 31))
        df = self.ai.df
        colClass = df['Class']
        fraudulen=colClass[colClass==1]
        self.assertEqual(len(fraudulen),self.fraud)
        counts = self.ai.valueCounts('Class')
        self.assertIsInstance(counts,pandas.core.series.Series)
        self.assertEqual(counts.name,'Class')
        self.assertEqual(counts.dtype,np.int64)
        self.assertEqual(counts[1:].tolist()[0],self.fraud)
        nonfraud=counts[0:].tolist()[0]
        self.assertFalse(self.ai.isBalance([self.fraud,nonfraud]))
    def test_preprocess(self):
        ai = self.ai
        df=ai.df
        objective = 'classify a transaction as either fraudulent or not based on past transactions'
        scoreTime=ai\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessLoadData(self.path)\
            .preprocessStratified()\
            .preprocessDropCols('Time')\
            .preprocessShuffle(3000)\
            .preprocessSplit()\
            .preprocessNormalize()\
            .buildModel()\
            .ModelEvaluationOptimization()\
            .predict()
        # Now count the number of samples for each class
        ls = ai.df_sample.tolist()
        self.assertEqual(ls,[2508,self.fraud ])
        self.assertTrue(ai.isBalance(ls))
        summary = ai.model.summary
        self.assertTrue(inspect.ismethod(summary))
        # Display the size of the train dataframe
        trainRows = 2400
        self.assertEqual(ai.train_feature.shape,(trainRows,29))
        # Display the size of test dataframe
        self.assertEqual(ai.train_label.shape,(trainRows,))
        self.assertIsInstance(ai.train_history,History)
        self.assertGreater(scoreTime[0][1],.97)
        self.assertLess(scoreTime[1][1],16)
        self.assertEqual(ai.testsInfo['False_positive_rate'],1)
        self.assertEqual(ai.testCounts,[600])
if __name__ == '__main__':
    unittest.main()
