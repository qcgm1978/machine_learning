import unittest,numpy as np
import pandas
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
        df=self.ai.df
        self.ai\
            .preprocessStratified()\
            .preprocessDropCols('Time')
        # Create a new data frame with the first "3000" samples
        df_sample = df.iloc[:3000, :]
        # Now count the number of samples for each class
        ls = self.ai.valueCounts('Class',df_sample).tolist()
        self.assertEqual(ls,[2508,self.fraud ])
        self.assertTrue(self.ai.isBalance(ls))
if __name__ == '__main__':
    unittest.main()
