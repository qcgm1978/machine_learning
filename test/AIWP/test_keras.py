import re,inspect, unittest,numpy as np
import pandas
from pandas.core.indexes import category
from tensorflow.python.keras.callbacks import History
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
        cls.cols='''1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)'''
    def test_cols(self):
        l=re.split(r'\n',self.cols)
        l=list(map(lambda item:re.sub(r'^[0-9. ]+','',item.strip()),l))
        self.ai.setColumnsName(l)
        self.assertEqual(self.ai.columnsName[8],'Class variable (0 or 1)')
    def test_data_preprocessing(self):
        ai = self.ai
        dim = 8
        objective = 'onset of diabetes as 1 or not as 0'
        temp=ai\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessLoadData(self.path,isPD=True)\
            .preprocessSplit(dim)\
            .preprocessNormalize()\
            .buildModel()\
            .ModelEvaluationOptimization()
            
        scoreTime=temp.predict()
        self.assertEqual(ai.shape,(767, 9))
        io=ai.getInputOutput()
        self.assertCountEqual(io,{'formula':'y = f(X)','input':list(range(0,9)),'output':[9]})
        # split into input (X) and output (y) variables
        # Purely integer-location based indexing for selection by position.
        X,y = ai.splitIO(range(0,dim))
        self.assertEqual(X.shape,(767,dim))
        self.assertEqual(ai.input_dim,dim)
        self.assertEqual(ai.layers,3)
        self.assertAlmostEqual(scoreTime[0][1],.8,1)
        # scoreTime,average=temp.predict(times=5)
        # self.assertAlmostEqual(average,.77,2)
        # make class predictions with the model
        binaryPredict = ai.predictdenseSequential(5)
        self.assertAlmostEqual(binaryPredict,1,0)
        if ai.hasLoadModel:
            self.assertRaises(RuntimeError,temp.debugSave,isJSON=False)
        # ai.debugSave(isSave=True)
    def test_load_model(self):
        pass
        accuracyScore=self.ai.debugLoadModel(isJSON=False,isSave=True).debugEvalModel().predict()
        self.assertEqual(accuracyScore,('accuracy','74.55%'))
if __name__ == '__main__':
    unittest.main()
