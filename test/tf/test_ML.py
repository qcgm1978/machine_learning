import os,time
t0=time.time()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import unittest
import numpy as np
from .beginners import *
from utilities import getPath,parseNumber,update_json,truncate
class TDD_ML_MODELS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cols=10
        a=Beginners(cls.cols)
        cls.b=a.prepare_data().get_model().op_model()
    def test_type(self):
        b=self.b
        self.assertEqual(type,type(Beginners))
        self.assertIsInstance(Beginners,object)
        self.assertEqual(object,Beginners.__base__)
        self.assertEqual('Beginners',Beginners.__name__)
        self.assertIsInstance(b,Beginners)
        self.assertEqual(type(b),Beginners)
        self.assertEqual(type(b),b.__class__)
    def test_ML_models(self):
        b=self.b
        cols=self.cols
        self.assertIsInstance(b.predictions,np.ndarray)
        self.assertEqual(b.predictions.shape,(1,cols))
        self.assertEqual(b.predictions.dtype,np.float32)
        self.assertEqual(len(b.probabilities),1)
        prob=map(lambda item:item>0 and item<1,b.probabilities[0])
        self.assertTrue(all(prob))
        self.assertTrue(callable(b.loss_fn))
        self.assertAlmostEqual(truncate(b.untrain_prob),truncate(b.negative_log_prob),0)
        self.assertTrue(b.is_history)
        self.assertFalse(b.is_s)
        self.assertAlmostEqual(b.sc[1],.98,2)
        self.assertEqual(b.p_m.shape,(5,cols))
    @classmethod
    def tearDownClass(cls):
        p = '/Users/zhanghongliang/Documents/ml/test/tf/json-update.json'
        duration=time.time()-t0
        data = update_json(p,duration)
        print( 'Previouse two duration: {0}'.format(data[-2:]),'\n','Difference: {0}'.format(data[-1]-data[-2]))
if __name__ == '__main__':
    unittest.main()
