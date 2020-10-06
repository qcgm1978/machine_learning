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
        cls.foo = 1
    def test_ML_models(self):
        self.assertIsInstance(predictions,np.ndarray)
        self.assertEqual(predictions.shape,(1,10))
        self.assertEqual(predictions.dtype,np.float32)
        self.assertEqual(len(probabilities),1)
        prob=map(lambda item:item>0 and item<1,probabilities[0])
        self.assertTrue(all(prob))
        self.assertTrue(callable(loss_fn))
        self.assertAlmostEqual(truncate(untrain_prob),truncate(negative_log_prob),0)
p = '/Users/zhanghongliang/Documents/ml/test/tf/json-update.json'
duration=time.time()-t0
data = update_json(p,duration)
print( data[-2:],'\n',data[-1]-data[-2])
if __name__ == '__main__':
    unittest.main()
