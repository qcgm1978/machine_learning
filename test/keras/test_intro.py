import json
import time
t0=time.time()
import unittest
# from tensorflow._api.v2.data import Dataset
# from utilities import getPath,parseNumber
# import numpy as np
from .Intro_m import Intro
class TDD_INTRO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intro = Intro([])
    def test_intro(self):
        self.assertIsInstance(self.intro,self.intro.getNp().ndarray)
        D=self.intro.getDataset()
        dataset = D.from_tensor_slices([1, 2, 3])
        self.assertIsInstance(dataset,D)
        self.assertTrue(self.intro.isValidData(self.intro))
        self.assertTrue(self.intro.isValidData(dataset))
p = '/Users/zhanghongliang/Documents/ml/test/keras/json-dump.json'
with open(p, "r") as jsonFile:
    data = json.load(jsonFile)
data.append(round(time.time()-t0))
with open(p, "w") as jsonFile:
    json.dump(data, jsonFile)
print( data[-2:],'\n',data[-1]-data[-2])
if __name__ == '__main__':
    unittest.main()
