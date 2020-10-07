import os,time
t0=time.time()
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import unittest
# from utilities import getPath,parseNumber,update_json
from .classification import getVersion
class TDD_CLASSIFY(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_classification(self):
        self.assertRaises(NameError,lambda:__version__)
        # self.assertEqual(getVersion(),'2.3.0')



















    @classmethod
    def tearDownClass(cls):
        p = '/Users/zhanghongliang/Documents/ml/test/tf/json-update.json'
        duration=time.time()-t0
        print(duration)
        # data = update_json(p,duration)
        # print( 'Previouse two duration: {0}'.format(data[-2:]),'\n','Difference: {0}'.format(data[-1]-data[-2]))
if __name__ == '__main__':
    unittest.main()

                