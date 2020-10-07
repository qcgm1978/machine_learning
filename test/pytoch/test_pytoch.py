#import os,time
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import unittest
# from utilities import getPath,parseNumber,update_json
from .what_is_pytoch import *
class TDD_PYTOCH(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_pytoch(self):
        gs=GetStarted()
        gs.Tensors()
        self.assertTrue(gs.is_t)
        self.assertTrue(gs.is_t2)
        self.assertEqual(gs.x3.shape,(5,3))
        flat_list = [not item for sublist in gs.x3 for item in sublist]
        self.assertTrue(all(flat_list))
        self.assertTrue(gs.is_f)
        self.assertTrue(gs.is_f1)
        self.assertTrue(gs.is_f2)
        self.assertEqual(gs.x5.shape,(5,3))
        self.assertEqual(gs.x6.shape,(5,3))
        self.assertEqual(list(gs.x6.size()),[5,3])

if __name__ == '__main__':
    unittest.main()
