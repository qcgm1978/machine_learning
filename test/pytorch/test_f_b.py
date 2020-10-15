import unittest,torch
# from utilities import getPath,parseNumber,update_json
from .evolution.np_f_b import NP
from .evolution.t_f_b import *
from .evolution.a_f_b import *
from .evolution.custom_f_b import *
from .evolution.nn_f_b import *
from .evolution.optim_f_b import *
from .evolution.subclass_f_b import *
from .evolution.share_f_b import *
class TDD_G_D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_g_d(self):
        l=NP()._l
        r = self.get_tendency(l,condition=lambda item:item<1)        
        r1 = self.get_tendency(l1)        
        r2 = self.get_tendency(l2)        
        r3 = self.get_tendency(l3)        
        r4 = self.get_tendency(l4)        
        r5 = self.get_tendency(l5)        
        r6 = self.get_tendency(l6)        
        r7 = self.get_tendency(l7)        
        self.assertTrue(r)
        self.assertTrue(r1)
        self.assertTrue(r2)
        self.assertTrue(r3)
        self.assertTrue(r4)
        self.assertTrue(r5)
        self.assertTrue(r6)
        # self.assertFalse(r7)
    
    def test_mm(self):
        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        mm = torch.mm(mat1, mat2)
        self.assertRaises(RuntimeError, torch.mm,mat2, mat1)
        self.assertEqual(mm.shape, (2, 3))
    def get_tendency(self,l,condition=None):
        r=True
        for index, item in enumerate(l):
            if index:
                meetCondition = condition(item) if condition else True
                if meetCondition and item >= l[index - 1]:
                    r = False
                    break        
        return r
if __name__ == '__main__':
    unittest.main()
