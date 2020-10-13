import unittest
# from utilities import getPath,parseNumber,update_json
from .np_f_b import *
from .t_g_d import *
class TDD_G_D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_g_d(self):
        r = self.get_tendency(l,condition=lambda item:item<1)        
        r1 = self.get_tendency(l1)        
        self.assertTrue(r)
        self.assertTrue(r1)

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
                