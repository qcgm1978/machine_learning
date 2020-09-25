import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest
from graphic.plot import Plot
from utilities import getPath,parseNumber
from machine_learning import df
class TDD_MACHINE_LEARNING(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class PlotAI(Plot):
            def __init__(self,arg=None):
                Plot.__init__(self)
        cls.d = PlotAI()
    def test_Machine_Learning(self):
        self.assertIsInstance(self.d,object)
        self.assertEqual(df.shape,(5,8))
        print(df.count().sort_values())

if __name__ == '__main__':
    unittest.main()
