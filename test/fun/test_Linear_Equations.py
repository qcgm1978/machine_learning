import sys
import os
from numpy.lib.function_base import append
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from mathMethods.doMath import DoMath
from utilities import getPath,parseNumber
class TDD_TEST_LINEAR_EQUATIONS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class PlotAI(Plot,DoMath):
            def __init__(self,arg=None):
                Plot.__init__(self)
        cls.d = PlotAI()
    def test_test_Linear_Equations(self):
        equation='y = 2x + 1'
        equation='y = 3x - 6'
        coefs=self.d.getCoefs(equation)
        intercepts=self.d.getIntercept(coefs)
        print(intercepts)
        append = [-1,1,2]
        x=self.d.getLinspaceData(-2,3,50,append=append)
        y=self.d.getYByFunc(x,coefs)
        self.d\
            .plotLine(x,y)\
            .pltCartesianCoordinate(hasLimit=True,intercepts=intercepts,other=append)\
            .saveAndShow()
if __name__ == '__main__':
    unittest.main()
