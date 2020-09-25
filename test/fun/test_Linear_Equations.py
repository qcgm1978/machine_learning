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
        # equation='5x = 6'
        coefs=self.d.getCoefs(equation)
        intercepts=self.d.getIntercept(coefs)
        append = [-1,1,2]
        x=self.d.getLinspaceData(-2,3,50,append=append)
        y=self.d.getYByFunc(x,coefs)
        
        start=intercepts[1].copy()
        start[0]+=2
        dxy=[-2,0]
        annos=[{
            'position':intercepts[1],
            'txt':'m',
            'fontsize':20,
            'rotation':45,
            'color':'#0081E7',
            'center':'right'
        },{
            'position':start,
            'txt':'b',
            'fontsize':20,
            'color':'#23A200',
            'vertical':'center',
            'center':'left'
        }]
        print(annos)
        self.d\
            .plotLine(x,y,color=annos[0]['color'])\
            .pltCartesianCoordinate(hasLimit=True,intercepts=intercepts,other=append)\
            .arrow(*start,*dxy,color='#23A200',head_width=.3)\
            .drawTxt(annos)\
            .saveAndShow('pltCartesianCoordinate')
if __name__ == '__main__':
    unittest.main()
