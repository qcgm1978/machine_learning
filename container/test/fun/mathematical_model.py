import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from mathMethods.doMath import DoMath
from utilities import getPath,parseNumber
class TDD_MATHEMATICAL_MODEL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class PlotAI(Plot,DoMath):
            def __init__(self,arg=None):
                Plot.__init__(self)
        cls.d = PlotAI()
    def test_mathematical_model(self):
        totalV,totalNewV,change=self.d.cardboardModel(200,300,400,5,4)
        self.assertEqual(totalV,2.1489e7)
        self.assertEqual(totalNewV,2.1977088e7)
        self.assertAlmostEqual(change,.02,2)
    def test_MostEconomicalSize(self):
        # Cost= $0.30 × (0.08/w+ 4w2)
        x=self.d.getLinspaceData(.01,.5,100)
        y=list(map(lambda x:(4*x**2+.08/x)*.3,x))
        minY=min(y)
        ind=y.index(minY)
        xVal=x[ind]
        val=(xVal,minY)
        position=(xVal+.1,minY-.1)
        vals=np.round(val,2)
        txt={'position':position,'txt':'({0},{1})'.format(*vals),'vertical':'top'}
        self.d\
            .plotLine(x,y)\
            .setXyLimits((0,.5),(0,.4))\
            .scatter(*val,s=40,c='red')\
            .drawTxt(txt)\
            .arrow(*position,-.09,.09,color='red',head_width =.02)\
            .saveAndShow()
if __name__ == '__main__':
    unittest.main()
