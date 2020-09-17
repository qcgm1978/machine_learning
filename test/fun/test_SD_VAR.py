import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from mysql_data.decorators_func import singleton
from graphic.plot import Plot
class TDD_TEST_SD_VAR(unittest.TestCase):
    @singleton
    def setUp(self):
        self.__class__.p=Plot()
    def test_test_SD_VAR(self):
        l='600mm, 470mm, 170mm, 430mm and 300mm'
        m=self.p.getMean(l)
        self.assertEqual(m,394)
        self.p.plotGrid(l)
        v=self.p.getVariance(l)
        self.assertEqual(v,21704)
        sd=self.p.getPSD(l)
        self.assertAlmostEqual(sd,147,0)
    def test_using(self):
        p=self.p.getProbability(1,isPercent=True)
        self.assertEqual(p,'68%')
        def func(ax,plt):
            x,y = [-.95,.04],[.2,.2]
            self.p.drawArrow(x,y,s=r"$\}$",fontsize=47,rotation=90,isScatter=True)
            self.p.drawTxt({'fontsize':18,'center':'right','color':'red','position':[-.45,.23],'txt':1})
            self.p.drawTxt({'fontsize':18,'center':'right','color':'red','position':[.55,.23],'txt':1})
            self.p.drawTxt({'fontsize':22,'center':'left','color':'red','position':[-1,.3],'txt':'Standard Deviations'})
            self.p.drawTxt({'fontsize':32,'center':'left','color':'white','position':[-.7,.1],'txt':'68%'})
            
        self.p.plotStdND(
            func=func,
            barCol='#0080CF',
            cutLineCol='black',
        ).saveAndShow()
if __name__ == '__main__':
    unittest.main()
