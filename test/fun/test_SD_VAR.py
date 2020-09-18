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
        # self.p.plotGrid(l)
        v=self.p.getVariance(l)
        v1=self.p.getVariance(l,ddof=1)
        self.assertEqual(v,21704)
        self.assertEqual(v1,27130)
        sd=self.p.getPSD(l)
        sd1=self.p.getSD(l)
        self.assertAlmostEqual(sd,147,0)
        self.assertAlmostEqual(sd1,165,0)
    def test_using(self):
        sd = 1
        p=self.p.getProbability(sd,isPercent=True)
        self.assertEqual(p,'68%')
        def func(ax,plt):
            yVal=self.p.getPdf(-sd,0,sd)
            x,y = [-sd,0],[yVal,yVal]
            self.p.annotate(x,y,s=r"$\}$",fontsize=47,rotation=90,isScatter=True)
            self.p.drawTxt({'fontsize':18,'center':'center','color':'red','position':[-.55,.28],'txt':1})
            self.p.drawTxt({'fontsize':18,'center':'center','color':'red','position':[.45,.28],'txt':1})
            self.p.drawTxt({'fontsize':22,'center':'center','color':'red','position':[0,.33],'txt':'Standard Deviations'})
            self.p.drawTxt({'fontsize':32,'center':'center','color':'white','position':[0,.1],'txt':p})
        # self.p.plotStdND(
        #     # func=func,
        #     barCol='#0080CF',
        #     cutLineCol='black',
        # ).saveAndShow()
        self.p.pltNdLine(callback=func,clip=(-sd,sd)).saveAndShow()
    def test_sample_data(self):
        pass
if __name__ == '__main__':
    unittest.main()
