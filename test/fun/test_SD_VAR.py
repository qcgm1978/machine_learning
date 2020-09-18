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
        self.assertEqual(v,21704)
        sd=self.p.getPSD(l)
        self.assertAlmostEqual(sd,147,0)
    def test_using(self):
        p=self.p.getProbability(1,isPercent=True)
        self.assertEqual(p,'68%')
        def func(ax,plt):
            yVal=self.p.getPdf(-1,0,1)
            x,y = [-1,0],[yVal,yVal]
            self.p.annotate(x,y,s=r"$\}$",fontsize=47,rotation=90,isScatter=True)
            self.p.drawTxt({'fontsize':18,'center':'center','color':'red','position':[-.55,.28],'txt':1})
            self.p.drawTxt({'fontsize':18,'center':'center','color':'red','position':[.45,.28],'txt':1})
            self.p.drawTxt({'fontsize':22,'center':'center','color':'red','position':[0,.33],'txt':'Standard Deviations'})
            self.p.drawTxt({'fontsize':32,'center':'center','color':'white','position':[0,.1],'txt':'68%'})
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
        # self.p.plotStdND(
        #     # func=func,
        #     barCol='#0080CF',
        #     cutLineCol='black',
        # ).saveAndShow()
        self.p.pltNdLine(callback=func,clip=30).saveAndShow()
if __name__ == '__main__':
    unittest.main()
