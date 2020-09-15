# https://www.mathsisfun.com/data/standard-normal-distribution.html
import sys
import os

from matplotlib.pyplot import xticks
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from mysql_data.decorators_func import singleton
class TDD_TEST_FUN_ND(unittest.TestCase):
    @singleton
    def setUp(self):
        self.__class__.d=Plot()
    def test_test_fun_nd(self):
        nd=self.d.getND(0,1)
        larger=filter(lambda item:item>0,nd)
        smaller=filter(lambda item:item<0,nd)
        self.assertAlmostEqual(len(list(larger)),len(list(smaller)),-3)
        mean,sd=self.d.getMeanSdByRange([1.1,1.7],.95)
        self.assertEqual(mean,1.4)
        self.assertAlmostEqual(sd,.15,2)
        name=self.d.getSdName()
        self.assertTrue( "z-score" in name)
        count,differ=self.d.standardizing(1.85,mean,sd)
        self.assertAlmostEqual(count,3)
        self.assertAlmostEqual(differ,.45)
    def test_plot(self):
        l=[26, 33, 65, 28, 34, 55, 25, 44, 50, 36, 26, 37, 43, 62, 35, 38, 45, 32, 28, 34]
        self.d.plotEvolution(l)
        self.d.saveAndShow('debug')
        # self.d.plotND(
        #     facecolor='pink', 
        #     l=l,  
        #     callback=callback,
        # )
        mean=self.d.getMean(l)
        sd=self.d.getPSD(l)
        self.assertAlmostEqual(sd,11.4)
        def x_format_fn(tick_val, tick_pos):
            tick = int(tick_val)
            if tick == 0:
                return mean
            elif abs(tick) == 4:
                return ''
            else:
                return '{1}{0}\n{2}'.format(tick,'+' if tick>0 else '',round(sd*tick+mean,1))
        def func(ax,plt):
            # ax.tick_params(axis='x', colors='red')
            firstTrimester=l[:3]
            l1=self.d.standardizing(firstTrimester,l=l,isPSD=True)
            line = plt.plot(l1[0],[0,0,0],'o',c='red',markersize=8)[0]
            line.set_clip_on(False)
            print(l1,'\n',xticks,'\n',l1[0][0])
        self.d.plotStdND(x_format_fn=x_format_fn,func=func)
    def test_save(self):
        self.d.saveAndShow()
if __name__ == '__main__':
    unittest.main()
