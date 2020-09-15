# https://www.mathsisfun.com/data/standard-normal-distribution.html
import sys
import os
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
        def format_fn(tick_val, tick_pos):
            if int(tick_val) == 0:
                return 0
            elif abs(int(tick_val)) == 4:
                return ''
            else:
                return '{0}σ'.format(int(tick_val))
        one = self.d.getProbability(σRange=[1, 0])/2*100
        oneσ = str(round(one, 1))+'%'
        two=self.d.getProbability(σRange=[2, 1])/2*100
        twoσ = str(round(two, 1))+'%'
        three=self.d.getProbability(σRange=[3, 2])/2*100
        threeσ = str(round(three, 1))+'%'
        four=self.d.getProbability(σRange=[4, 3]) / 2 * 100
        fourσ = str(round(four, 1)) + '%'
        self.assertAlmostEqual((one+two+three)*2,self.d.getProbability(3)*100)
        self.assertAlmostEqual((one + two + three + four) * 2, 100)
        
        def callback(plt,ax,np,bins):
            plt.setp( ax.yaxis.get_majorticklabels(), rotation=90 )
            y=np.linspace(0,.4,5)
            ax.set_yticks(y)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            # add a 'best fit' line
            # sigma = 1
            # mu=0
            # plt,ax,x,y=self.d.getXyData(ax=ax,x=np.array(sorted(l)),σ=sigma,μ=mu)
            # ax.plot(x, y, '-', c=self.d.black, linewidth=3)
        self.d.plotND( l=[l],  bars=100,facecolor=self.d.black)
    def test_save(self):
        self.d.saveAndShow()
if __name__ == '__main__':
    unittest.main()
