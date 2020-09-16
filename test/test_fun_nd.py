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
        firstTrimester=l[:3]
        l1=self.d.standardizing(firstTrimester,l=l,isPSD=True)
        roundScore = list(map(lambda item:round(item,2),l1[0]))
        data=list(zip(firstTrimester,roundScore))
        def func(ax,plt):
            line = plt.plot(roundScore,[0,0,0],'o',c='red',markersize=8)[0]
            line.set_clip_on(False)
            self.d.drawArrow(ax,( -1.2,-3), [ .01,.09],arrowstyle='->')
            self.d.drawArrow(ax,( -.5,-1.9), [ .01,.17],arrowstyle='->')
            self.d.drawArrow(ax,( 2.3,2.3), [ .01,.09],arrowstyle='->')
        # self.d.plotStdND(
        #     x_format_fn=x_format_fn,
        #     func=func,
        #     annotation=[
        #         {'position':[-3,.1],'txt':data[0],'color':'black'},
        #         {'position':[-1.9,.18],'txt':data[1],'color':'black'},
        #         {'position':[2.3,.1],'txt':data[2],'color':'black'},
        #     ]
        # )
    def test_standardizing(self):
        l=[20, 15, 26, 32, 18, 28, 35, 14, 26, 22, 17]
        l1=self.d.getSdLowerThan(l=l,limitSD=-1)
        l1=map(lambda item:(item[0],round(item[1],2)),l1)
        self.assertEqual(list(l1),[(15, -1.21), (14, -1.36)])
    def test_half_std(self):
        def x_format_fn(tick_val, tick_pos):
            tick = tick_val
            if tick == 0:
                return 0
            else:
                sign = '+' if tick>0 else ''
                num=sign+str(int(tick))
                isInt=int(tick)==tick 
                return '{0}\n{1}σ\n{2}'.format(int(tick),num,'' if  abs(tick)==4 else '2%') if isInt else tick
        def func(ax,plt):
            x=np.linspace(-4,4,17)
            ax.set_xticks(x)
            # plt.xticks(fontsize=10)
            # line = plt.plot(roundScore,[0,0,0],'o',c='red',markersize=8)[0]
            # line.set_clip_on(False)
            # self.d.drawArrow(ax,( -1.2,-3), [ .01,.09],arrowstyle='->')
            # self.d.drawArrow(ax,( -.5,-1.9), [ .01,.17],arrowstyle='->')
            # self.d.drawArrow(ax,( 2.3,2.3), [ .01,.09],arrowstyle='->')
        percentages=[[.5, 0],[1, .5],[1.5, 1],[2,1.5],[2.5, 2],[3, 2.5],[3.5,3]]
        annos=self.d.getPercentage(percentages)
        self.d.plotStdND(
            x_format_fn=x_format_fn,
            func=func,
            xInterval=.5,
            barCol='#0084C8',
            cutLineCol='#14A5F4',
            annotation=annos,
            # annotation=[
            #     {'position':[-3,.1],'txt':data[0],'color':'black'},
            #     {'position':[-1.9,.18],'txt':data[1],'color':'black'},
            #     {'position':[2.3,.1],'txt':data[2],'color':'black'},
            # ]
        )
    def test_save(self):
        self.d.saveAndShow()
if __name__ == '__main__':
    unittest.main()
