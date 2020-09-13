import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from do_statistics.doStats import DoStats
from mysql_data.decorators_func import singleton
class TDD_GRAPHIC_TECH(unittest.TestCase):
    @singleton
    def setUp(self):
        class PlotStat(Plot, DoStats):
            pass
        self.__class__.d = PlotStat()
        l1 = self.d.getND(0, 1, size=8000)
        
        one = self.d.getProbability(σRange=[1, 0])/2*100
        oneσ = str(round(one, 1))+'%'
        two=self.d.getProbability(σRange=[2, 1])/2*100
        twoσ = str(round(two, 1))+'%'
        three=self.d.getProbability(σRange=[3, 2])/2*100
        threeσ = str(round(three, 1))+'%'
        four=self.d.getProbability(σRange=[4, 3]) / 2 * 100
        fourσ = str(round(four, 1)) + '%'
        self.__class__.annotation=[
            {'position': (-.5, .2), 'txt': oneσ, 'color': self.d.white},
            {'position': (.5, .2), 'txt': oneσ, 'color': self.d.white},
            {'position': (-1.5, .02), 'txt': twoσ, 'color': self.d.white},
            {'position': (1.5, .02), 'txt': twoσ, 'color': self.d.white},
            {'position': (-2.5, .03), 'txt': threeσ , 'color': self.d.black,'hasLine':True},
            {'position': (2.5, .03), 'txt': threeσ , 'color': self.d.black,'hasLine':True},
            {'position': (-3.3, .01), 'txt':fourσ , 'color': self.d.black,'hasLine':True},
            {'position': (3.3, .01), 'txt':fourσ , 'color': self.d.black,'hasLine':True}
        ]
        self.__class__.x = np.array(sorted(l1))
    def test_fill(self):
        sigma = 1
        mu = 0
        colors = ['#005792', '#008BC4', '#69A8D4', '#BCC8E4']
        allCol=list(reversed(colors))+colors
        for i in range(8):
            self.d.drawFunction(x=self.x[i*1000:i*1000+1000], σ=sigma, μ=mu,isPlot=False,c=allCol[i])
    def test_graphic_tech(self):
        yLable = 'probability density'
        def format_fn(tick_val, tick_pos):
            if int(tick_val) == 0:
                return 0
            elif abs(int(tick_val)) == 4:
                return ''
            else:
                return '{0}σ'.format(int(tick_val))
        x = [-4, 4]
        y = [0, .4]
        plt=self.d.getPlt()
        plt.xlim(x[0], x[1])
        plt.ylim(y[0], y[1])
        # self.d.setAxes(self.ax,axisX,axisY,yLable,format_fn,plt=self.plt)
    def test_show(self):
        self.d.saveAndShow()
if __name__ == '__main__':
    unittest.main()
