import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from mysql_data.decorators_func import singleton
class TDD_TEST_ND_DATA_RULE(unittest.TestCase):
    @singleton
    def setUp(self):
        self.__class__.d = Plot()
        mu=[1070,950]
        l1 = self.d.getND(mu, 1, size=8000)
        sigma = 1
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
        x = np.array(sorted(l1))
        self.__class__.plt, self.__class__.ax,self.__class__.x, self.__class__.y = self.d.getXyData(x=x, σ=sigma, μ=mu)
    def test_fit_line(self):
        self.d.drawTxt(self.ax, self.annotation, center=True)
        colors = ['#005792', '#008BC4', '#69A8D4', '#BCC8E4']
        allCol=list(reversed(colors))+colors
        oneEighth = 1000
        self.d.setShadedRegion(self.ax, 0, 2, ix=self.x[ : 2000], iy=self.y[ : 2000], facecolor=allCol[0])
        for i in range(2,6):
            oneEighth = 1000
            self.d.setShadedRegion(self.ax, i-4, i-4 + 1, ix=self.x[i *oneEighth  : i *oneEighth  +oneEighth ], iy=self.y[i *oneEighth  : i *oneEighth  +oneEighth ], facecolor=allCol[i])
        self.d.setShadedRegion(self.ax, 2, 4, ix=self.x[6000 : ], iy=self.y[6000 : ], facecolor=allCol[0])
    def test_axis(self):
        self.plt.setp(self.ax.yaxis.get_majorticklabels(), rotation=90)
        yLable = 'probability density'
        def format_fn(tick_val, tick_pos):
            if int(tick_val) == 0:
                return 0
            elif abs(int(tick_val)) == 4:
                return ''
            else:
                return '{0}σ'.format(int(tick_val))
        axisX = [-4, 4]
        axisY=[0, .4]
        self.d.setAxes(self.ax,axisX,axisY,yLable,format_fn,plt=self.plt)
        axisY=np.linspace(0,.4,5)
        self.ax.set_yticks(axisY)
        self.plt.yticks(fontsize=14)
        self.plt.xticks(fontsize=14)
    def test_show(self):
        self.d.saveAndShow()
if __name__ == '__main__':
    unittest.main()
