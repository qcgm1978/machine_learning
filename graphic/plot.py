import math,inspect, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import colors
import matplotlib.patches as patches
class Plot(object):
    red='#F11F10'
    white='#fff'
    black='#0E0E0E'
    def plotGroupedBar(
        self,
        l1,
        l2,
        title="Grouped bar chart with labels",
        l1txt="observed",
        l2txt="Predict",
        prop="Frenquency",
        txt=None,
    ):
        if txt is None:
            compare = self.compareByVariance([l1, l2])
            txt = "Variance Ratio: " + str(round(compare, 2))
        l1, l2,minLen = self.normalize(l1, l2)
        labels = range(1, minLen + 1)
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, l1, width, label=l1txt)
        rects2 = ax.bar(x + width / 2, l2, width, label=l2txt)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(prop)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.figtext(
            0.5, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=12
        )
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    "{}".format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        self.show()
    def normalize(self, l1, l2):
        lenO = len(l1)
        lenP = len(l2)
        if lenO <= lenP:
            minLen = lenO
            l2 = l2[:minLen]
        else:
            minLen = lenP
            l1 = l1[:minLen]
        return  l1, l2,minLen
    def plotBar(
        self, height, x=None,
    ):
        if x is None:
            x = self.list
        plt.bar(x, height)
        self.show()
    def plotND(self,x,y,yLable,mean, bars=100,l=None,labels=None):
        if l is None:
            l = self.list
        if isinstance(l, set):
            l = list(l)
        if len(np.array(l).shape) == 1:
            l = [l]
        fig, ax = plt.subplots()
        
        self.drawGridLines(ax,x,y)
        self.plotHist(ax,l,bars,labels,y)
        self.setHistAxes(ax,x,y,yLable)
        self.plotDotLine(ax,mean,y)
        this_function_name = inspect.currentframe().f_code.co_name
        self.save(plt,this_function_name)
        # self.show()
    def plotDotLine(self,ax,mean,y):
        x1 = x2 = mean
        y1, y2 = y[0], y[1]-40
        dashes = [5, 5]  # 10 points on, 5 off, 100 on, 5 off
        line1, = ax.plot([x1,x2],[y1,y2], '--', linewidth=1,color='black')
        line1.set_dashes(dashes)
        txt='Average = {0}'.format(mean)
        position=(mean,y2+10)
        self.drawTxt(ax,position,txt,center=True,color=self.black)
    def setHistAxes(self,ax,x,y,yLable):
        # set the y limit
        plt.xlim(x[0], x[1])
        plt.ylim(y[0],y[1])
        x=range(0,x[1]+1,10)
        ax.set_xticks(x)
        plt.xticks(fontsize=8)
        def format_fn(tick_val, tick_pos):
            if int(tick_val) ==0:
                return '<{0}'.format(x[0])
            elif int(tick_val) == 230:
                return '    {0}+'.format(x[-1])
            else:
                return tick_val
        self.setLables(ax, isMajor=False, format_fn=format_fn)
        ax.set_ylabel(yLable)
    def plotHist(self, ax,l,bars,labels,y):
        colors=['#FF5252','#535EB2']+plt.rcParams['axes.prop_cycle'].by_key()['color']
        for index, val in enumerate(l):
            n, bins, rects = ax.hist(val, bars,color=colors[index],alpha=.8, label=labels[index])
            for r in rects:
                height = r.get_height() / r.get_width()
                if height>y[1]-50:
                    r.set_height(y[1]-100+height/10*np.random.rand() )
                else:
                    r.set_height(height)
        ax.legend(loc='upper right')
    def drawGridLines(self, ax,x,y):
        wid = 5
        hei = 10
        nrows = y[1]
        ncols = x[1]
        xx = np.arange(x[0], ncols, wid)
        yy = np.arange(y[0], nrows, hei)
        for ind,xi in enumerate(xx):
            for index,yi in enumerate(yy):
                sq = patches.Rectangle((xi, yi), wid, hei, fill=True,color='white' if (index+ind)%2 else '#E6E6E6')
                ax.add_patch(sq)
        ax.relim()
        ax.autoscale_view()
    def polynomialRegressionLine(self):
        x = self.info["x"]
        y = self.info["y"]
        mymodel = np.poly1d(np.polyfit(x, y, 3))
        minX = int(min(x))
        maxX = int(max(x))  
        maxY = int(max(y))
        myline = np.linspace(minX, maxX, maxY)
        self.scatter()
        plt.plot(myline, mymodel(myline))
        self.show()
    def scatterDots(self,x,y):
        self.scatter(x,y)
        self.show()
    def setTxt(self,ax,title,xTxt,yTxt):
        ax.set_title('\n'.join(title),loc='left')
        ax.set_xlabel(xTxt)
        ax.set_ylabel(yTxt)
    def setLables(self,ax,format_fn,isMajor=True):
        if isMajor:
            locator = ax.xaxis.set_major_locator
        else:
            locator = ax.xaxis.set_minor_locator
        ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
        locator(MaxNLocator(integer=True))
    def scatterGrouped(self,l,title='',xTxt='',yTxt=''):
        fig, ax = plt.subplots()
        self.setTxt(ax,title,xTxt,yTxt)
        self.addScatter(ax,l)
        labels = list(map(lambda item: item[0], l))
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in range(1,3):
                return labels[int(tick_val)-1]
            else:
                return ''
        self.setLables(ax, labels,format_fn=format_fn)
        this_function_name = inspect.currentframe().f_code.co_name
        self.save(plt,this_function_name)
        # self.show()
    def save(self,plt,this_function_name):
        plt.savefig("img/{0}.png".format(this_function_name))
    def addScatter(self,ax,l):
        for ind,i in enumerate(l):
            y=i[1]
            if isinstance(i[0],str):
                x=[ind+1]*len(y)
            if len(y)==1:
                c=self.white  
            else:
                c=[self.black]*(len(y)-1)+[self.red]
                self.addArrowTxt(ax,ind,l)
            ax.scatter(x, y,c=c)
    def addArrowTxt(self,ax,ind,l ):
        txt=l[ind][2]
        x=ind+1
        mean=l[ind][1][-1]
        ratio=mean/10**math.ceil(math.log10(mean))
        x1, y1 = x+.1, mean-mean*ratio
        x2, y2 = x+.1, mean+mean*ratio
        position = (x + .2, mean - 300 * ratio)
        self.drawArrow(ax,[x1, x2], [y1, y2])
        self.drawTxt(ax,position,txt)
    def drawTxt(self, ax, position, txt, center=False, color=None):
        if color is None:
            color=self.red
        if isinstance(txt, str):
            txt=[txt]
        strings = [str(item) for item in txt]
        ax.text(position[0], position[1], "\n".join(strings),
        horizontalalignment='center' if center else 'left',
        # verticalalignment='top',
        # transform=ax.transAxes,
        color=color)
    def drawArrow(self,ax,x,y,arrowstyle="<->"):
        ax.plot(x,y)
        # Axes.annotate(self, text, xy, *args, **kwargs)
        # Annotate the point xy with text 'text'.
        # Optionally, the text can be displayed in another position xytext. An arrow pointing from the text to the annotated point xy can then be added by defining arrowprops.
        ax.annotate("",
                    xy=(x[0], y[0]), xycoords='data',
                    xytext=(x[1], y[1]), textcoords='data',
                    arrowprops=dict(arrowstyle=arrowstyle, color=self.red,
                                    shrinkA=0, shrinkB=0,
                                    ),
                    )
    def scatter(self, x=None, y=None):
        if x is None or y is None:
            x = self.info["x"]
            y = self.info["y"]
        l1, l2,minLen = self.normalize(x,y)
        plt.scatter(l1, l2)
    def show(self):
        plt.show()
    def scatterLine(self):
        mymodel = self.getModel()
        self.scatter()
        plt.plot(self.info["x"], mymodel)
        self.show()
