import math,inspect, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator,PercentFormatter
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from do_statistics.doStats import DoStats
class Plot(DoStats):
    red='#F11F10'
    white='#fff'
    black = '#0E0E0E'
    ax=None
    def plts(self,t):
        length=len(t)
        fig, axes = plt.subplots(length, 1)
        # add a big axes, hide frame
        pltIndex=length*100+11
        for index,item in enumerate(t):
            # ax=fig.add_subplot(pltIndex+index, )
            # ax = plt.subplot(pltIndex+index)
            if 'isScatter' in item:
                func=axes[index].scatter
            else:
                func=axes[index].plot
            func(item['x'], item['y'],  c=self.black, linewidth=3)
            axes[index].set_title(item['title'])
            axes[index].set_ylabel('PD' if index else 'Val')
        plt.tight_layout()
    def getPlt(self):
        return plt
    # Make the shaded region
# a, b = 2, 9  # integral limits
    def setShadedRegion(self, ax, a, b,ix=None,iy=None,facecolor='red'):
        def func(x):
            return (x - 3) * (x - 5) * (x - 7) + 85
        # Make the shaded region
        if ix is None:
            ix = np.linspace(a, b)
        if iy is None:
            iy = func(ix)
        verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
        poly = Polygon(verts, facecolor=facecolor, edgecolor='0.5')
        ax.add_patch(poly)
    def getXyData(self,ax=None,**kwargs):
        x,y = self.getProbabilityDensity(**kwargs)
        if self.ax is None:
            if ax is None:
                fig, ax = plt.subplots()
            self.ax=ax
        return plt,self.ax,x,y
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
    def getFigAx(self):
        return plt.subplots()
    def plotND(self,x=None,y=None,yLable=None,mean=None, bars=100,l=None,labels=None,density=False,format_fn=None,annotation=None,callback=None):
        if l is None:
            l = self.list
        if isinstance(l, set):
            l = list(l)
        if len(np.array(l).shape) == 1:
            l = [l]
        fig, ax = self.getFigAx()
        self.drawGridLines(ax,x,y,density)
        n, bins, rects=self.plotHist(ax,l,bars,labels,y,density=density)
        self.setAxes(ax,x,y,yLable,format_fn,plt)
        self.plotLines(ax, mean, y, density)
        if annotation is None and y:
            annotation = [{
                'txt': 'Average = {0}'.format(mean),
                'position': (mean, y[1] + 10),
                'color': self.black
            }]
        self.drawTxt(ax, annotation, center=True)
        if callable(callback):
            callback(plt,ax,np,bins)
        this_function_name = inspect.currentframe().f_code.co_name
        self.saveAndShow(this_function_name)
        # self.show()
    def setBarsCol(self,N, bins, patches):
        colors = ['#005792', '#008BC4', '#69A8D4', '#BCC8E4']
        allCol=list(reversed(colors))+colors
        i = -3
        m = 0
        length=len(patches)
        for index,thispatch in enumerate(patches):
            if thispatch.xy[0] > i:
                if m<len(allCol)-1:
                    m +=1
                i += 1
            color = allCol[m]
            if index<length-1:
                nextPatch = patches[index + 1]
                if nextPatch.xy[0] > i:
                    color=self.white
            thispatch.set_facecolor(color)
    def plotLines(self, ax, mean, y,notHasOffset,color=None):
        if y is None or mean is None:
            return
        if color is None:
            color=self.white if notHasOffset else self.black
        x1 = x2 = mean
        y1, y2 = y[0], y[1] if notHasOffset else y[1]-40
        dashes = [5, 5]  # 10 points on, 5 off, 100 on, 5 off
        line1, = ax.plot([x1,x2],[y1,y2],'-' if notHasOffset else '--', linewidth=1,color=color)
        not notHasOffset and line1.set_dashes(dashes)
    def setAxes(self, ax, x, y, yLable,format_fn=None,plt=None):
        if y is None:
            return
        if format_fn is None:
            format_fn=self.format_fn
        # set the y limit
        plt.xlim(x[0], x[1])
        plt.ylim(y[0], y[1])
        if format_fn is None:
            x=range(0,x[1]+1,10)
            ax.set_xticks(x)
            plt.xticks(fontsize=8)
        self.setLables(ax, isMajor=False, format_fn=format_fn)
        ax.set_ylabel(yLable)
    def format_fn(self,tick_val, tick_pos):
            if int(tick_val) ==0:
                return '<{0}'.format(tick_val)
            elif int(tick_val) == 230:
                return '    {0}+'.format(tick_val)
            else:
                return tick_val
    def plotHist(self, ax,l,bars,labels,y,density=False):
        colors=['#FF5252','#535EB2']+plt.rcParams['axes.prop_cycle'].by_key()['color']
        for index, val in enumerate(l):
            n, bins, rects = ax.hist(val, bins=bars, color=colors[index], alpha=.8, orientation='vertical', density=density, label=labels[index] if labels else '')
            self.setBarsCol(n, bins, rects)
            if y is not None and len(l)>1:
                for r in rects:
                    height = r.get_height() / r.get_width()
                    if height>y[1]-50>0:
                        r.set_height(y[1]-100+height/10*np.random.rand() )
                    else:
                        r.set_height(height)
        labels and ax.legend(loc='upper right')
        return n, bins, rects
    def drawGridLines(self, ax, x, y,density):
        if y is None or density:
            return
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
    def saveAndShow(self, this_function_name=None, enableShow=False):
        if this_function_name is None:
            if hasattr(self,'this_function_name'):
                this_function_name=self.this_function_name
            else:
                this_function_name='drawFunction'
        plt.savefig("img/{0}.png".format(this_function_name))
        enableShow and self.show()
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
    def drawTxt(self, ax, annotation,center=False):
        if annotation is None:
            return
        for anno in annotation:
            d={'hasLine':False,**anno}
            hasLine,position, txt,color=d.values()
            if color is None:
                color=self.red
            if isinstance(txt, str):
                txt=[txt]
            strings = [str(item) for item in txt]
            ax.text(position[0], position[1], "\n".join(strings),
            horizontalalignment='center' if center else 'left',
            fontsize=14,
            # verticalalignment='top',
            # transform=ax.transAxes,
            color=color)
            if hasLine:
                self.plotLines(ax, position[0], (position[1]-.02,position[1]), notHasOffset=True,color=self.black)
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
