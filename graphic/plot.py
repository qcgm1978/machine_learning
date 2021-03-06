import math,inspect,traceback, numpy as np
from matplotlib.pyplot import ylabel
from PIL.Image import NONE
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from pylab import arrow,grid
from do_statistics.doStats import DoStats
from utilities import getPath
class Plot(DoStats):
    red='#F11F10'
    white='#fff'
    black = '#0E0E0E'
    barCol=''
    ax=None
    xInterval=1
    freezeIndex=0
    def __init__(self):
        self.fig, self.ax = self.getFigAx()
        self.freezeIndex=0   
    def setInfo(self,info):
        self.info=info
    def getAnnos(self,data,txt='error',predictY=None):
        if len(np.array(data).shape)==1:
            data=[data]
        annos=[]
        for i,item in enumerate(data):
            t=txt
            annos.append({'position': item, 'txt': t, 'color': 'red','fontsize':18,'center':'left','vertical':'bottom' if predictY and item[1]<predictY[i] else 'top'})
        return annos
    def arrow(self,x1,y1,dx,dy,color='#9F8DFF',head_width = 0.15):
        arrow( x1, y1, dx, dy, length_includes_head = True, head_width = head_width,color=color)
        return self
    def pltCartesianCoordinate(self,hideAxis=False,hasLimit=False,intercepts=None,other=None):
        ax = self.ax
        left,right = ax.get_xlim()
        low,high = ax.get_ylim()
        arrow( left, 0, right -left, 0, length_includes_head = True, head_width = 0.15,color='#9F8DFF' )
        arrow( 0, low, 0, high-low, length_includes_head = True, head_width = 0.15,color='#F78E9E' ) 
        if hasLimit:
            x=self.x
            y=self.y
            minX=min(x)
            maxX=max(x)
            minY=min(y)
            maxY=max(y)
            xs=[0,minX,maxX,0,0]
            ys=[0,0,0,minY,maxY]
            if intercepts is None:
                intercepts=[]
            if other is None:
                other =[]
            otherData=intercepts+other
            for item in otherData:
                if isinstance(item,list):
                    xs.append(item[0])
                    ys.append(item[1])
                else:
                    ind=np.where(x==item)
                    yInd=y[ind]
                    xs.append(item)
                    ys.append(yInd[0])
            self.scatter(xs,ys)
            annos=[]
            for item in zip(xs,list(ys)):
                if item[0] in other:
                    txt='({0:.{2}f},{1:.{2}f})'.format(item[0],item[1],0)
                else:
                    t=list(filter(lambda v:v!=0,item))
                    if len(t):
                        intT=int(t[0])
                        if intT==t[0]:
                            txt=intT
                        else:
                            txt=t[0]
                    else:
                        txt= 0
                annos.append({'position': item, 'txt': txt, 'color': 'black','fontsize':18,'center':'left','vertical':'top'})
            self.drawTxt(annos)
        if hideAxis:
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False
            ) 
        grid()
        self.setImgName()
        return self
    def getLinspaceData(self,start,stop,count,append=None):
        lin=np.linspace(start,stop,count)
        if append:
            if not isinstance(append,list):
                append=[append]
            for item in append:
                lin=np.append(lin,item)
        return lin
    def grid(self,x=False,y=False,color='#7F1299'):
        y and self.ax.yaxis.grid(c=color)        
        x and self.ax.xaxis.grid(c=color)        
        return self
    def plotGrid(self,l):
        if isinstance(l,str):
            l=self.strToL(l)
        m=self.getMean(l)
        sd=self.getPSD(l)
        self.ax.set_facecolor('darkgray')
        self.fig.set_facecolor('darkgray')
        self.grid(y=True)
        for index,item in enumerate(l):
            self.ax.plot([index,index+1,index+.7,index+.7],[item,item,item+40,item],c='red',linewidth=3)
            self.annotate([index+1,index+1],[item,m],arrowstyle='<-',c='#760093',linewidth=3)
            diff=item-m
            x=index+1+(.1 if index<4 else -.5)
            self.drawTxt({'fontsize':18,'center':'left','color':'#7F1299','position':[x,diff/2+m],'txt':int(diff)})
        self.ax.plot([.1,5],[m+sd,m+sd],linewidth=3,c='#3700FF')
        self.annotate([.5,.5],[m+sd,m],arrowstyle='<->',c='#fff',linewidth=3)
        self.annotate([.5,.5],[m-sd,m],arrowstyle='<->',c='#fff',linewidth=3)
        self.drawTxt({'fontsize':18,'center':'right','color':'#fff','position':[.4,(m+sd+m)/2],'txt':int(sd)})
        self.drawTxt({'fontsize':18,'center':'right','color':'#fff','position':[.4,(m-sd+m)/2],'txt':int(sd)})
        self.ax.plot([.1,5],[m,m],linewidth=3,c='#00FF00')
        self.ax.plot([.1,5],[m-sd,m-sd],linewidth=3,c='#3700FF')
        y=np.linspace(0,600,4)
        self.ax.set_yticks(y)
        self.ax.tick_params(axis='y', colors='#7F1299',labelsize=20)
        self.setImgName()
        self.saveAndShow()
    def plotEvolution(self,l):
        t=self.getEvolutiveData(l)
        self.plts(t)
    def plts(self,t):
        length=len(t)
        fig, axes = plt.subplots(length, 1)
        # add a big axes, hide frame
        for index,item in enumerate(t):
            if 'isScatter' in item:
                func=axes[index].scatter
            else:
                func=axes[index].plot
            func(item['x'], item['y'],  c=self.black, linewidth=3)
            axes[index].set_title(item['title'])
            yLabel='PD' if index else 'Val'
            ax=axes[index]
            self.setXyLabel(ax,yLabel)
        plt.tight_layout()
    def setXyLabel(self,ax=None,xLabel=None,yLabel=None):
        if ax is None:
            ax=self.ax
        yLabel and ax.set_ylabel(yLabel)
        xLabel and ax.set_xlabel(xLabel)
        return self
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
    
    def getXyData(self,ax=None,func=None,**kwargs):
        if func is None:
            func=self.getProbabilityDensity
        x,y = func(**kwargs)
        if ax is None:
            if self.ax is None:
                fig, ax = plt.subplots()
            else:
                ax=self.ax
        return plt,ax,x,y
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
        self.__show()
    def normalize(self, l1, l2):
        if not isinstance(l1,list):
            l1=[l1]
        if not isinstance(l2,list):
            l2=[l2]
        lenO = len(l2)
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
        self.__show()
    def getFigAx(self):
        self.fig,self.ax=plt.subplots()
        return self.fig,self.ax
    def plotND(self,x=None,y=None,yLable=None,mean=None, bars=100,l=None,labels=None,density=False,format_fn=None,annotation=None,callback=None,facecolor=None):
        fig, ax = self.fig,self.ax
        self.drawGridLines(ax,x,y,density)
        n, bins, rects=self.plotHist(l,bars,labels,y,density=density,facecolor=facecolor)
        self.setAxes(ax,x,y,yLable,format_fn,plt)
        self.plotLines( mean, y, density)
        if annotation is None and y and mean:
            annotation = [{
                'txt': 'Average = {0}'.format(mean),
                'position': (mean, y[1] + 10),
                'color': self.black
            }]
        self.drawTxt(annotation)
        if callable(callback):
            callback(plt,ax,np,bins)
        self.setImgName()
        return self
    def setBarsCol(self,N, bins, patches):
        length=len(patches)
        if self.barCol:
            allCol=[self.barCol]*length
        else:
            colors = ['#005792', '#008BC4', '#69A8D4', '#BCC8E4']
            allCol=list(reversed(colors))+colors
        i = -3
        m = 0
        for index,thispatch in enumerate(patches):
            if thispatch.xy[0] > i:
                if m<len(allCol)-1:
                    m +=1
                i += self.xInterval
            color = allCol[m]
            if index<length-1:
                nextPatch = patches[index + 1]
                if nextPatch.xy[0] > i:
                    color=self.cutLineCol if hasattr(self,'cutLineCol') else self.white
            thispatch.set_facecolor(color)
    def plotLines(self, mean, y,notHasOffset,color=None):
        if y is None or mean is None:
            return
        if color is None:
            color=self.white if notHasOffset else self.black
        x1 = x2 = mean
        y1, y2 = y[0], y[1] if notHasOffset else y[1]-40
        dashes = [5, 5]  # 10 points on, 5 off, 100 on, 5 off
        line1, = ax.plot([x1,x2],[y1,y2],'-' if notHasOffset else '--', linewidth=1,color=color)
        not notHasOffset and line1.set_dashes(dashes)
    def setXyLimits(self,x,y):
        plt.xlim(*x)
        plt.ylim(*y)
        return self
    def setAxes(self, ax, x, y, yLable,format_fn=None,plt=None):
        if y is None:
            return
        # set the y limit
        self.setXyLimits(x,y)
        if format_fn is None:
            x=range(0,x[1]+1,10)
            ax.set_xticks(x)
            plt.xticks(fontsize=8)
        self.setLables(ax, isMajor=False, )
        ax.set_ylabel(yLable)
    def plotHist(self, l,bars,labels=None,y=None,density=False,facecolor=None,ax=None,insertBar=True,barCol=None):
        if l is None:
            l = self.list
        if isinstance(l, set):
            l = list(l)
        if len(np.array(l).shape) == 1:
            l = [l]
        if ax is None:
            ax=self.ax
        colors=['#FF5252','#535EB2']+plt.rcParams['axes.prop_cycle'].by_key()['color']
        for index, val in enumerate(l):
            if labels:
                label=labels[index] if labels else ''
            else:
                label=''
            n, bins, rects = ax.hist(val, bins=bars, color=barCol if barCol else colors[index], alpha=.8, orientation='vertical', density=density, label=label)
            facecolor and ax.set_facecolor(facecolor)
            insertBar and self.setBarsCol(n, bins, rects)
            if y is not None and len(l)>1:
                for r in rects:
                    height = r.get_height() / r.get_width()
                    if height>y[1]-50>0:
                        r.set_height(y[1]-100+height/10*np.random.rand() )
                    else:
                        r.set_height(height)
        labels and ax.legend(loc='upper right')
        self.setImgName()
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
    def polynomialRegressionLine(self,dataType='None'):
        x,y=self.getData(dataType)
        mymodel = np.poly1d(np.polyfit(x, y, 4))
        minX = int(min(x))
        maxX = int(max(x))  
        maxY = int(max(y))
        myline = np.linspace(minX, maxX, maxY)
        plt.plot(myline, mymodel(myline))
        self.setImgName()
        return self
    def scatterDots(self,x,y):
        self.scatter(x,y)
        self.__show()
    def setTxt(self,ax,title,xTxt,yTxt):
        ax.set_title('\n'.join(title),loc='left')
        ax.set_xlabel(xTxt)
        ax.set_ylabel(yTxt)
    def setLables(self,ax,isMajor=True):
        if isMajor:
            locator = ax.xaxis.set_major_locator
        else:
            locator = ax.xaxis.set_minor_locator
        xFormat=self.x_format_fn
        xFormat and self.setXyFormat(xFormat)
        locator(MaxNLocator(integer=True))
    def setXyFormat(self,ax=None,xFormat=None,yFormat=None):
        if ax is None:
            ax=self.ax
        xFormat and ax.xaxis.set_major_formatter(FuncFormatter(xFormat))
        yFormat and ax.yaxis.set_major_formatter(FuncFormatter(yFormat))
        return self
    def scatterGrouped(self,l,title='',xTxt='',yTxt=''):
        fig, ax = self.getFigAx()
        self.setTxt(ax,title,xTxt,yTxt)
        self.addScatter(ax,l)
        labels = list(map(lambda item: item[0], l))
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in range(1,3):
                return labels[int(tick_val)-1]
            else:
                return ''
        self.x_format_fn=format_fn
        self.setLables(ax, labels)
        self.setImgName()
    def clear(self):
        # self.ax.clear()
        # self.fig.clear()
        plt.clf()
        return self
    def saveAndShow(self, this_function_name=None, enableShow=False):
        if self.freezeIndex<Plot.freezeIndex:
            return self
        if this_function_name is None:
            if hasattr(self,'this_function_name'):
                this_function_name=self.this_function_name
            else:
                this_function_name='demo'
        # Hide the right and top spines
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        plt.tight_layout()
        fname = "img/{0}.png".format(this_function_name)
        file=getPath(fname)
        plt.savefig(file)
        enableShow and self.__show()
        Plot.freezeIndex=self.freezeIndex
        return self
    def activate(self):
        Plot.isFreezing=False
        return self
    def freeze(self,ind=0):
        self.freezeIndex=ind
        self.clear()
        return self
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
        self.annotate([x1, x2], [y1, y2],ax=ax)
        self.drawTxt({'position':position,'txt':txt})
    def drawTxt(self,  annotation,ax=None,position=None):
        if ax is None:
            ax=self.ax
        if annotation is None:
            return
        if not isinstance(annotation,(list,tuple)):
            annotation=[annotation]
        for anno in annotation:
            d={'hasLine':False,'fontsize':14,'center':'center','color':'red','vertical':'baseline','rotation':None,**anno}
            txt=d['txt']
            if isinstance(d['txt'], str):
                txt=[d['txt']]
            strings = [str(item) for item in txt] if isinstance(txt,list) else [str(txt)]
            ax.text(
                d['position'][0], 
                d['position'][1], "\n".join(strings),
                horizontalalignment=d['center'],
                fontsize=d['fontsize'],
                verticalalignment=d['vertical'],
                # transform=ax.transAxes,
                color=d['color'],
                rotation=d['rotation']
            )
            if d['hasLine']:
                self.plotLines(d['position'][0], (d['position'][1]-.02,d['position'][1]), notHasOffset=True,color=self.black)
        return self
    def annotate(self,x,y,c='red',ax=None,arrowstyle="<->",linewidth=1,s='',fontsize=None,rotation=None,isScatter=False):
        if ax is None:
            ax=self.ax
        ax.scatter(x,y,s=0) if isScatter else ax.plot(x,y)
        # Axes.annotate(self, text, xy, *args, **kwargs)
        # Annotate the point xy with text 'text'.
        # Optionally, the text can be displayed in another position xytext. An arrow pointing from the text to the annotated point xy can then be added by defining arrowprops.
        length=len(x) if isScatter else 1
        for i in range(length):
            ax.annotate(s,
                    xy=(x[i], y[i]), 
                    rotation=rotation,
                    fontsize=fontsize,
                    xycoords='data',
                    xytext=(x[i], y[i]),  
                    color='red',      
                    textcoords='data',
                    arrowprops=None if isScatter else dict
                    (
                        arrowstyle=arrowstyle, 
                        color=c,
                        shrinkA=0, 
                        shrinkB=0,
                        linewidth=linewidth
                    ),
        )
    def plotScatter(self, dataType="All"):
        x, y = self.getData(dataType)
        return self.scatter(x, y)
    def scatter(self, x=None, y=None,s=None,c=None):
        if isinstance(x,dict):
            self.info=x
        if x is None or y is None:
            x = self.info["x"]
            y = self.info["y"]
        l1, l2,minLen = self.normalize(x,y)
        plt.scatter(l1, l2,s=s,c=c)
        self.setImgName()
        return self
    def setImgName(self):
        name=traceback.extract_stack(None, 2)[0][2]
        # self.this_function_name=name
        return name
    
    def plotFitLine(self,color=None):
        x=self.info["x"]
        y = self.getModel()
        self.plotLine(x, y,color=color)
        self.setImgName()
        return self
    def plotLine(self,x, y,color=None,isSeparate=False):
        self.x=x
        self.y=y
        if isSeparate:
            #  two-dimensional (in that case, the columns represent separate data sets).
            x=np.transpose( x)
            y=np.transpose( y)
        plt.plot(x, y,color=color)
        self.setImgName()
        return self
    def format_fn(tick_val, tick_pos):
            if int(tick_val) ==0:
                return '<{0}'.format(tick_val)
            elif int(tick_val) == 230:
                return '    {0}+'.format(tick_val)
            else:
                return tick_val
    def getPercentage(self,percentages):
        annos=[]
        cumulative={}
        for index,item in enumerate(percentages):
            val = self.getProbability(σRange=item)/2*100
            cumu=self.getProbability(item[0])/2+.5
            cumulative[item[0]]=round(cumu*100,1)
            cumulative[-item[0]]=round(100-cumu*100,1)
            percent = str(round(val, 1))+'%'
            y= .3-.075*index
            hasLine=False
            color='#BCF54C'
            x=item[0]-.5
            if y<0:
                y=.03
                x+=.04
                hasLine=True
                color='#FA9805'
            annos.append({'position': (x,y), 'txt': percent, 'color': color,'fontsize':10,'hasLine':hasLine,'center':'left'})
            annos.append({'position': (-x,y), 'txt': percent, 'color': color,'fontsize':10,'hasLine':hasLine,'center':'right'})
        return annos,cumulative
    def plotStdND(self,x_format_fn=None,func=None,annotation=None,xInterval=None,barCol=None,cutLineCol=None,l=None):
        self.x_format_fn=x_format_fn
        if xInterval is not None:
            self.xInterval=xInterval
        if barCol is not None:
            self.barCol=barCol
        if cutLineCol is not None:
            self.cutLineCol=cutLineCol
        if l is None:
            l = self.getND(0, 1, size=10000)
        def callback(plt,ax,np,bins):
            plt.setp( ax.yaxis.get_majorticklabels(), rotation=90 )
            y=np.linspace(0,.4,5)
            ax.set_yticks(y)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            # add a 'best fit' line
            sigma = 1
            mu=0
            plt,ax,x,y=self.getXyData(ax=ax,x=np.array(sorted(l)),σ=sigma,μ=mu)
            ax.plot(x, y, '-', c=self.black, linewidth=3)
            callable(func) and func(ax,plt)
        if annotation is not None:
            annotation=annotation
        self.plotND(x=[-4, 4], y=[0, .4], l=[l],  bars=100, yLable='probability density', density=True,
        format_fn=True,
        callback=callback,
        annotation=annotation)
        return self
    def pltNdLine(self,callback=None,clip=None):
        mu = 0
        variance = 1
        limit=4
        times=10
        count=2*limit*times
        sigma = math.sqrt(variance)
        x = np.linspace(mu - limit*sigma, mu + limit*sigma, count)
        y = self.getPdf(x, mu, sigma)
        _, ax = self.getFigAx()
        ax.plot(x, y,c='black',linewidth=.5)
        if isinstance(clip,(list,tuple)):
            x1 = np.linspace(-limit, limit, limit*2+1)
            start=np.where(x1==clip[0])[0][0]*times
            end=np.where(x1==clip[1])[0][0]*times
            ax.fill_between(x[start:end], y[start:end], color='#0080CF', alpha=1)
        else:
            ax.fill_between(x, y, color='#0080CF', alpha=1)
        callable(callback) and callback(ax,plt)
        self.setImgName()
        return self
    def __show(self):
        plt.show()