from numpy import random
from scipy import stats
from scipy.stats import ttest_ind_from_stats
import math,warnings, numpy as np
class DoStats(object):
    
    def getEvolutiveData(self,l):
        sortedX = sorted(l)
        plt,ax,x,y=self.getXyData(x=l)
        plt,ax,x1,y1=self.getXyData(x=sortedX)
        t = ( 
            {'x':range(len(l)),'y':l,'title':'Values and its Index','isScatter':True}, 
            {'x':l,'y':y,'title':'Not Standardizing unsorted Line chart'}, 
            {'x':l,'y':y,'title':'Not Standardizing unsorted Scatter chart','isScatter':True}, 
            {'x':sortedX,'y':y1,'title':'Not Standardizing sorted'}, 
            {'x':x1,'y':y1,'title':'Standardizing sorted'}, 
        )
        return t
    def getSdName(self):
        l=['standard deviations', "Standard Score", "sigma" , "z-score"]
        return l
    def getMeanSdByRange(self,range,percent):
        half=abs(range[0]-range[1])/2
        mean=half+range[0]
        sdCount=self.getSdCountByPercent(percent)
        sd=half/sdCount
        return mean,sd
    # to convert a value to a Standard Score ("z-score"):
    # first subtract the mean,
    # then divide by the Standard Deviation
    # And doing that is called "Standardizing":
    def standardizing(self,val,μ,σ):
        differ=val-μ
        zScore=differ/σ
        return zScore,differ
    def getND(self, mean, SD,size=1000):
        mean=self.getMean(mean)
        x = np.random.normal(loc=mean, scale=SD, size=size)
        return x
    # μ is the expected value of the random variables, σ equals their distribution's standard deviation divided by n**(1/2), and n is the number of random variables
    def getProbabilityDensity(self,x,σ=None,μ=None):
        if μ is None:
            μ=self.getMean(x)
        if σ is None:
            σ = self.getPSD(x)
        zScore,differ=self.standardizing(x,self.getMean(μ),σ)
        μ=0
        y=((1 / (np.sqrt(2 * np.pi) * σ)) *
                np.exp(-0.5 * (1 / σ * (zScore - μ))**2))
        return zScore,y
    def getTtest(self):
        return ttest_ind_from_stats(mean1=15.0, std1=np.sqrt(87.5),
                     mean2=12.0, std2=np.sqrt(39.0) )
    def getDistanceByPopulationPercent(self,percent,σ=1):
        return σ/math.sqrt(1-percent)
    def getMode(self):
        return stats.mode(self.list)
    def getLinregress(self, x=None):
        if x is None:
            x = self.info["x"]
        return stats.linregress(x, self.info["y"])
    def getBenfordLaw(self, n=1):
        return math.log10((n + 1) / n)
    def getMean(self, l=None):
        # return sum(self['speed'])/len(self['speed'])
        if l is None:
            l = self.list
        if isinstance(l,(int,float)):
            return l
        if isinstance(l, set):
            l=list(l)
        #μ: the population mean or expected value in probability and statistics
        if len(np.array(l).shape) == 1:
            μ=np.mean(l)
        else:
            mulProb = map(lambda item: item[0] * item[1], l)
            μ=sum(mulProb)
        return μ
    def getMedian(self):
        # speed = self['speed'].copy()
        # speed.sort()
        # return speed[len(speed)//2]
        return np.median(self.list)
    #  by convention, only effects more than two standard deviations away from a null expectation are considered statistically significant, by which normal random error or variation in the measurements is in this way distinguished from likely genuine effects or associations.
    def isNormalSD(self, l=None):
        α = self.getSD(l)
        return α < 2
    # A five-sigma level translates to one chance in 3.5 million that a random fluctuation would yield the result
    def getParticalPhyStd(self):
        return {'σ': 5, 'randomCount':3.5e6}
    # In the population standard deviation formula, the denominator is N instead of N − 1.
    def getPSD(self,l=None):
        return self.getSD(l, ddof=0)
    def getSdWithProb(self,val):
            μ = self.getMean(val)
            l1 = map(lambda item: item[1]*(item[0] - μ) ** 2, val)
            summation = sum(l1)
            σ = math.sqrt(summation )
            return σ
    def getDdof(self, ddof):
        if isinstance(ddof, int):
            return ddof
        elif isinstance(ddof, str):
            #  corrected sample standard deviation (using N − 1), and this is often referred to as the "sample standard deviation"
            # the uncorrected estimator (using N) yields lower mean squared error
            # N − 1.5 (for the normal distribution) almost completely eliminates bias
            d = {
                'CSSD': 1,
                'LMSE': 0,
                'ND': 1.5,
                # A more accurate approximation is to replace N-1.5 with N-1.5+1/(8(N-1)), e.g. N-(1.5-1/(8(N-1)))
                'ACCURATE':lambda N:1.5-1/(8*(N-1))
            }
            return d.get(ddof,None)
    # Standard deviation may be abbreviated SD,
    def getSD(self, l=None,ddof=1,expected='μ',isEqlProb=True):
        """
        :return: Not all random variables have a standard deviation, since these expected values need not exist.
        Standard deviation may be abbreviated SD, and is most commonly represented in mathematical texts and equations by the lower case Greek letter sigma σ
        :rtype: list
        """
        if expected != 'μ':
            return None
        if l is None:
            l = self.list
        if len(np.array(l).shape) == 1 or not isEqlProb or isinstance(l,set):
            l = [l]
        ddof=self.getDdof(ddof)
        ret = []
        for val in l:
            if isinstance(val, set):
                val=list(val)
            if isEqlProb:
                length=len(val)
                # for N>75 the bias is below 1%
                # if ddof == 0 and length <= 75:
                #     warnings.warn('Uncorrected sample standard deviation, the bias is most significant for small or moderate sample sizes')
                if callable(ddof):
                    ddof=ddof(length)
                σ = np.std(val,ddof=ddof)
            else:
                σ = self.getSdWithProb(val)
            ret.append(σ)
        return ret if len(ret) > 1 else ret[0]
    def getFreedomDegrees(self, l=None):
        if l is None:
            l = self.list
        return len(l)-1
    def get1stdProbability(self):
        μ = self.getMean()
        minusSquare = map(lambda x: (x - μ) ** 2, self.list)
        probability = self.getMeanSqr(minusSquare)
        return probability
    def getDistance1std(self,expect=None):
        if expect is None:
            expect = self["expectation"]
        if expect:
            μ = self.getMean()
            unitStd = self.get1stdProbability()
            difference = expect - μ
            differenceStd = difference / unitStd
            return differenceStd
        else:
            return self.getPSD()
    def getPercentile(self, percent):
        # listP = self.list.copy()
        # listP.sort()
        # lessIndex=round(self.len*percent)
        # val = listP[lessIndex-1]
        # return val
        return np.percentile(self.list, percent * 100)
    # In statistics, the 68–95–99.7 rule, also known as the empirical rule, is a shorthand used to remember the percentage of values that lie within a band around the mean in a normal distribution with a width of two, four and six standard deviations, respectively; more precisely, 68.27%, 95.45% and 99.73% of the values lie within one, two and three standard deviations of the mean, respectively.
    def getProbability(self, σ=None, isNormal=True, σRange=None):
        if σRange is None:
            σRange=[]
        if σ is None:
            if len(σRange):
                σ=σRange[0]
            else:
                std = self.getDistance1std()
                σ = round(std, 2)
        if isNormal:
            if σ == 0:
                ret= 0
            elif σ == 1.00:
                ret= 0.6827
            elif σ == 1.87:
                ret= 0.015
            elif σ == 2.00:
                ret= 0.9545
            elif σ == 3.00:
                ret = 0.9973
            else:
                ret=1
            return ret - self.getProbability(σRange[1]) if len(σRange)==2 else ret
    def getSdCountByPercent(self,percent):
        if percent == 0:
            σ= None
        elif abs(percent - 0.6827)<.01:
            σ= 1
        elif abs(percent - 0.015)<.01:
            σ= 1.87
        elif abs(percent - 0.9545)<.01:
            σ= 2
        elif abs(percent - 0.9973)<.01:
            σ = 3
        else:
            σ=0
        return σ
    def getVariance(self, l=None):
        if l is None:
            l = [self.list]
        ret = []
        for val in l:
            """Algorithm
            μ = self.getMean()
            difference = map(lambda x: x - μ, self.list)
            square = map(lambda x: x ** 2, difference)
            squareList = list(square)
            variance = sum(squareList) / len(squareList)
            or the square root of its variance
            σ=self.getSD([val])**2
            """
            v = np.var(val)
            ret.append(v)
        return ret if len(ret) > 1 else ret[0]
    def compareByVariance(self, l):
        # If the two variances are not significantly different, then their ratio will be close to 1.
        l1=self.getVariance(l)
        if len(l) == 2:
            v1, v2 = l1
            return v1/v2
        else:
            return l1[0]
