from scipy import stats
import math, numpy as np
class DoStats(object):
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
                'ND': 1.5
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
        if len(np.array(l).shape) == 1 or not isEqlProb:
            l = [l]
        ddof=self.getDdof(ddof)
        ret = []
        for val in l:
            if isEqlProb:
                σ = np.std(val,ddof=ddof)
            else:
                σ = self.getSdWithProb(val)
            ret.append(σ)
        return ret if len(ret) > 1 else ret[0]
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
    def getProbability(self,percent=None,isNormal=True):
        if percent is None:
            std = self.getDistance1std()
            percent = round(std, 2)
        if isNormal:
            if percent == 0:
                return 1
            elif percent == 1.00:
                return 0.6827
            elif percent == 1.87:
                return 0.015
            elif percent == 2.00:
                return 0.9545
            elif percent == 3.00:
                return 0.9973
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
        if len(l) == 2:
            v1, v2 = self.getVariance(l)
        return v1 / v2
