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
        return np.mean(l)
    def getMedian(self):
        # speed = self['speed'].copy()
        # speed.sort()
        # return speed[len(speed)//2]
        return np.median(self.list)
    #  by convention, only effects more than two standard deviations away from a null expectation are considered statistically significant, by which normal random error or variation in the measurements is in this way distinguished from likely genuine effects or associations.
    def isNormalSD(self, l=None):
        sd = self.getSD(l)
        return sd < 2
    # In the population standard deviation formula, the denominator is N instead of N − 1.
    def getPSD(self,l=None):
        return self.getSD(l,ddof=0)
    # Standard deviation may be abbreviated SD,
    def getSD(self, l=None,ddof=1):
        """
        :return: Standard deviation may be abbreviated SD, and is most commonly represented in mathematical texts and equations by the lower case Greek letter sigma σ
        :rtype: list
        """
        if l is None:
            l = self.list
        if len(np.array(l).shape) == 1:
            l = [l]
        σ = []
        for val in l:
            """Algorithm
            """
            # m = self.getMean(val)
            # l1 = map(lambda item: (item - m) ** 2, val)
            # sqaredSum = sum(l1)
            # denominator = (len(val) - ddof)
            # s = math.sqrt(sqaredSum / denominator)
            s = np.std(val,ddof=ddof)
            σ.append(s)
        return σ if len(σ) > 1 else σ[0]
    def get1stdProbability(self):
        mean = self.getMean()
        minusSquare = map(lambda x: (x - mean) ** 2, self.list)
        probability = self.getMeanSqr(minusSquare)
        return probability
    def getDistance1std(self,expect=None):
        if expect is None:
            expect = self["expectation"]
        if expect:
            mean = self.getMean()
            unitStd = self.get1stdProbability()
            difference = expect - mean
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
    def getProbability(self):
        std = self.getDistance1std()
        std2decimal = round(std, 2)
        if std2decimal == 1.00:
            return 0.683
        elif std2decimal == 1.87:
            return 0.015
        elif std2decimal == 2.00:
            return 0.954
    def getVariance(self, l=None):
        if l is None:
            l = [self.list]
        ret = []
        for val in l:
            """Algorithm
            mean = self.getMean()
            difference = map(lambda x: x - mean, self.list)
            square = map(lambda x: x ** 2, difference)
            squareList = list(square)
            variance = sum(squareList) / len(squareList)
            or the square root of its variance
            s=self.getSD([val])**2
            """
            s = np.var(val)
            ret.append(s)
        return ret if len(ret) > 1 else ret[0]
    def compareByVariance(self, l):
        # If the two variances are not significantly different, then their ratio will be close to 1.
        if len(l) == 2:
            v1, v2 = self.getVariance(l)
        return v1 / v2
