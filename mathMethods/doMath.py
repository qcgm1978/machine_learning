import math
from .doNp import DoNumpy
class DoMath(DoNumpy):
    def getMeanSqr(self, m):
        l = list(m)
        sumVal = sum(l)
        return math.sqrt(sumVal / len(l))
    def getPolynomialModel(self):
        x = self.info["x"]
        y = self.info["y"]
        mymodel = self.poly1d(x, y)
        return mymodel