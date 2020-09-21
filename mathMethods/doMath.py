import math
from .doNp import DoNumpy
class DoMath(DoNumpy):
    def getMeanSqr(self, m):
        l = list(m)
        sumVal = sum(l)
        return math.sqrt(sumVal / len(l))