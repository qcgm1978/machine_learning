import math,re
from .doNp import DoNumpy
class DoMath(DoNumpy):
#     A polynomial can have:
# constants (like 3, −20, or ½)
# variables (like x and y)
# exponents (like the 2 in y2), but only 0, 1, 2, 3, ... etc are allowed
    def isPolynomial(self,l):
        if isinstance(l,str):
            l=re.split('(?<!\*)[+-]',l)
        l=list(map(lambda item:re.sub(r'\s','',item),l))
        for item in l:
            pConstant=re.compile('[1-9./]+')
            pVariable=re.compile('\w+')
            pExponents=re.compile('\*\*[0-9]+')
            isConstant=pConstant.fullmatch(item)
            isVariables=pVariable.fullmatch(item)
            isExponents=pExponents.fullmatch(item)
            if isConstant or isVariables or isExponents:
                continue
            else:
                l0=re.split('\*\*',item)
                if len(l0)==2:
                    if '.' in l0[1] :
                        return False
                l2=re.split('/',item)
                if len(l2)==2:
                    if re.match(r'^[A-z]',l2[1]):
                        return False
                l1 = re.split('\d|\w|/',item)
                b=all(map(lambda i:i=='' or i=='**',l1))
                if b:
                    continue
                else:
                    return False
        return True
    def getMeanSqr(self, m):
        l = list(m)
        sumVal = sum(l)
        return math.sqrt(sumVal / len(l))