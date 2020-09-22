import math,re
from .doNp import DoNumpy
class DoMath(DoNumpy):
    def getVariables(self,l):
        if isinstance(l,(int,float)):
            return 0
        a=re.findall('[A-z]',l)
        single=list(dict.fromkeys(a))
        return single
    # Monomial, Binomial, Trinomial,quadrinomial,quintinomial
    def getNomialTerms(self,l):
        l=re.split('(?<!\*)[+-]',l)
        return len(l)
    def getStandardForm(self,s):
        l=re.findall(r"(?<=\*\*)\d+", s)
        intL=list(map(lambda item:int(item),l))
        intL.sort(reverse=True)
        ret=''
        for item in intL:
            search=re.search(r'[-+]?(\s*)?(\d+)?x\*\*{0}'.format(item),s)
            g = search.group()
            ret+=' '
            if g.startswith('+') or g.startswith('-'):
                ret+=g
            else:
                ret+='+ '+g
        constantTerm=re.search(r'(?<!\*)[−+]?(\s+)?\d+(?!x)','3x**2 − 7 + 4x**3 + x**6')
        polyS = ret+' '+constantTerm.group()
        return re.sub(r'^\s*\+\s*','',polyS)
    def getDegree(self,s):
        try:
            float(s)
            return 0
        except ValueError:
            l=re.findall(r"(?<=\*\*)\d+", s)
            nums=list(map(lambda item:int(item),l))
            return max(nums) if len(nums) else 1
#     A polynomial can have:
# constants (like 3, −20, or ½)
# variables (like x and y)
# exponents (like the 2 in y2), but only 0, 1, 2, 3, ... etc are allowed
    def isPolynomial(self,l):
        if isinstance(l,str):
            l=re.split('(?<!\*)[+-]',l)
        elif isinstance(l,(int,float)):
            return True
        l=list(map(lambda item:re.sub(r'\s','',item),l))
        for item in l:
            pConstant=re.compile('[1-9./*()]+')
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