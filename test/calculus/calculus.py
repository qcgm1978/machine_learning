import math
from sympy import *
import random
class Calculus(object):
    def is_read(self):
        return 'the integral from a to b of f-of-x with respect to x'
    def is_infinitesimal(self,sign):
        if sign=='dx' or sign=='dy':
            return True
        else:
            return False
    def get_ratio(self):
        return 'dy/dx'
    def get_limite_shorthand(self):
        return self.get_ratio(),'dy with respect to x'
    def get_replaced_sign(self):
        return ('ξ','Δ')
    def get_Lagrange_notation(self):
        return {'definition':'the derivative of a function called f','denoted':'f′', 'pronounced': "f prime"}
    def get_difference_quotient(self):
        return '(f(a+h) - f(a)) / h'
    def get_squaring_derivative(self,a):
        # \lim_{h \to 0}{f(a+h) - f(a)\over{h}}.
        x = symbols('x')
        derivative=diff(x**2,x)
        return derivative.evalf(subs={x: a})
    def doubling_function(self,a):
        return a*2
    def differentiation_operator(self,f):
        r=random.random()
        if f(r)==r**2:
            return lambda x:x*2
    def get_distance_v_constants(self,v,t):
        return v*t
    def coin_set(self,h,d,truncate):
        tg1 = 0.01746
        power=10**truncate
        return math.floor(h/d/tg1*power)/power