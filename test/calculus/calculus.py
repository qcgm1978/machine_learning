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
        return self.get_ratio(),'dy with respect to x','dy by dx','dy over dx'
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
    def coin_set(self,h,d,truncate,prob_truncate):
        tg1 = 0.01746
        power=10**truncate
        power1=10**prob_truncate
        degree = math.floor(h/d/tg1*power)/power
        prob = math.floor(degree*2/180*power1)/power1
        return degree,prob
    def read_partial_derivative(self):
        return 'the partial derivative of z with respect to x','∂z/∂x'
    def find_point_slope(self,point,derivative):
        return self.read_partial_derivative()[0]+' at ({0},{1}) is {2}'.format(point[0],point[1],derivative)
    def derivative_every_point(self):
        return 'f′',('the derivative function','the derivative of f')
    def get_expr_derivate(self,difference,xi):
        cancellation=difference[:-2]
        derivate=sympify(cancellation)
        return derivate.evalf(subs={'xi':xi,'h':0})
    def get_change_rate(self, seconds):
        # (f \circ g)'(t) = f'(g(t))\cdot g'(t).
        t, e = symbols('t e')
        g_t='1/2*g*t**2'#.5*g*((t+h)**2-t**2)/h, .5*g*2t=g*t
        h='4000-{0}'.format(g_t)
        f_prime_h = '-10.1325*e**(-0.0001*({0}))'.format(h)
        g_t_prime='g*t'
        f_g_prime_t = '{0}*(-{1})'.format(f_prime_h, g_t_prime)
        print(f_g_prime_t)
        change_rate = sympify(f_g_prime_t).evalf(subs={'t': seconds, 'e': math.e,'g':9.8})
        return change_rate