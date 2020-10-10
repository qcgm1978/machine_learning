from sympy import *
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
        