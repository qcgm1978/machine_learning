import unittest
# from utilities import getPath,parseNumber,update_json
from .calculus import Calculus
class TDD_CALCULUS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=Calculus()
    def test_calculus(self):
        c=self.c
        self.assertEqual(c.is_read(),'the integral from a to b of f-of-x with respect to x')
        self.assertTrue(c.is_infinitesimal('dy'))
        self.assertEqual(c.get_ratio(),'dy/dx')
        self.assertEqual(c.get_replaced_sign(),('ξ','Δ'))
        self.assertEqual(c.get_Lagrange_notation()['pronounced'],"f prime")
        self.assertEqual(c.get_difference_quotient(),'(f(a+h) - f(a)) / h')
        input = 3
        derivative = 6
        self.assertEqual(c.get_squaring_derivative(input),derivative)
        self.assertEqual(c.doubling_function(input),derivative)
        double_f=c.differentiation_operator(lambda x:x**2)
        self.assertEqual(double_f(input),derivative) 
        self.assertEqual(c.get_distance_v_constants(50,3),150)
    def test_geometry_probability(self):
        c=self.c
        self.assertEqual(c.coin_set(1.85,25,2),4.23)
if __name__ == '__main__':
    unittest.main()
                