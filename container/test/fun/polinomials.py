import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest
from graphic.plot import Plot
from mathMethods.doMath import DoMath
from utilities import getPath,parseNumber
class TDD_POLINOMIALS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class PlotAI(Plot,DoMath):
            def __init__(self,arg=None):
                Plot.__init__(self)
        self.p = PlotAI()
    def test_polinomials(self):
        b=self.p.isPolynomial('4xy**2+3x-5')
        b1=self.p.isPolynomial('1/4xy**2+3x-5')
        b2=self.p.isPolynomial('1/4xyz**2+3x-5')
        b3=self.p.isPolynomial('1/4xyz**2+3x**.5-5')
        b4=self.p.isPolynomial('1/4xyz**2+3x**.5-5x**1/2')
        b5=self.p.isPolynomial('1/4xyz**2+5x**1/2-1')
        b6=self.p.isPolynomial('1/4xyz**2+5x**(1/2)-1')
        b7=self.p.isPolynomial('1/4xyz**2+3x-1/z')
        b8=self.p.isPolynomial('1/4xyz**2+3x**-2-1')
        b9=self.p.isPolynomial('1/4xyz**2+3x/(x-2)-1')
        b10=self.p.isPolynomial('1/4xyz**2+3x/(2-x)-1')
        self.assertTrue(b)
        self.assertTrue(b1)
        self.assertTrue(b2)
        self.assertFalse(b3)
        self.assertFalse(b4)
        self.assertTrue(b5)
        self.assertFalse(b6)
        self.assertFalse(b7)
        self.assertFalse(b8)
        self.assertFalse(b9)
        self.assertFalse(b10)
        self.assertTrue(self.p.isPolynomial('x - 2'))
        self.assertTrue(self.p.isPolynomial(5))
        self.assertFalse(self.p.isPolynomial('3xy**-2'))
        self.assertFalse(self.p.isPolynomial('2/(x+2)'))
        self.assertFalse(self.p.isPolynomial('1/x'))
        self.assertFalse(self.p.isPolynomial('x**(1/2)'))
        self.assertTrue(self.p.isPolynomial('x/2'))
        self.assertTrue(self.p.isPolynomial('3x/8'))
        self.assertTrue(self.p.isPolynomial('2**(1/2)'))
        # self.assertTrue(self.p.isPolynomial('-6y**2 - ( 7/9)x'))
        # s='''3x
        #     x - 2
        #     -6y**2 - ( 7/9)x
        #     3xyz + 3xy2z - 0.1xz - 200y + 0.5
        #     512v5 + 99w5
        #     5
        # '''
        # l=s.split('\n')
        # for item in l:
        #     print(item)
        #     self.assertTrue(self.p.isPolynomial(item))
    def test_getNomialTerms(self):
        self.assertEqual(self.p.getNomialTerms('3xy**2'),1)
        self.assertEqual(self.p.getNomialTerms('5x-1'),2)
        self.assertEqual(self.p.getNomialTerms('3x+5y**2-3'),3)
        self.assertEqual(self.p.getNomialTerms('3x+5y**2-3+4z'),4)
        self.assertEqual(self.p.getNomialTerms('3x+5y**2-3+4z+z**20'),5)
    def test_getVariables(self):
        self.assertEqual(self.p.getVariables(21),0)
        self.assertEqual(self.p.getVariables(' x4 − 2x2 + x '),['x'])
        self.assertEqual(self.p.getVariables('xy4 − 5x2z'),['x','y','z'])
    def test_graph_polynomial(self):
        x=self.p.getLinspaceData(-2,2,40)
        coefs=[1,0,-2,1,0]
        y=self.p.getYByFunc(x,coefs)
        self.p\
            .plotLine(x,y)\
            .pltCartesianCoordinate(hideAxis=True,hasLimit=True)\
            .saveAndShow('drawFunction')
        degree=self.p.getDegree('4x**3-x+3')
        degree1=self.p.getDegree('-x+3')
        degree2=self.p.getDegree('+3')
        self.assertEqual(degree,3)
        self.assertEqual(degree1,1)
        self.assertEqual(degree2,0)
        sf=self.p.getStandardForm('3x**2 − 7 + 4x**3 + x**6')
        self.assertEqual(sf,'x**6 + 4x**3 + 3x**2 − 7')
if __name__ == '__main__':
    unittest.main()

                