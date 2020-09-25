import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest
from graphic.plot import Plot
from utilities import getPath,parseNumber
class TDD_TEST_BAYES_THEOREM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class PlotAI(Plot):
            def __init__(self,arg=None):
                Plot.__init__(self)
        cls.d = PlotAI()
    def test_test_Bayes_Theorem(self):
        PA=.01
        PB=.1
        PBA=.9
        PAB=self.d.getBayes(PA,PBA,PB)
        self.assertAlmostEqual(PAB,.09)
        PA=.1
        PB=.4
        PBA=.5
        PAB=self.d.getBayes(PA,PBA,PB,isPercent=True)
        self.assertAlmostEqual(PAB,'12.5%')
        PA=.4
        PB=.25
        PBA=.125
        PAB=self.d.getBayes(PA,PBA,PB)
        self.assertAlmostEqual(PAB,.2)
        done=.8
        undone=.1
        percentAll=.01
        PAB=self.d.getCatAllergy(done,undone,percentAll)
        self.assertAlmostEqual(PAB,.075,3)
        description='''Pam put in 15 paintings, 4% of her works have won First Prize.
Pia put in 5 paintings, 6% of her works have won First Prize.
Pablo put in 10 paintings, 3% of his works have won First Prize.'''
        PAB=self.d.getBayesByMore(description,'Pam')
        print(PAB)
        # self.assertEqual(PAB,.075,3)
if __name__ == '__main__':
    unittest.main()

                