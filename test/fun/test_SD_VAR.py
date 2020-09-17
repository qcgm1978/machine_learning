import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest
from mysql_data.decorators_func import singleton
from graphic.plot import Plot
class TDD_TEST_SD_VAR(unittest.TestCase):
    @singleton
    def setUp(self):
        self.__class__.p=Plot()
    def test_test_SD_VAR(self):
        l='600mm, 470mm, 170mm, 430mm and 300mm'
        m=self.p.getMean(l)
        self.assertEqual(m,394)
        self.p.plotGrid(l)
        v=self.p.getVariance(l)
        self.assertEqual(v,21704)
        sd=self.p.getPSD(l)
        self.assertAlmostEqual(sd,147,0)
if __name__ == '__main__':
    unittest.main()
