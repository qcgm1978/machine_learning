# https://www.mathsisfun.com/data/standard-normal-distribution.html
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import unittest,numpy as np
from graphic.plot import Plot
from do_statistics.doStats import DoStats
from mysql_data.decorators_func import singleton
class TDD_TEST_FUN_ND(unittest.TestCase):
    def test_test_fun_nd(self):
        d=DoStats()
        nd=d.getND(0,1)
        larger=filter(lambda item:item>0,nd)
        smaller=filter(lambda item:item<0,nd)
        self.assertAlmostEqual(len(list(larger)),len(list(smaller)),-3)
        mean,sd=d.getMeanSdByRange([1.1,1.7],.95)
        self.assertEqual(mean,1.4)
        self.assertAlmostEqual(sd,.15,2)
        name=d.getSdName()
        self.assertTrue( "z-score" in name)
        count,differ=d.standardizing(1.85,mean,sd)
        self.assertAlmostEqual(count,3)
        self.assertAlmostEqual(differ,.45)
if __name__ == '__main__':
    unittest.main()

                