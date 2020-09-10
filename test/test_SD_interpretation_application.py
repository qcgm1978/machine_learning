import sys, os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from mysql_data.decorators_func import singleton
from datatype import DataTypes
import unittest
class TDD_TEST_SD_INTERPRETATION_APPLICATION(unittest.TestCase):
    @singleton
    def setUp(self):
        self.__class__.d = DataTypes()
        self.__class__.s1 = {0, 0, 14, 14}
        self.__class__.s2 = {0, 6, 8, 14}
        self.__class__.s3 = {6, 6, 8, 8}
        self.__class__.s4= {1000, 1006, 1008, 1014}
    def test_test_SD_interpretation_application(self):
        m1 = self.d.getMean(self.s1)
        m2 = self.d.getMean(self.s2)
        m3 = self.d.getMean(self.s3)
        m4 = self.d.getMean(self.s4)
        self.assertEqual(m1,7)
        self.assertEqual(m2,7)
        self.assertEqual(m3, 7)
        self.assertEqual(m4, 1007)
        sd1 = self.d.getPSD(self.s1)
        sd2 = self.d.getPSD(self.s2)
        sd3 = self.d.getPSD(self.s3)
        sd4 = self.d.getPSD(self.s4)
        self.assertAlmostEqual(sd1,7,0)
        self.assertAlmostEqual(sd2,5,0)
        self.assertAlmostEqual(sd3,1,0)
        self.assertAlmostEqual(sd4, 5, 0)
    def test_plot(self):
        l1=self.d.getND(100,10,size=10000)
        l2=self.d.getND(100,50,size=10000)
        self.d.plotND(l=[l1,l2],bars=100,labels=[10,50])
if __name__ == '__main__':
    unittest.main()