import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datatype import DataTypes
import math, unittest, numpy as np
from mysql_data.decorators_func import singleton
from do_statistics.physical_constants import constantsFren, total
class TDD_TEST_DATA(unittest.TestCase):
    @singleton
    def setUp(self):
        sqlData = [{"speed": [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]}]
        self.__class__.d = DataTypes(
            {"sqlData": sqlData, "unique": ["address", "speed"]}
        )
        return super().setUp()
    def test_mean(self):
        m = self.d.getMean()
        self.assertAlmostEqual(m, 89.77, 2)
        middle = self.d.getMedian()
        self.assertAlmostEqual(middle, 87, 0, 1)
        l = [99, 86, 87, 88, 86, 103, 87, 94, 78, 77, 85, 86]
        count = self.d.updateField({"field": "speed", "to": l})
        self.assertEqual(count, 1)
        mode = self.d.getMode()[0]
        self.assertEqual(mode, 86)
    def test_std(self):
        l = [86, 87, 88, 86, 87, 85, 86]
        count = self.d.updateField({"field": "speed", "to": l})
        self.assertEqual(count, 1)
        sd = self.d.getPSD()
        self.assertAlmostEqual(sd, 0.9, 1)
        self.assertTrue(self.d.isNormalSD())
        self.assertAlmostEqual(self.d.getMean(), 86.4, 1)
        v = self.d.getVariance()
        self.assertAlmostEqual(sd,math.sqrt(v),1)
    def test_Benford(self):
        b = self.d.getBenfordLaw(1)
        self.assertAlmostEqual(b, 0.30, 2)
        self.assertEqual(total, 442)
        # expected number of each leading digit per Benford's law
        r = range(1, 10)
        ben = [round(total * self.d.getBenfordLaw(i)) for i in r]
        print(ben, "\n", constantsFren)
        # self.d.plotBar(ben,r)
        # self.d.plotBar(constantsFren,r)
        self.d.plotGroupedBar(
            l1=constantsFren,
            l2=ben,
        )
if __name__ == "__main__":
    unittest.main()
