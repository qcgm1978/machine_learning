# pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
import unittest, pandas as pd
import numpy as np
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from mysql_data.decorators_func  import singleton
class TDD_DATAFRAME(unittest.TestCase):
    @singleton
    def setUp(self):
        data = {
            "name": ["Jack", "Frank", "Kelly", "Rebecca", "Monica"],
            "year": [2015, 2011, 2010, 2014, 0],
            "reports": [24, 4, 2, 31, 0],
        }
        self.__class__.df = pd.DataFrame(
            data,
            index=["New York", "New Orleans", "Budapest", "Helsinki", "Cologne"],
        )
        print(self.df)
        # remember that it was setup already
        return super().setUp()
    # pandas.DataFrame.loc
    # property DataFrame.loc
    # Access a group of rows and columns by label(s) or a boolean array.
    def test_DataFrame_loc(self):
        df = pd.DataFrame(
            [[1, 2], [4, 5], [5, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        self.assertEqual(df.shape, (3, 2))
        l = df.loc[df["shield"] > 6]
        self.assertEqual(list(l.values[0]), [5, 8])
        l = df.loc[df["shield"] > 6, ["max_speed"]]
        self.assertEqual(list(l.values[0]), [5])
        l = df[(df["shield"] > 6) & (df["max_speed"] > 6)]
        self.assertEqual(list(l.values), [])
        l = df[(df["shield"] > 6) & (df["max_speed"] < 6)]
        self.assertEqual(list(l.values[0]), [5, 8])
        # http://scrapingauthority.com/pandas-dataframe-filtering/
        # 7 Ways To Filter A Pandas Dataframe
    def test_get_columns(self):
        n = self.df["name"]
        self.assertEqual(list(n), ["Jack", "Frank", "Kelly", "Rebecca", "Monica"])
        n = self.df[["name", "year"]]
        # print(n.values)
        self.assertEqual(
            n.values.tolist(),
            [
                ["Jack", 2015.0],
                ["Frank", 2011.0],
                ["Kelly", 2010.0],
                ["Rebecca", 2014.0],
                ["Monica", 0],
            ],
        )
    def test_filter_rows_where(self):
        df = self.df
        g = df[df["year"] > 2012]
        self.assertEqual(
            g.values.tolist(), [["Jack", 2015, 24.0], ["Rebecca", 2014, 31.0]]
        )
        t = df[(df["year"] > 2012) & (df["reports"] < 30)]
        self.assertEqual(t.values.tolist(), [["Jack", 2015, 24.0]])
    def test_first_last_rows(self):
        df = self.df
        t = df[:2]
        self.assertEqual(list(t), ["name", "year", "reports"])
        self.assertEqual(t.values.tolist(), [["Jack", 2015, 24], ["Frank", 2011, 4]])
        l = df[-1:]
        self.assertEqual(l.values.tolist(), [["Monica", 0, 0]])
        self.assertRaises(KeyError, lambda: df[-1])
    def test_query_string(self):
        df = self.df
        q = df.query('year > 2012 | name == "Frank"')
        self.assertEqual(
            q.values.tolist(),
            [["Jack", 2015, 24], ["Frank", 2011, 4], ["Rebecca", 2014, 31]],
        )
    def test_WhereValueIsInSpecifiedList(self):
        df = self.df
        numbers = [4, 2]
        reports = df["reports"].isin(numbers)
        self.assertEqual(reports.values.tolist(),[False, True, True, False, False])
        r = df[reports]
        self.assertEqual(r.values.tolist(), [["Frank", 2011, 4], ["Kelly", 2010, 2]])
    def test_WhereValueIsNotNaN(self):
        data = {
                "name": ["Rebecca", "Monica"],
                "year": [2014, None],
                "reports": [31, 0],
            }
        df = pd.DataFrame(
                data,
                index=["New York", "New Orleans"]
            )
        n = df[df['year'].notnull()]
        i = df[df['year'].isnull()]
        self.assertEqual(n.values.tolist(),[['Rebecca',2014,31]])
        self.assertTrue(pd.isna(i.values.tolist()[0][1]))
if __name__ == "__main__":
    unittest.main()
