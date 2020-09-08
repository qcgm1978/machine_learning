from datatype import DataTypes
import numpy as np
import unittest
import pandas


class TDD_DECISION_TREE(unittest.TestCase):
    def setUp(self):
        file = "shows.csv"
        d1 = {"UK": 0, "USA": 1, "N": 2}
        d2 = {"YES": 1, "NO": 0}
        mapData = {"Nationality": d1, "Go": d2}
        self.d = DataTypes({"file": file, "mapData": mapData, 'target': "Go"})
        return super().setUp()

    def test_decision_tree(self):
        file = "shows.csv"
        img = "mydecisiontree.png"
        X = ["Age", "Experience", "Rank", "Nationality"]
        y = "Go"
        d1 = {"UK": 0, "USA": 1, "N": 2}
        d2 = {"YES": 1, "NO": 0}
        dictionary = {"Nationality": d1, "Go": d2}
        d = DataTypes()
        d\
            .getAndFormatData(file, dictionary)\
            .createDecisionTreeData(X, y)\
            .graphByData(img)\
            .show()

    def test_first_step(self):
        df = self.d.df
        y = df["Go"]
        self.assertIsInstance(y, pandas.core.series.Series)
        Gini, n, value = self.d.getGini()
        self.assertAlmostEqual(Gini, .497, 3)
        self.assertEqual(n, 13)
        self.assertEqual(value, [6, 7])
        # print(dfZero)

    def test_second_true_step(self):
        gini, n, value = self.d.getGini(
            getSample=lambda df: df[df["Rank"] <= 6.5])
        self.assertEqual(gini, 0)
        self.assertEqual(n, 5)
        self.assertEqual(value, [5, 0])

    def test_second_false_step(self):
        def getSample(df): return df[df["Rank"] > 6.5]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertAlmostEqual(Gini, .219, 3)
        self.assertEqual(n, 8)
        self.assertEqual(value, [1, 7])

    def test_third_true_step(self):
        def getSample(df): return df[(
            df['Nationality'] <= .5) & (df['Rank'] > 6.5)]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertAlmostEqual(Gini, .375, 3)
        self.assertEqual(n, 4)
        self.assertEqual(value, [1, 3])

    def test_third_false_step(self):
        def getSample(df): return df[(
            df['Nationality'] > .5) & (df['Rank'] > 6.5)]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertEqual(Gini, 0)
        self.assertEqual(n, 4)
        self.assertEqual(value, [0, 4])

    def test_fourth_true_step(self):
        def getSample(df): return df[(df['Age'] <= 35.5) & (
            df['Nationality'] <= .5) & (df['Rank'] > 6.5)]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertEqual(Gini, 0)
        self.assertEqual(n, 2)
        self.assertEqual(value, [0, 2])

    def test_fourth_false_step(self):
        def getSample(df): return df[(df['Age'] > 35.5) & (
            df['Nationality'] <= .5) & (df['Rank'] > 6.5)]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertEqual(Gini, .5)
        self.assertEqual(n, 2)
        self.assertEqual(value, [1, 1])

    def test_fifth_true_step(self):
        def getSample(df): return df[(df['Experience'] <= 9.5) & (
            df['Age'] > 35.5) & (df['Nationality'] <= .5) & (df['Rank'] > 6.5)]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertEqual(Gini, 0)
        self.assertEqual(n, 1)
        self.assertEqual(value, [0, 1])

    def test_fifth_false_step(self):
        def getSample(df): return df[(df['Experience'] > 9.5) & (
            df['Age'] > 35.5) & (df['Nationality'] <= .5) & (df['Rank'] > 6.5)]
        Gini, n, value = self.d.getGini(getSample=getSample)
        self.assertEqual(Gini, 0)
        self.assertEqual(n, 1)
        self.assertEqual(value, [1, 0])

    def test_predict_decision_tree(self):
        X = ["Age", "Experience", "Rank", "Nationality"]
        p = self.d.predictbyDecisionTree(X, [40, 10, 7, 1])
        # The Decision Tree does not give us a 100% certain answer. It is based on the probability of an outcome, and the answer will vary.
        self.assertTrue(list(p)[0] in [0, 1])
        p = self.d.predictbyDecisionTree(X, [40, 10, 6, 1])
        self.assertTrue(list(p)[0] in [0, 1])


if __name__ == "__main__":
    unittest.main()
