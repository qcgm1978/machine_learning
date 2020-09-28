from datatype import DataTypes
import numpy as np
import unittest
import pandas
class TDD_DECISION_TREE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = "data/shows.csv"
        d1 = {"UK": 0, "USA": 1, "N": 2}
        d2 = {"YES": 1, "NO": 0}
        cls.mapData = (("Nationality", d1), ("Go", d2))
        cls.d = DataTypes({"file": file, ##"mapData": self.mapData, 
        'target': "Go"})
        rep=cls.mapData
        cls.d.preprocessReplace(rep)
        cls.X = ["Age", "Experience", "Rank", "Nationality"]
        cls.y = "Go"
    def test_decision_tree(self):
        file = "data/shows.csv"
        d = DataTypes()
        objective = 'go to a comedy show or not'
        p=d\
            .defineObjective(objective)\
            .gatherData(skip=True)\
            .preprocessLoadData(file)\
            .preprocessReplace(self.mapData
            # .createDecisionTreeData(self.X, self.y)
            )\
            .explorePredictor(self.X, self.y)\
            .buildModel()\
            .ModelEvaluationOptimization()\
            .predictDecisionTree(val=[40, 10, 6, 1])
        self.assertEqual(p,(0, 'NO'))
        features=d.getXcolumns()
        self.assertEqual(features.tolist(),['Age', 'Experience', 'Rank','Nationality'])
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
        self.assertTrue(p[0] in (0,1))
        p = self.d.predictbyDecisionTree(X, [40, 10, 6, 1])
        self.assertEqual(p, (0, 'NO'))
    def test_DecisionTree(self):
        self.d.setAttribute({'X':self.d.df[self.X],'y':self.d.df[self.y]})
        p=self.d\
            .buildModel()\
            .ModelEvaluationOptimization()\
            .predictDecisionTree(val=[40, 10, 6, 1])
        self.assertEqual(p,(0, 'NO'))
if __name__ == "__main__":
    unittest.main()
