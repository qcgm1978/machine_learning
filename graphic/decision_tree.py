from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
class DecisionTree(object):
    def getDtree(self, features, y=None):
        if y is None:
            y=self.target
        df = self.df
        X = df[features]
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X, df[y])
        return dtree
    def createDecisionTreeData(self, features, y):
        dtree = self.getDtree(features, y)
        self.graphData = tree.export_graphviz(
            dtree, out_file=None, feature_names=features
        )
        return self