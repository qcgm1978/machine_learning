from predict import Predict
from data.handle_data import HandleData
from graphic.plot import Plot
from graphic.decision_tree import DecisionTree
class DataTypes(HandleData, Predict, Plot, DecisionTree):
    def __init__(self,*arg):
        HandleData.__init__(self,*arg)
        Plot.__init__(self)
    def getGini(self, getSample=None):
        if callable(getSample):
            samples = getSample(self.df)
        else:
            samples = self.df
            # Gini = 1 - (x/n)2 - (y/n)2
        # Where x is the number of positive answers("GO"), n is the number of samples, and y is the number of negative answers ("NO"), which gives us this calculation:
        n = samples.shape[0]
        x = samples.loc[samples[self.target] == 1].shape[0]
        y = n - x
        Gini = 1 - (x / n) ** 2 - (y / n) ** 2
        return Gini, n, [y, x]
    