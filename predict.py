# Machine Learning is a program that analyses data and learns to predict the outcome.
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from do_statistics.doStats import DoStats
from graphic.decision_tree import DecisionTree
from AI import DoAI
class Predict(DoStats,DecisionTree,DoAI):
    def predictbyDecisionTree(self,features,condition, y=None):
        dtree=self.getDtree(features,y)
        return dtree.predict([condition])
    def predictMultipleRegression(self, file, predictVals):
        X = self.info["x"]
        y = self.info["y"]
        df = self.readCsv(file)
        regr = linear_model.LinearRegression()
        regr.fit(df[X], df[y])
        predict = regr.predict([predictVals])
        return predict[0], list(regr.coef_)
    def predictScale(self, file, toTransformVals):
        df = self.readCsv(file)
        X = df[self.info["x"]]
        y = df[self.info["y"]]
        scale = StandardScaler()
        scaledX = scale.fit_transform(X)
        regr = linear_model.LinearRegression()
        regr.fit(scaledX, y)
        scaled = scale.transform([toTransformVals])
        predict = regr.predict([scaled[0]])
        return predict[0], list(regr.coef_)
    def predictPolynomialRegression(self, predictX):
        mymodel = self.getPolynomialModel()
        return mymodel(predictX)
    def predict(self, predictX):
        slope, intercept, r, p, std_err = self.getLinregress()
        return slope * predictX + intercept
    def getModel(self):
        x = self.info["x"]
        y = list(map(self.predict, x))
        return y
