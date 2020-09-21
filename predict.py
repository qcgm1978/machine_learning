# Machine Learning is a program that analyses data and learns to predict the outcome.
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from do_statistics.doStats import DoStats
from graphic.decision_tree import DecisionTree
from sklearn.metrics import r2_score
class Predict(DoStats,DecisionTree):
    degree=3
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
    def predict4PolynomialRegression(self, predictX):
        self.degree=4
        return self.predictPolynomialRegression(predictX)
    def predict(self, predictX):
        # slope, intercept, r, p, std_err = self.getLinregress()
        slope, intercept,lineEquation,fitVals,error = self.leastSquaresRegression()
        return slope * predictX + intercept
    def leastSquaresRegression(self,x=None,y=None,roundTo=1):
        if x is None or y is None:
            if hasattr(self,'info'):
                if x is None:
                    x = self.info["x"]
                if y is None:
                    y=self.info["y"]
        xSquares=map(lambda x:x**2,x)
        xy=map(lambda t:t[0]*t[1],zip(x,y))
        sumX=sum(x)
        sumY=sum(y)
        sumSquares=sum(xSquares)
        sumXy=sum(xy)
        N=len(x)
        # m =  (N * Σ(xy) − Σx Σy) / (N * Σ(x2) − (Σx)2)
        m=(N*sumXy-sumX*sumY)/(N*sumSquares-sumX**2)
        # b =  (Σy − m Σx) / N
        b=(sumY-m*sumX)/N
        lineEquation='y = {0}x + {1}'.format(round(m,roundTo),round(b,roundTo))
        fitVals=list(map(lambda x:m*x+b,x))
        error=fitVals-y
        return m,b,lineEquation,fitVals,error
    def getModel(self):
        x = self.info["x"]
        y = list(map(self.predict, x))
        return y
    def getRSquared(self, dataType="All"):
        x = self.info["x"]
        y = self.info["y"]
        x, y = self.getData(dataType)
        mymodel = self.getPolynomialModel()
        return r2_score(y, mymodel(x))
    def getR(self):
        slope, intercept, r, p, std_err = self.getLinregress()
        return r
    
    def getPolynomialModel(self):
        x = self.info["x"]
        y = self.info["y"]
        mymodel = self.poly1d(x, y)
        return mymodel
    def poly1d(self, x, y):
        return np.poly1d(np.polyfit(x, y, self.degree))  