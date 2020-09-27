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
        return self.regrPredict(dtree, condition)
    def predictMultipleRegression(self, predictVals, file=None,isStandard=False):
        if file is None:
            file=self.file
        # Tip: It is common to name the list of independent values with a upper case X, and the list of dependent values with a lower case y.
        X = self.info["x"]
        y = self.info["y"]
        df = self.readCsv(file)
        dfX = df[X]
        dfY = df[y]
        regr, predict = self.predictByRegr(dfX, dfY, predictVals)
        coef_ = list(regr.coef_)
        self.file=file
        if isStandard:
            self.predict = predict
            self.coef_=coef_
        return predict[0], coef_,list(zip(X,predictVals,np.round(regr.coef_,4)))

    def predictByRegr(self, dfX, dfY, predictVals):
        regr = self.getRegr(dfX, dfY)
        predict = self.regrPredict(regr, predictVals)
        return regr, predict

    def getRegr(self, dfX, dfY):
        regr = linear_model.LinearRegression()
        regr.fit(dfX, dfY)
        return regr
    def predictByIncrement(self,increment):
        if isinstance(increment,tuple):
            increment=[increment]
        incre=0
        for item in increment:
            ind=self.info["x"].index(item[0])
            incre+=self.coef_[ind]*item[1]
        return self.predict[0]+incre
    def predictScale(self, file, toTransformVals,predictIndex):
        df = self.readCsv(file)
        X = df[self.info["x"]]
        y = df[self.info["y"]]
        scale = StandardScaler()
        scaledX = scale.fit_transform(X)
        scaled = scale.transform([toTransformVals])
        val = scaled[predictIndex]
        regr, predict = self.predictByRegr(scaledX, y, val)
        return predict[0], list(regr.coef_)

    def regrPredict(self, regr, val):
        if len(np.array(val).shape)==1:
            val=[val]
        return regr.predict(val)
    def predictPolynomialRegression(self, predictX):
        mymodel = self.getPolynomialModel()
        return mymodel(predictX)
    def predict4PolynomialRegression(self, predictX):
        self.degree=4
        return self.predictPolynomialRegression(predictX)
    def predict(self, predictX):
        if not hasattr(self,'info'):
            raise ValueError("self.info without value")
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
    def getRSquared(self, dataType=None):
        x = self.info["x"]
        y = self.info["y"]
        x, y = self.getData(dataType)
        mymodel = self.getPolynomialModel()
        return r2_score(y, mymodel(x))
    def getData(self, dataType="All"):
        x = self.info["x"]
        y = self.info["y"]
        if dataType == "train":
            x = x[:80]
            y = y[:80]
        elif dataType == "test":
            x = x[80:]
            y = y[80:]
        return x, y
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