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
        # slope, intercept, r, p, std_err = self.getLinregress()
        slope, intercept,lineEquation = self.leastSquaresRegression()
        return slope * predictX + intercept
    def leastSquaresRegression(self,x=None,y=None):
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
        lineEquation='y={0}x+{1}'.format(round(m,1),round(b,1))
        return m,b,lineEquation
    def getModel(self):
        x = self.info["x"]
        y = list(map(self.predict, x))
        return y
