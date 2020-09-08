# Machine Learning is a step into the direction of artificial intelligence (AI).
from sklearn.metrics import r2_score
from do_statistics.doStats import DoStats
from mathMethods.doMath import DoMath
class DoAI(DoStats):
    def getRSquared(self, dataType="All"):
        x = self.info["x"]
        y = self.info["y"]
        x, y = self.getData(dataType)
        mymodel = self.getPolynomialModel()
        return r2_score(y, mymodel(x))
    def getR(self):
        slope, intercept, r, p, std_err = self.getLinregress()
        return r