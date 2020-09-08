# Machine Learning is making the computer learn from studying data and statistics.
import pandas, pydotplus, math, numpy as np, matplotlib.image as pltimg, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mathMethods.doMath import DoMath
from do_statistics.doStats import DoStats
from mysql_data.mysqlOp import MysqlOp
class HandleData(DoMath, DoStats, MysqlOp):
    def __init__(self, n=None):
        if isinstance(n, dict):
            unique = n["unique"] if "unique" in n else None
            sqlData = n["sqlData"] if "sqlData" in n else None
            MysqlOp.__init__(
                self, "data", sqlData, db="machine_learning", unique=unique
            )
            if "sqlData" in n:
                n = sqlData[0]
            self.info = n
            listProp = list(
                filter(
                    lambda key: isinstance(n[key], (list, np.ndarray)) and key, n.keys()
                )
            )
            if len(listProp):
                self.prop = listProp[0]
                self.list = self[self.prop]
                self.len = len(self.list)
            if "target" in n:
                self.target = n["target"]
            if "file" in n:
                self.df = self.readCsv(n["file"])
                if "mapData" in n:
                    self.df = self.mapStrToNum(n["mapData"])
        else:
            self.n = n
    def __getitem__(self, i):
        try:
            return self.info[i]
        except KeyError:
            return None
    def getAndFormatData(self, file, dictionary):
        df = self.readCsv(file)
        self.mapStrToNum(dictionary, df)
        return self
    def graphByData(self, img):
        graph = pydotplus.graph_from_dot_data(self.graphData)
        graph.write_png(img)
        img = pltimg.imread(img)
        imgplot = plt.imshow(img)
        return self
    def scale(self, file, scaleCols):
        scale = StandardScaler()
        df = self.readCsv(file)
        X = df[scaleCols]
        scaledX = scale.fit_transform(X)
        return scaledX
    def readCsv(self, file):
        self.df = pandas.read_csv(file)
        return self.df
    def convertSeriesToList(self, p):
        return p.values.tolist()
    def getDfCol(self, x=None):
        if x is None:
            x = self.info["x"][1]
        return self.df[x]
    def queryDf(self, s, colName=None):
        if colName is None and self.info['x']:
            colName=self.info['x'][1]
        data = self.df.query(s)
        return data[colName] if colName else data
    def mapStrToNum(self, dictionary, df=None):
        if df is None:
            df = self.df
        for field, v in dictionary.items():
            df[field] = df[field].map(v)
        self.df = df
        return df
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
    def Numerical(self):
        return self.Discrete() or self.Continuous()
    def Discrete(self):
        return isinstance(self.n, int)
    def Continuous(self):
        return isinstance(self.n, float)
    def Categorical(self):
        return "color"
    def Ordinal(self):
        return "school grades"
