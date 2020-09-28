# Machine Learning is making the computer learn from studying data and statistics.
import logging,pandas as pd, pydotplus, math, numpy as np, matplotlib.image as pltimg, matplotlib.pyplot as plt
from sklearn import tree
from pint import UnitRegistry
from sklearn.preprocessing import StandardScaler
from mathMethods.doMath import DoMath
from mysql_data.mysqlOp import MysqlOp
from scipy import  constants
class HandleData(DoMath, MysqlOp):
    def __init__(self, n=None):
        if isinstance(n, dict):
            unique = n["unique"] if "unique" in n else None
            if "sqlData" in n:
                sqlData = n["sqlData"] 
                MysqlOp.__init__(
                    self, "data", sqlData, db="machine_learning", unique=unique
                )
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
        except (KeyError,AttributeError):
            return None
    def setAttribute(self,d):
        for p,v in d.items():
            setattr(self,p,v)
    def getSortedXyInt(self,t1,t2):
        rs = np.random.seed(42)
        x=np.random.randint(*t1)
        y=np.random.randint(*t2)
        x.sort()
        y.sort()
        return x, y
    def getConstants(self):
        return constants
    def to(self, unit, toUnit, num=1):
        ureg = UnitRegistry()
        current=num*ureg[unit]
        return current.to(ureg[toUnit])
    def getAndFormatData(self,  dictionary):
        # self.preprocessLoadData(file)
        self.preprocessReplace(dictionary)
        # self.mapStrToNum(dictionary)
        return self
    def graphByData(self, img = "img/mydecisiontree.png"):
        if not isinstance(self.features,list):
            return self
        if not hasattr(self,'graphData'):
            self.graphData = tree.export_graphviz(
            self.clf_dt, out_file=None, feature_names=self.features
        )
        graph = pydotplus.graph_from_dot_data(self.graphData)
        graph.write_png(img)
        img = pltimg.imread(img)
        imgplot = plt.imshow(img)
        return self
    def scale(self, file, ScaleCols=None):
        if ScaleCols is None:
            ScaleCols=self.info['x']
        scale = StandardScaler()
        df = self.readCsv(file)
        X = df[ScaleCols]
        scaledX = scale.fit_transform(X)
        return scaledX
    def preprocessLoadData(self,p):
        if not hasattr(self,'df'):
        #Load the data set
            df = pd.read_csv(p)
            self.df=df
        return self
    def readCsv(self, file):
        self.df = pd.read_csv(file)
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
    def preprocessReplace(self,l):
        for item in l:
            self.df[item[0]].replace(item[1],inplace = True)
        return self
    def getXcolumns(self):
        return self.X.columns[self.selector.get_support(indices=True)]
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
