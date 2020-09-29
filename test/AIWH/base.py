import pandas as pd,numpy as np
class Base(object):
    def __init__(self,d=None):
        if d is None:
            self.topFeaturesNum=0
        else:
            self.topFeaturesNum=d['topFeaturesNum']
    @property
    def shape(self):
        return self.df.shape
    def preprocessLoadData(self,p):
        if not hasattr(self,'df'):
        #Load the data set
            df = pd.read_csv(p)
            self.df=df
        return self
    def preprocessDropCols(self,columns):
        self.df=self.df.drop(columns=columns,axis=1)
        return self
    def valueCounts(self,colName,df=None):
        if df is None:
            df = self.df
        return df[colName].value_counts()
    def isBalance(self,l):
        for i in range(len(l)-1):
            if abs(l[i]-l[i+1])>3000:
                return False
        return True