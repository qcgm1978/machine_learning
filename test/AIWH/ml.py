# For linear algebra
import numpy as np
# For data processing
import pandas as pd,re
from scipy import stats
from sklearn import preprocessing
#Using SelectKBest to get the top features!
from sklearn.feature_selection import SelectKBest, chi2
class Panda(object):
    def DataFrame(self,s,cols):
        l=re.split(r'\n',s)
        rows=list(map(lambda item:re.split(r'\s+',item.lstrip()),l))
        columns=re.split(r'\s+',cols)
        # print(rows,columns)
        df = pd.DataFrame(np.array(rows),
            columns=columns     
        )
        return df
    def to_csv(self,df,p):
        df.to_csv(p,index=False)
    def loadDataSet(self,p):
        if not hasattr(self,'df'):
        #Load the data set
            df = pd.read_csv(p)
            self.df=df
        return self
    def removeOutliers(self):
        z = np.abs(stats.zscore(self.df._get_numeric_data()))
        self.df= self.df[(z < 3).all(axis=1)]
        return self
    def replace(self):
        #Change yes and no to 1 and 0 respectvely for RainToday and RainTomorrow variable
        self.df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
        self.df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
        return self
    def normalize(self):
        scaler = preprocessing.MinMaxScaler()
        # print(df[1:2].to_string(header=True))
        scaler.fit(self.df)
        self.df = pd.DataFrame(scaler.transform(self.df), index=self.df.index, columns=self.df.columns)
        self.df.iloc[4:10]
        return self
    def preprocessingData(self,columns):
        self.df=self.df.drop(columns=columns,axis=1)
        #Removing null values
        self.df = self.df.dropna(how='any')
        self.removeOutliers().replace().normalize()
        return self
    def explore(self):
        df=self.df
        X = df.loc[:,df.columns!='RainTomorrow']
        y = df[['RainTomorrow']]
        selector = SelectKBest(chi2, k=3)
        selector.fit(X, y)
        X_new = selector.transform(X)
        print(X.columns[selector.get_support(indices=True)])
        return self
