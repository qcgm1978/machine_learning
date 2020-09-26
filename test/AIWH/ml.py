# For linear algebra
import numpy as np
# For data processing
import pandas as pd,re
from scipy import stats
from sklearn import preprocessing

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
    def read_csv(self,p):
        #Load the data set
        df = pd.read_csv(p)
        self.df=df
        return self.df
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
        df=self.df
        scaler = preprocessing.MinMaxScaler()
        # print(df[1:2].to_string(header=True))
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
        df.iloc[4:10]
        return self
    def preprocessingData(self,columns):
        self.df=self.df.drop(columns=columns,axis=1)
        #Removing null values
        self.df = self.df.dropna(how='any')
        self.removeOutliers()
        return self
 