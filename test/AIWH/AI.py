import pandas as pd,numpy as np
from sklearn.utils import shuffle
import urllib.request
class AI(object):
    def __init__(self,d=None):
        if d is None:
            self.topFeaturesNum=0
        else:
            self.topFeaturesNum=d['topFeaturesNum']
    @property
    def shape(self):
        return self.df.shape
    def setCategory(self,category):
        self.category=category
        return self
    def defineObjective(self,objective):
        self.objective=objective
        return self
    def gatherData(self,url=None,skip=False,location=None):
        if not skip:
            if url is None:
                return print('No url supplied')
            else:
                try:
                    print('Beginning file download with urllib2...')
                    urllib.request.urlretrieve(url, location)
                except urllib.error.URLError as e:
                    print(e)
                    if not skip:
                        raise
        return self
    def preprocessLoadData(self,p):
        if not hasattr(self,'df'):
        #Load the data set
            df = pd.read_csv(p)
            self.df=df
        return self
    def preprocessDropCols(self,columns):
        self.df=self.df.drop(columns=columns,axis=1)
        return self
    def preprocessShuffle(self, n):
        self.n=n
        # Create a new data frame with the first n samples
        df_sample = self.df.iloc[:n, :]
        self.shuffle_df = shuffle(df_sample, random_state=42)
        self.df_sample=self.valueCounts('Class',df_sample)
        return self
    def preprocessSplit(self,featuresEnd):
        # Spilt the dataset into train and test data frame
        endInd = int(self.n*.8)
        df_train = self.shuffle_df[0:endInd]
        df_test = self.shuffle_df[endInd:]
        # Spilt each dataframe into feature and label
        self.train_feature = np.array(df_train.values[:, 0:featuresEnd])
        self.train_label = np.array(df_train.values[:, -1])
        self.test_feature = np.array(df_test.values[:, 0:featuresEnd])
        self.test_label = np.array(df_test.values[:, -1])
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