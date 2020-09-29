import pandas as pd,numpy as np
from sklearn.utils import shuffle
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
    def preprocessShuffle(self, df_sample):
        self.shuffle_df = shuffle(df_sample, random_state=42)
        return self
    def preprocessSplit(self):
        # Spilt the dataset into train and test data frame
        df_train = self.shuffle_df[0:2400]
        df_test = self.shuffle_df[2400:]
        # Spilt each dataframe into feature and label
        self.train_feature = np.array(df_train.values[:, 0:29])
        self.train_label = np.array(df_train.values[:, -1])
        self.test_feature = np.array(df_test.values[:, 0:29])
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