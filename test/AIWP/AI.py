import pandas as pd,numpy as np
from sklearn.utils import shuffle
import urllib.request
class AI(object):
    def __init__(self,d=None):
        if d is None:
            self.topFeaturesNum=0
        else:
            self.topFeaturesNum=d['topFeaturesNum']
        self.categories={
        'Regression': {'type':'Supervised','output':'continuous','aim':'predict','algorithm':'Linear Regression'},
        'Classification':{'type':'Supervised', 'output':'categorical' ,'aim':'category' ,'algorithm':'Logistic Regression'},
        'Clustering':{'type':'Unsupervised','output':'clusters','aim':'group','algorithm':'K-means'}
    }
        self.normalized=False
    @property
    def shape(self):
        return self.df.shape
    @property
    def layers(self):
        return 3
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
    def preprocessLoadData(self,p,isPD=True):
        if not hasattr(self,'df'):
        #Load the data set
            self.df=pd.read_csv(p) if isPD else np.loadtxt(p, delimiter=',')
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
    def preprocessSplit(self,featuresEnd=None):
        if featuresEnd is None:
            featuresEnd=self.input_dim
        # Spilt the dataset into train and test data frame
        if hasattr(self,'n'):
            n = self.n
        else:
            n=self.shape[0]
        endInd = int(n*.8)
        if hasattr(self,'shuffle'):
            shuffle = self.shuffle_df
        else:
            shuffle=self.df
        df_train = shuffle[0:endInd]
        df_test = shuffle[endInd:]
        # Spilt each dataframe into feature and label
        self.train_feature = np.array(df_train.values[:, 0:featuresEnd])
        self.train_label = np.array(df_train.values[:, -1])
        self.test_feature = np.array(df_test.values[:, 0:featuresEnd])
        self.test_label = np.array(df_test.values[:, -1])
        return self
    def debugSave(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("test/AIWP/models/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("test/AIWP/models/model.h5")
        print("Saved model to disk")
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
    def splitIO(self,r):
        # split into input (X) and output (y) variables
        # Purely integer-location based indexing for selection by position.
        self.input = self.df.iloc[:,r]
        self.output = self.df.iloc[:,r.stop]
        self.input_dim=self.input.shape[1]
        return self.input,self.output
    def getInputOutput(self):
        formula='y = f(X)'
        df=self.df
        input=[]
        output=[]
        for i in range (self.shape[1]):
            data=df[:i]
            numbers = [1, 0]
            reports = data.isin(numbers)
            if all(reports):
                output.append(i)
            else:
                input.append(i)
        return {'formula':formula,'input':input,'output':output}
    def getProblemCategory(self):
        return self.categories[self.category]
    def setCategory(self,category):
        self.category=category
        if self.isDL:
            self.categories[category].update({'isDL':True})
        return self
    def setColumnsName(self,l):
        self.columnsName=l
    