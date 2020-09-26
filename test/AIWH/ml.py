# For linear algebra
import numpy as np
# For data processing
import pandas as pd,re
from scipy import stats
from sklearn import preprocessing
#Using SelectKBest to get the top features!
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
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
        self.X = df.loc[:,df.columns!='RainTomorrow']
        self.y = df[['RainTomorrow']]
        self.selector = SelectKBest(chi2, k=3)
        self.selector.fit(self.X, self.y)
        X_new =self. selector.transform(self.X)
        self.simplify()
        return self
    def buildModel(self):
        #Logistic Regression
        #Calculating the accuracy and the time taken by the classifier
        t0=time.time()
        #Data Splicing
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y.values.ravel(),test_size=0.25)
        clf_logreg = LogisticRegression(random_state=0)
        #Building the model using the training data set
        clf_logreg.fit(preprocessing.scale(X_train),y_train)
        #Evaluating the model using testing data set
        y_pred = clf_logreg.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        #Printing the accuracy and the time taken by the classifier
        self.scoreTime= ('score',score),('time',time.time()-t0)
        return self
    def getScoreTime(self):
        return self.scoreTime
    def simplify(self):
        #The important features are put in a data frame
        self.df = self.df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
        #To simplify computations we will use only one feature (Humidity3pm) to build the model
        X = self.df[['Humidity3pm']]
        y = self.df[['RainTomorrow']]
    def getXcolumns(self):
        return self.X.columns[self.selector.get_support(indices=True)]
