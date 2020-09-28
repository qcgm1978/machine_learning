# For linear algebra
import numpy as np
# For data processing
import pandas as pd,re
from scipy import stats
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import urllib.request
class ML(object):
    @property
    def shape(self):
        return self.df.shape
    @property
    def target(self):
        return self.__target
    @property
    def observations(self):
        return self.df.shape[0]
    @property
    def features(self):
        return self.df.shape[1]
    def defineObjective(self,objective):
        self.objective=objective
        return self
    def gatherData(self,url=None,skip=False):
        if not skip:
            if url is None:
                return print('No url supplied')
            else:
                try:
                    print('Beginning file download with urllib2...')
                    urllib.request.urlretrieve(url, '/Users/zhanghongliang/Documents/machine_learning/test/AIWH/data/weatherAUS.csv')
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
    def preprocessDrop(self,columns=None):
        if columns is None:
            return
        self.df=self.df.drop(columns=columns,axis=1)
        #Removing null values
        self.df = self.df.dropna(how='any')
        self.removeOutliers()
        return self
    def explorePredictor(self,target):
        self.__target=target
        df=self.df
        self.X = df.loc[:,df.columns!=self.target]
        self.y = df[[self.target]]
        #Using SelectKBest to get the top features!
        self.selector = SelectKBest(chi2, k=len(self.X.columns))
        self.selector.fit(self.X, self.y)
        self.y=self.y.values.ravel()
        X_new =self. selector.transform(self.X)
        return self
    def buildModel(self,classificationModel=0):
        models=(self.LogisticRegression,self.RandomForest,self.DecisionTree,self.supportVectorMachine)
        self.model=models[classificationModel]
        self.scoreTime=None
        return self
    def ModelEvaluationOptimization(self):
        score,t0=self.model()
        #Return the accuracy and the time taken by the classifier
        self.scoreTime= ('score',score),('time',time.time()-t0)
        return self
    def predict(self):
        return self.scoreTime
    
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
    def removeOutliers(self):
        z = np.abs(stats.zscore(self.df._get_numeric_data()))
        self.df= self.df[(z < 3).all(axis=1)]
        return self
    def preprocessReplace(self,l):
        for item in l:
            self.df[item[0]].replace(item[1],inplace = True)
        return self
    def preprocessNormalize(self):
        scaler = preprocessing.MinMaxScaler()
        # print(df[1:2].to_string(header=True))
        scaler.fit(self.df)
        self.df = pd.DataFrame(scaler.transform(self.df), index=self.df.index, columns=self.df.columns)
        self.df.iloc[4:10]
        return self
    def supportVectorMachine(self):
        #Calculating the accuracy and the time taken by the classifier
        t0=time.time()
        #Data Splicing
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.25)
        # clf_svc = svm.SVC(kernel='linear') # too slow
        clf_svc = svm.LinearSVC()
        #Building the model using the training data set
        clf_svc.fit(X_train,y_train)
        #Evaluating the model using testing data set
        y_pred = clf_svc.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        return score,t0
    #Decision Tree Classifier
    def DecisionTree(self):
        #Calculating the accuracy and the time taken by the classifier
        self.t0=time.time()
        #Data Splicing
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.25)
        clf_dt = DecisionTreeClassifier(random_state=0)
        #Building the model using the training data set
        clf_dt.fit(X_train,y_train)
        self.clf_dt=clf_dt
        self.X_test=X_test
        self.y_test=y_test
    def predictDecisionTree(self,X_test=None,y_test=None):
        if X_test is None:
            X_test=self.X_test
        if y_test is None:
            y_test=self.X_test
        #Evaluating the model using testing data set
        y_pred = self.clf_dt.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        return score,self.t0
    def RandomForest(self):
        #Random Forest Classifier
        #Calculating the accuracy and the time taken by the classifier
        t0=time.time()
        #Data Splicing
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.25)
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
        #Building the model using the training data set
        clf_rf.fit(X_train,y_train)
        #Evaluating the model using testing data set
        y_pred = clf_rf.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        return score,t0
    def LogisticRegression(self):
        #Logistic Regression
        #Calculating the accuracy and the time taken by the classifier
        t0=time.time()
        #Data Splicing
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.25)
        clf_logreg = LogisticRegression(random_state=0)
        #Building the model using the training data set
        clf_logreg.fit(preprocessing.scale(X_train),y_train)
        #Evaluating the model using testing data set
        y_pred = clf_logreg.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        return score,t0
    def exploreSimplify(self,input=None):
        if not isinstance(input,list):
            input=[input]
        #The important features are put in a data frame
        self.xColumns = self.getXcolumns()
        self.df = self.df[self.xColumns.tolist()+[self.target]]
        #To simplify computations we will use only one feature (Humidity3pm) to build the model
        self.X = self.df[input]
        self.y = self.df[self.target]
        return self
    def getXcolumns(self):
        return self.X.columns[self.selector.get_support(indices=True)]
