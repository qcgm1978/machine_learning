# For linear algebra
import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in function %(funcName)s] %(message)s',
    # datefmt='',
    level=logging.INFO)
import numpy as np
# For data processing
import pandas as pd,re,pydotplus,matplotlib.image as pltimg, matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm,tree
from .AI import AI
class ML(AI):
    @property
    def target(self):
        return self.__target
    @property
    def observations(self):
        return self.df.shape[0]
    @property
    def dataCols(self):
        return self.df.shape[1]
    categories={
        'Regression': {'type':'Supervised','output':'continuous','aim':'predict','algorithm':'Linear Regression'},
        'Classification':{'type':'Supervised', 'output':'categorical' ,'aim':'category' ,'algorithm':'Logistic Regression'},
        'Clustering':{'type':'Unsupervised','output':'clusters','aim':'group','algorithm':'K-means'}
    }
    def getProblemCategory(self):
        return self.categories[self.category]
    def preprocessDrop(self,columns=None):
        if columns is None:
            return
        self.preprocessDropCols(columns)
        #Removing null values
        self.df = self.df.dropna(how='any')
        self.removeOutliers()
        return self
    def EDApredictor(self,features=None,target=None):
        df=self.df
        self.__target=target
        if features:
            self.X=df[features]
        else:
            self.X = df.loc[:,df.columns!=self.target]
        self.features=self.X.columns
        # logging.info(self.features)
        self.y=df[target]
        #Using SelectKBest to get the top features!
        topFeaturesNum = self.topFeaturesNum or len(self.X.columns)
        self.selector = SelectKBest(chi2, k=topFeaturesNum)
        self.selector.fit(self.X, self.y)
        self.y=self.y.values.ravel()
        X_new =self. selector.transform(self.X)
        return self
    def buildModel(self,classificationModel=0):
        models=(self.LogisticRegression,self.RandomForest,self.DecisionTree,self.supportVectorMachine)
        self.evalModel=models[classificationModel]
        #Data Splicing
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.25)
        self.scoreTime=None
        return self
    def ModelEvaluationOptimization(self,enableGraph=True):
        #Calculating the time taken by the classifier
        self.t0=time.time()
        self.evalModel()
        #Evaluating the model using testing data set
        if hasattr(self,'clf'):
            X_test=self.X_test
            y_test=self.y_test
            y_pred = self.clf.predict(X_test)
            self.score = accuracy_score(y_test,y_pred)
            #Set the accuracy and the time taken by the classifier
            self.scoreTime= ('score',self.score),('time',time.time()-self.t0)
        enableGraph and self.graphByData()
        # self.saveAndShow()
        return self
    def predict(self,*args,custom=False,**kwargs):
        if custom:
            name = 'predict'+self.evalModel.__name__
            if hasattr(self,name):
                return getattr(self,name)(*args,**kwargs)
        else:
            return self.scoreTime
    def graphByData(self, img = "img/mydecisiontree.png"):
        if not hasattr(self,'features'):
            return self
        if not hasattr(self,'graphData'):
            try:
                self.graphData = tree.export_graphviz(
                    self.clf, out_file=None, feature_names=self.features
                )
            except (AttributeError,ValueError) as e:
                logging.info(e)
                return self
        graph = pydotplus.graph_from_dot_data(self.graphData)
        graph.write_png(img)
        img = pltimg.imread(img)
        imgplot = plt.imshow(img)
        return self
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
        #Calculating the accuracy
        #Data Splicing
        # clf_svc = svm.SVC(kernel='linear') # too slow
        clf_svc = svm.LinearSVC()
        #Building the model using the training data set
        clf_svc.fit(self.X_train,self.y_train)
        self.clf=clf_svc
        return self
    #Decision Tree Classifier
    def DecisionTree(self):
        #Calculating the accuracy
        clf_dt = DecisionTreeClassifier(random_state=0)
        #Building the model using the training data set
        clf_dt.fit(self.X_train,self.y_train)
        self.clf=clf_dt
        return self
    def predictDecisionTree(self,val=None):
        if val:
            if len(np.array(val).shape)==1:
                val=[val]
            p = self.clf.predict(val)[0]
            return (p,'GO' if p else 'NO')
    def RandomForest(self):
        #Random Forest Classifier
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
        #Building the model using the training data set
        clf_rf.fit(self.X_train,self.y_train)
        self.clf=clf_rf
        return self
    def LogisticRegression(self):
        #Logistic Regression
        #Calculating the accuracy and the time taken by the classifier
        clf_logreg = LogisticRegression(random_state=0)
        #Building the model using the training data set
        clf_logreg.fit(preprocessing.scale(self.X_train),self.y_train)
        self.clf=clf_logreg
        return self
    def exploreSimplify(self,input=None):
        if input and not isinstance(input,list):
            input=[input]
        #The important features are put in a data frame
        self.xColumns = self.extractFeatures()
        self.features = input if input else self.xColumns+[self.target]
        self.X = self.df[self.features]
        #To simplify computations we will use only one feature (Humidity3pm) to build the model
        self.y = self.df[self.target]
        return self
    def extractFeatures(self):
        return self.X.columns[self.selector.get_support(indices=True)]
