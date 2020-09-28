from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
class DecisionTree(object):
    # @property
    def features(self):
        return self.df.shape[1]
    def getDtree(self, features, y=None):
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(self.X, self.y)
        return dtree
    def explorePredictor(self,features,y,target=None):
        if y is None:
            y=self.target
        df = self.df
        X = df[features]
        self.features=features
        self.X=X
        self.y=df[y]
        if target:
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
    def createDecisionTreeData(self, features, y):
        if y is None:
            y=self.target
        df = self.df
        X = df[features]
        self.X=X
        self.y=df[y]
        # dtree = self.getDtree(features, y)
        self.DecisionTree()
        # start the other fit
        self.selector = SelectKBest(chi2, k=len(self.X.columns))
        self.selector.fit(self.X, self.y)
        # end
        self.graphData = tree.export_graphviz(
            self.clf_dt, out_file=None, feature_names=features
        )
        return self
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
        return self
    def buildModel(self,classificationModel=0):
        models=(
            # self.LogisticRegression,self.RandomForest,
            self.DecisionTree,
            # self.supportVectorMachine
            )
        self.model=models[classificationModel]
        self.scoreTime=None
        return self
    def ModelEvaluationOptimization(self):
        self.model()
        X_test=self.X_test
        y_test=self.y_test
        #Evaluating the model using testing data set
        y_pred = self.clf_dt.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        #Return the accuracy and the time taken by the classifier
        self.scoreTime= ('score',score),('time',time.time()-self.t0)
        self.graphByData()
        self.saveAndShow()
        return self