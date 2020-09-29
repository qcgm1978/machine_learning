#Import requires packages
import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import matplotlib.pyplot as plt,time
# Import Keras, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from .base import Base
class DL(Base):
    def __init__(self,d=None):
        self.imgIndex=0
        Base.__init__(self,d)
    def preprocessStratified(self):
        #Sort the dataset by "class" to apply stratified sampling
        self.df.sort_values(by='Class', ascending=False, inplace=True)
        return self
    def preprocessNormalize(self):
        # Standardize the features coloumns to increase the training speed
        scaler = MinMaxScaler()
        scaler.fit(self.train_feature)
        self.train_feature_trans = scaler.transform(self.train_feature)
        self.test_feature_trans = scaler.transform(self.test_feature)
        return self
    def buildModel(self,classificationModel=0):
        #Select model type
        models=(self.denseSequential,)
        self.evalModel=models[classificationModel]
        return self
    def ModelEvaluationOptimization(self,enableGraph=True):
        #Calculating the time taken by the classifier
        self.t0=time.time()
        self.model=self.evalModel()
        self.optimize().evaluate().model_efficacy()
        #Evaluating the model using testing data set
        # if hasattr(self,'clf'):
        #     X_test=self.X_test
        #     y_test=self.y_test
        #     y_pred = self.clf.predict(X_test)
        #     self.score = accuracy_score(y_test,y_pred)
        #     #Set the accuracy and the time taken by the classifier
        self.scoreTime= ('score',self.score),('time',time.time()-self.t0)
        # Display the accuracy curves for training and validation sets
        if enableGraph:
            self.graphData(self.train_history, 'accuracy', 'val_accuracy')
            # Display the loss curves for training and validation sets
            self.graphData(self.train_history, 'loss', 'val_loss')
        return self
    def predict(self,*args,custom=False,**kwargs):
        if custom:
            name = 'predict'+self.evalModel.__name__
            if hasattr(self,name):
                return getattr(self,name)(*args,**kwargs)
        else:
            return self.scoreTime
    def evaluate(self):
        # Testing set for model evaluation
        scores = self.model.evaluate(self.test_feature_trans,self. test_label)
        # Display accuracy of the model
        print('n')
        print('Accuracy=', scores[1])
        self.score= scores[1]
        prediction =self. model.predict(self.test_feature_trans)
        df_ans = pd.DataFrame({'Real Class': self.test_label})
        df_ans['Prediction'] = prediction
        df_ans['Prediction'].value_counts()
        df_ans['Real Class'].value_counts()
        cols = ['Real_Class_1', 'Real_Class_0'] # Gold standard
        rows = ['Prediction_1', 'Prediction_0'] # Diagnostic tool (our prediction)
        B1P1 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
        B1P0 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
        B0P1 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class'] == 0)])
        B0P0 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 0)])
        self.conf = np.array([[B1P1, B0P1], [B1P0, B0P0]])
        df_cm = pd.DataFrame(self.conf, columns=[i for i in cols], index=[i for i in rows])
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(df_cm, annot=True, ax=ax, fmt='d')
        # Making x label be on top is common in textbooks.
        ax.xaxis.set_ticks_position('top')
        print('Total number of test cases: ', np.sum(self.conf))
        return self.saveAndShow()
    # Model summary function
    def model_efficacy(self):
        conf=self.conf
        total_num = np.sum(conf)
        sen = conf[0][0] / (conf[0][0] + conf[1][0])
        spe = conf[1][1] / (conf[1][0] + conf[1][1])
        false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
        false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])
        print('Total number of test cases: ', total_num)
        print('G = gold standard, P = prediction')
        # G = gold standard; P = prediction
        print('G1P1: ', conf[0][0])
        print('G0P1: ', conf[0][1])
        print('G1P0: ', conf[1][0])
        print('G0P0: ', conf[1][1])
        print('--------------------------------------------------')
        print('Sensitivity: ', sen)
        print('Specificity: ', spe)
        print('False_positive_rate: ', false_positive_rate)
        print('False_negative_rate: ', false_negative_rate)
        return self
    def saveAndShow(self,enableShow=False):
        file='img/dl{0}.png'.format(self.imgIndex)
        self.imgIndex+=1
        plt.savefig(file)
        enableShow and plt.show()
        return self
    def optimize(self):
        # Using 'Adam' to optimize the Accuracy matrix
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
        # Fit the model
        # number of epochs = 200 and batch size = 500
        self.train_history =self. model.fit(x=self.train_feature_trans, y=self.train_label,
        validation_split=0.8, epochs=200,
        batch_size=500, verbose=2)
        return self
    def denseSequential(self):
        model=Sequential() # Adding a Dense layer with 200 neuron units and ReLu activation function
        model.add(Dense(units=200,
        input_dim=29,
        kernel_initializer='uniform',
        activation='relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        # Second Dense layer with 200 neuron units and ReLu activation function
        model.add(Dense(units=200,
        kernel_initializer='uniform',
        activation='relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        # The output layer with 1 neuron unit and Sigmoid activation function
        model.add(Dense(units=1,
        kernel_initializer='uniform',
        activation='sigmoid'))
        return model
    def graphData(self,train_history, train, validation,enableShow=False):
        # A function to plot the learning curves
        # def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='best')
        self.saveAndShow()
