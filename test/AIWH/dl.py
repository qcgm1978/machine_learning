#Import requires packages
import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import matplotlib.pyplot as plt
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
        # self.t0=time.time()
        self.model=self.evalModel()
        self.optimize()
        #Evaluating the model using testing data set
        # if hasattr(self,'clf'):
        #     X_test=self.X_test
        #     y_test=self.y_test
        #     y_pred = self.clf.predict(X_test)
        #     self.score = accuracy_score(y_test,y_pred)
        #     #Set the accuracy and the time taken by the classifier
        #     self.scoreTime= ('score',self.score),('time',time.time()-self.t0)
        # Display the accuracy curves for training and validation sets
        if enableGraph:
            self.graphData(self.train_history, 'accuracy', 'val_accuracy')
            # Display the loss curves for training and validation sets
            self.graphData(self.train_history, 'loss', 'val_loss')

        # self.saveAndShow()
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
        file='img/dl{0}.png'.format(self.imgIndex)
        self.imgIndex+=1
        plt.savefig(file)
        enableShow and plt.show()
