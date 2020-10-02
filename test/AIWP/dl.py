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
from .ml import ML
class DL(ML):
    def __init__(self,d=None):
        self.trainField = 'accuracy'
        self.isDL=True
        self.imgIndex=0
        self.testCounts=[]
        ML.__init__(self,d)
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
        self.normalized=True
        return self
    def buildModel(self,classificationModel=0):
        #Select model type
        models=(self.denseSequential,)
        self.evalModel=models[classificationModel]
        return self
    def ModelEvaluationOptimization(self,enableGraph=True):
        #Calculating the time taken by the classifier
        self.t0=time.time()
        if self.hasLoadModel:
            self.debugEvalModel()
        else:
            self.evalModel().optimize().evaluate().model_efficacy()
        self.accuracyTime= (self.trainField,self.score),('time',time.time()-self.t0)
        # Display the accuracy curves for training and validation sets
        # if enableGraph:
        #     self.graphData(self.train_history,self.trainField , 'val_'+self.trainField)
        #     # Display the loss curves for training and validation sets
        #     self.graphData(self.train_history, 'loss', 'val_loss')
        return self
    def predict(self,*args,custom=False,times=1,**kwargs):
        if custom:
            name = 'predict'+self.evalModel.__name__
            if hasattr(self,name):
                return getattr(self,name)(*args,**kwargs)
        else:
            if times==1:
                return self.accuracyTime
            else:
                l=[self.accuracyTime]
                tot=0
                for i in range(times-1):
                    if self.hasLoadModel:
                        self.debugEvalModel()
                    else:
                        self.ModelEvaluationOptimization(enableGraph=False)
                    l.append(self.accuracyTime)
                    tot+=self.accuracyTime[0][1]
                return l,tot/times
    def evaluate(self):
        # Testing set for model evaluation
        scores = self.model.evaluate(self.test_feature_trans,self. test_label)
        # Set accuracy of the model
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
        self.tests = np.array([[B1P1, B0P1], [B1P0, B0P0]])
        df_cm = pd.DataFrame(self.tests, columns=[i for i in cols], index=[i for i in rows])
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(df_cm, annot=True, ax=ax, fmt='d')
        # Making x label be on top is common in textbooks.
        ax.xaxis.set_ticks_position('top')
        self.testCounts.append(np.sum(self.tests))
        return self.saveAndShow()
    def predictdenseSequential(self,n):
        # make class predictions with the model
        X = self.currentFitData['x']
        predictions = self.model.predict(X)
        # summarize the first n cases
        numerator= denominator=0
        for i,(p,y) in enumerate(zip(predictions, self.currentFitData['y'])):
            if i==n:
                break
            # print(p,y)
            if round(p[0])==y:
                numerator+=1
            else:
                denominator+=1
        return numerator/denominator if denominator else 1
    # Model summary function
    def model_efficacy(self):
        conf=self.tests
        total_num = np.sum(conf)
        sen = conf[0][0] / (conf[0][0] + conf[1][0])
        spe = conf[1][1] / (conf[1][0] + conf[1][1])
        false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
        false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])
        testsInfo={}
        testsInfo['count']= total_num
        # 'G = gold standard, P = prediction'
        testsInfo['G1P1']= conf[0][0]
        testsInfo['G0P1']= conf[0][1]
        testsInfo['G1P0']= conf[1][0]
        testsInfo['G0P0']= conf[1][1]
        testsInfo['Sensitivity']= sen
        testsInfo['Specificity']= spe
        testsInfo['False_positive_rate']= false_positive_rate
        testsInfo['False_negative_rate']= false_negative_rate
        self.testsInfo=testsInfo
        return self
    def saveAndShow(self,enableShow=False):
        file='img/dl{0}.png'.format(self.imgIndex)
        self.imgIndex+=1
        plt.savefig(file)
        enableShow and plt.show()
        return self
    def optimize(self):
        # Using 'Adam' to optimize the Accuracy matrix
        self.model.compile(
            loss='binary_crossentropy', 
            optimizer='adam',
            metrics=[self.trainField])
        # Fit the model
        self.currentFitData={
            'x':self.train_feature_trans, 
            'y':self.train_label,
        }
        self.train_history =self. model.fit(
            **self.currentFitData,
            validation_split=0.8, 
            epochs=200,
            batch_size=500, 
            # verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).
            verbose=0
        )
        return self
    def getHistory(self):
        return self.train_history.history.keys()
    def denseSequential(self):
        model=Sequential() 
        if hasattr(self,'input_dim'):
            input_dim=self.input_dim
        else:
            input_dim=self.shape[1]-1
        # Adding a Dense layer with 200 neuron units and ReLu activation function
        model.add(Dense(units=200,
        input_dim=input_dim,
        kernel_initializer='uniform',
        activation='relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        # Second Dense layer with 200 neuron units and ReLu activation function
        model.add(
            Dense(units=200,
                kernel_initializer='uniform',
                activation='relu')
        )
        # Add Dropout
        model.add(Dropout(0.5))
        # The output layer with 1 neuron unit and Sigmoid activation function
        model.add(Dense(units=1,
            kernel_initializer='uniform',
            activation='sigmoid'))
        self.model=model
        self.setCategory('Classification')
        return self
    def interpretPlotCurve(self):
        history=self.train_history
        # # summarize history for accuracy
        # plt.plot(list(history.history['accuracy']))
        plt.plot(history.history['accuracy'])
        # # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        self.saveAndShow()
        # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # self.saveAndShow()
        return self
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
    def sequential(self):
        model = Sequential()
        model.add(Dense(2, input_dim=1, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        self.model=model
        return self