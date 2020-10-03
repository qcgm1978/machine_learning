import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
from functools import reduce
import keras,tensorflow as tf
from keras import layers
class FitEvaluate(object):
    def __init__(self,topDense,l,epochs):
        self.compileParam=self.getCompileParam(l)
        self.epochs=epochs
        inputs = keras.Input(shape=(784,), name="digits")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(topDense, activation="softmax", name="predictions")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
    def getKeras(self):
        return keras
    def getParam(self,s):
        re.search(r'[A-Z].*(?=.>)',s).group()
    def getCompileParam(self,l):
        ret=[]
        for item in l:
            ret.append(self.conver2str(item))
        return ret
    def conver2str(self, s):
        if not isinstance(s,str):
            s=str(s)
        def addUnderscore(matchobj):
            s=matchobj.group(0)
            if re.search(r'[A-Z]',s): 
                return '_'+s.lower()
            else: 
                return s
        g=re.search(r'[A-Z].*(?=.>)',s).group()
        g=re.sub(r'[A-Z]+[a-z]+',addUnderscore,g)
        g=''.join(list(g))[1:]
        return g
    def custom_mean_squared_error(self,y_true, y_pred):
        return tf.math.reduce_mean(tf.square(y_true - y_pred))
    def hotFit(self):
        # We need to one-hot encode the labels to use MSE
        y_train_one_hot = tf.one_hot(self.y_train, depth=10)
        return self.model.fit(self.x_train, y_train_one_hot, batch_size=64, epochs=1)
    def fit(self,isHot=True):
        (self.x_train, self.y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.cols=self.x_train.shape[1]
        # Preprocess the data (these are NumPy arrays)
        self.x_train = self.x_train.reshape(60000, 784).astype("float32") / 255
        self.x_test = x_test.reshape(10000, 784).astype("float32") / 255
        self.y_train = self.y_train.astype("float32")
        self.y_test = y_test.astype("float32")
        # Reserve 10,000 samples for validation
        x_val = self.x_train[-10000:]
        y_val = self.y_train[-10000:]
        self.x_train = self.x_train[:-10000]
        self.y_train = self.y_train[:-10000]
        self.model.compile(
            optimizer=keras.optimizers.Adam(), loss=self.custom_mean_squared_error,
            # metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        print("Fit model on training data")
        if isHot:
            return self.hotFit()
        else:
            return self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=64,
            epochs=self.epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_val, y_val),
        )
    def test(self):
        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=128)
        print("test loss, test acc:", results)
        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        print("Generate predictions for 3 samples")
        predictions = self.model.predict(self.x_test[:3])
        print("predictions shape:", predictions.shape)
        return predictions