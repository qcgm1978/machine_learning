import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
from functools import reduce
import keras
from keras import layers
class FitEvaluate(object):
    def __init__(self,topDense,l):
        self.compileParam=self.getCompileParam(l)
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
    def fitEval(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.cols=x_train.shape[1]
        # Preprocess the data (these are NumPy arrays)
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        self.x_test = x_test.reshape(10000, 784).astype("float32") / 255
        y_train = y_train.astype("float32")
        self.y_test = y_test.astype("float32")
        # Reserve 10,000 samples for validation
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]
        self.model.compile(
            # optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # # Loss function to minimize
            # loss=keras.losses.SparseCategoricalCrossentropy(),
            # # List of metrics to monitor
            # metrics=[keras.metrics.SparseCategoricalAccuracy()],
            # optimizer="rmsprop",
            # loss="sparse_categorical_crossentropy",
            # metrics=["sparse_categorical_accuracy"],
            optimizer=self.compileParam[0],
            loss=self.compileParam[1],
            metrics=[self.compileParam[2]]
        )
        print("Fit model on training data")
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=2,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_val, y_val),
        )
        return history
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