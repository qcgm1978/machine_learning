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
    class CustomMSE(keras.losses.Loss):
        def __init__(self, regularization_factor=0.1, name="custom_mse"):
            super().__init__(name=name)
            self.regularization_factor = regularization_factor
        def call(self, y_true, y_pred):
            mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
            reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
            return mse + reg * self.regularization_factor
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
            optimizer=keras.optimizers.Adam(), loss=self.CustomMSE(),
            metrics=[CategoricalTruePositives()],
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
    def handle_matrics_function(self):
        inputs = keras.Input(shape=(784,), name="digits")
        x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x2 = layers.Dense(64, activation="relu", name="dense_2")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.add_loss(tf.reduce_sum(x1) * 0.1)
        model.add_metric(keras.backend.std(x1), name="std_of_activation", aggregation="mean")
        model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        return model.fit(self.x_train,self. y_train, batch_size=64, epochs=1)
    def handle_matrics(self):
        inputs = keras.Input(shape=(784,), name="digits")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        # Insert std logging as a layer.
        x = self.MetricLoggingLayer()(x)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(10, name="predictions")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        return model.fit(self.x_train, self.y_train, batch_size=64, epochs=1)
    class MetricLoggingLayer(layers.Layer):
        def call(self, inputs):
            # The `aggregation` argument defines
            # how to aggregate the per-batch values
            # over each epoch:
            # in this case we simply average them.
            self.add_metric(
                keras.backend.std(inputs), name="std_of_activation", aggregation="mean"
            )
            return inputs  # Pass-through layer.
    class ActivityRegularizationLayer(layers.Layer):
        def call(self, inputs):
            self.add_loss(tf.reduce_sum(inputs) * 0.1)
            return inputs  # Pass-through layer.
    def getModel(self):
        inputs = keras.Input(shape=(784,), name="digits")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        # Insert activity regularization as a layer
        x = self.ActivityRegularizationLayer()(x)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(10, name="predictions")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        return model
    def getHistory(self):
        # The displayed loss will be much higher than before
        # due to the regularization component.
        return self.getModel().fit(self.x_train, self.y_train, batch_size=64, epochs=1)
    def compile_no_loss(self):
        import numpy as np
        inputs = keras.Input(shape=(3,), name="inputs")
        targets = keras.Input(shape=(10,), name="targets")
        logits = keras.layers.Dense(10)(inputs)
        predictions = LogisticEndpoint(name="predictions")(logits, targets)
        model = keras.Model(inputs=[inputs, targets], outputs=predictions)
        model.compile(optimizer="adam")  # No loss argument!
        data = {
            "inputs": np.random.random((3, 3)),
            "targets": np.random.random((3, 10)),
        }
        return model.fit(data)
class CategoricalTruePositives(keras.metrics.Metric):
        def __init__(self,  **kwargs):
            super(CategoricalTruePositives, self).__init__('categorical_true_positives', **kwargs)
            self.true_positives = self.add_weight(name="ctp", initializer="zeros")
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
            values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
            values = tf.cast(values, "float32")
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, "float32")
                values = tf.multiply(values, sample_weight)
            self.true_positives.assign_add(tf.reduce_sum(values))
        def result(self):
            return self.true_positives
        def reset_states(self):
            # The state of the metric will be reset at the start of each epoch.
            self.true_positives.assign(0.0)
class LogisticEndpoint(keras.layers.Layer):
        def __init__(self, name=None):
            super(LogisticEndpoint, self).__init__(name=name)
            self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            self.accuracy_fn = keras.metrics.BinaryAccuracy(name="accuracy")
        def call(self, targets, logits, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weights)
            self.add_loss(loss)
            # Log accuracy as a metric and add it
            # to the layer using `self.add_metric()`.
            acc = self.accuracy_fn(targets, logits, sample_weights)
            self.add_metric(acc)
            # Return the inference-time prediction tensor (for `.predict()`).
            return tf.nn.softmax(logits)