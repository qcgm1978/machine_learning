from tensorflow.keras.layers.experimental.preprocessing import CenterCrop, Rescaling
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def compileModel(hp):
    return  Dataset.model.compile(optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])), loss="mse", run_eagerly=True)
import kerastuner
tuner = kerastuner.tuners.Hyperband(
  compileModel,
  objective='val_loss',
  max_epochs=100,
  max_trials=200,
  executions_per_trial=2,
  directory='my_dir')    
class Dataset(object):
    # Let's say we expect our inputs to be RGB images of arbitrary size
    inputs = keras.Input(shape=(None, None, 3))
    # Create a dataset.
    dataset = keras.preprocessing.image_dataset_from_directory(
        'test/keras', batch_size=64, image_size=(200, 200))
    datasetTxt = keras.preprocessing.text_dataset_from_directory(
        'test/keras', batch_size=64)
    vectorizer = TextVectorization(output_mode="int")
    model=None
    def getModel(self, outputs):
        model = keras.Model(inputs=self.inputs, outputs=outputs)
        self.model = model
        return model
    def getNp(self):
        return np
    def getKeras(self):
        return keras
    def getTf(self):
        return tf
    def getType(self):
        return tf.python.data.ops.dataset_ops.BatchDataset
    def getShapeDtype(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        # For demonstration, iterate over the batches yielded by the dataset.
        l = []
        for data, labels in dataset:
            l.append({'data': data, 'labels': labels})
        return l
    def asynPreprocess(self,samples,isAsyn=True):
        samples=np.array(samples)
        labels = [[0], [1]]
        # Prepare a TextVectorization layer.
        self.vectorizer.adapt(samples)
        # Asynchronous preprocessing: the text vectorization is part of the tf.data pipeline.
        # First, create a dataset
        dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
        if isAsyn:
            inputs = keras.Input(shape=(None,), dtype="int64")
            # Apply text vectorization to the samples
            dataset = dataset.map(lambda x, y: (self.vectorizer(x), y))
            # Prefetch with a buffer size of 2 batches
            dataset = dataset.prefetch(2)
            x = layers.Embedding(input_dim=10, output_dim=32)(inputs)
        else:
            inputs = keras.Input(shape=(1,), dtype="string")
            x = self.vectorizer(inputs)
            x = layers.Embedding(input_dim=10, output_dim=32)(x)
        # Our model should expect sequences of integers as inputs
        outputs = layers.Dense(1)(x)
        Dataset.model = keras.Model(inputs, outputs)
        Dataset.model=compileModel()
        return Dataset.model.fit(dataset)
    def e2e(self,samples):
        inputs = keras.Input(shape=(28, 28))
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(10, activation="softmax")(x)
        end_to_end_model = keras.Model(inputs, outputs)
        isE2E=True
        return end_to_end_model,isE2E
    def str2sequence(self, l, isBigrams=True):
        # Example training data, of dtype `string`.
        training_data = np.array(l)
        # Create a TextVectorization layer instance. It can be configured to either
        # return integer token indices, or a dense token representation (e.g. multi-hot
        # or TF-IDF). The text standardization and text splitting algorithms are fully
        # configurable.
        if isBigrams:
            vectorizer = TextVectorization(output_mode="binary", ngrams=2)
        else:
            vectorizer = TextVectorization(output_mode="int")
        # Calling `adapt` on an array or dataset makes the layer generate a vocabulary
        # index for the data, which can then be reused when seeing new data.
        vectorizer.adapt(training_data)
        # After calling adapt, the layer is able to encode any n-gram it has seen before
        # in the `adapt()` data. Unknown n-grams are encoded via an "out-of-vocabulary"
        # token.
        integer_data = vectorizer(training_data)
        return (integer_data)
    def normalizeFeatures(self):
        from tensorflow.keras.layers.experimental.preprocessing import Normalization
        # Example image data, with values in the [0, 255] range
        self.setImgData((64, 200, 200, 3))
        normalizer = Normalization(axis=-1)
        normalizer.adapt(self.training_data)
        normalized_data = normalizer(self.training_data)
        return normalized_data
    def rescaleCenterCrop(self,  height, width):
        cropper = CenterCrop(height=height, width=width)
        scaler = Rescaling(scale=1.0 / 255)
        output_data = scaler(cropper(self.training_data))
        return output_data
    def setImgData(self, size):
        # Example image data, with values in the [0, 255] range
        self.training_data = np.random.randint(
            0, 256, size=size).astype("float32")
        return self
    def get16maps(self):
        dense = keras.layers.Dense(units=16)
        return dense
    def layerTransform(self, num_classes):
        # Center-crop images to 150x150
        x = CenterCrop(height=150, width=150)(self.inputs)
        # Rescale images to [0, 1]
        x = Rescaling(scale=1.0 / 255)(x)
        # Apply some convolution and pooling layers
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
        # Apply global average pooling to get flat feature vectors
        x = layers.GlobalAveragePooling2D()(x)
        # Add a dense classifier on top
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return outputs
    def summary(self):
        l = []
        self.model.summary(print_fn=lambda item: l.append(item))
        return l
    def compile(self):
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy')
        return self
    def fit(self):
        self.model.fit(numpy_array_of_samples, numpy_array_of_labels,
                       batch_size=32, epochs=10)
    def toy(self, hasCallback=False, Model=keras.Model, run_eagerly=False):
        # Get the data as Numpy arrays
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[0:10]
        y_train = y_train[0:10]
        # Build a simple model
        inputs = keras.Input(shape=(28, 28))
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(10, activation="softmax")(x)
        model = self.modelCompile(Model, inputs, outputs, run_eagerly)
        # Train the model for 1 epoch from Numpy data
        batch_size = 64
        print("Fit on NumPy data")
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(batch_size)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='test/keras/model/model_{epoch}',
                save_freq='epoch'),
            keras.callbacks.TensorBoard(log_dir='test/keras/logs')
        ] if hasCallback else []
        self.history = model.fit(x_train, y_train, batch_size=batch_size,
                                 epochs=1, validation_data=self.val_dataset, callbacks=callbacks)
        # Train the model for 1 epoch using a dataset
        # dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        # print("Fit on Dataset")
        # self.history = model.fit(dataset, epochs=1)
        self.model = model
        return self
    def modelCompile(self, Model, inputs, outputs, run_eagerly):
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        # Open a strategy scope.
        if tf.test.gpu_device_name(): 
            with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
                model = self.__modelCompile(Model, inputs, outputs, run_eagerly)
        else:
                model = self.__modelCompile(Model, inputs, outputs, run_eagerly)
        return model
    def __modelCompile(self, Model, inputs, outputs, run_eagerly):
        model = Model(inputs, outputs)
        # model.summary()
        # Compile the model
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",
                      metrics=[
                          keras.metrics.SparseCategoricalAccuracy(name="acc")],
                      run_eagerly=run_eagerly
                      )
        return model
    def evaluate(self):
        loss, acc = self.model.evaluate(
            self.val_dataset)  # returns loss and metrics
        return loss, acc
    def predict(self):
        predictions = self. model.predict(self.val_dataset)
        return predictions
