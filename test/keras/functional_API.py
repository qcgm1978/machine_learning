from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop,Rescaling
import keras
def get_outputs(inputs,num_classes):
    # Center-crop images to 150x150
    x = CenterCrop(height=150, width=150)(inputs)
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
class Chain(object):
    def __call__(self, act):
        print( "I am:")
        method = getattr(self, act)
        if not method:
            pass
        method()
        return self
    def setInputs(self):
        self.inputs=keras.Input(shape=(784,))
        return self
    def setDense(self):
        self.dense=self.getDense()
        return self
    def getDense(self):
        return layers.Dense(64, activation="relu")
    def setX(self,x=None):
        if x is None:
            self.x=self.dense(self.inputs)
        else:
            self.x=self.getDense()(self.x)
        return self
    # x = layers.Dense(64, activation="relu")(x)
    def outputs(self):
        self.outputs = layers.Dense(10)(self.x)
        return self
    def createModel(self):
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name="mnist_model")
        return self
    def summary(self):
        self.model.summary()
        return self
    def plot(self, show_shapes=False):
        keras.utils.plot_model(self.model, "img/demo.png", show_shapes=show_shapes)
        return self
    def t_e_i(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )
        history =self. model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2,verbose=0)
        test_scores = self.model.evaluate(x_test, y_test, verbose=2)
        self.loss= test_scores[0]
        self.accuracy=test_scores[1]
        return self
    def save(self):
        self.model.save("test/keras/model")
        return self
    def delete(self):
        del self.model
        return self
    def load(self):
        # Recreate the exact same model purely from the file:
        self.model = keras.models.load_model("test/keras/model")
        return self
    def get_reverse_layer(self,l):
        pair1=[layers.Conv2D,layers.MaxPooling2D]
        pair2=[layers.Conv2DTranspose,layers.UpSampling2D]
        l1=[]
        for item in l:
            ind=pair1.index(item)
            if isinstance(ind,int):
                l1.append(pair2[ind])
            else:
                raise ValueError('No corresponding reverse evaluation')
        return l1
    def encoder(self):
        l=[layers.Conv2D,layers.Conv2D,layers.MaxPooling2D,layers.Conv2D,layers.Conv2D]
        encoder_input = keras.Input(shape=(28, 28, 1), name="img")
        x = l[0](16, 3, activation="relu")(encoder_input)
        x = l[1](32, 3, activation="relu")(x)
        x = l[2](3)(x)
        x = l[3](32, 3, activation="relu")(x)
        x = l[4](16, 3, activation="relu")(x)
        self.encoder_input=encoder_input
        self.x=x
        self.l=l
        return self
    def decoder(self):
        l=self.get_reverse_layer(self.l)
        x=self.x
        encoder_output = layers.GlobalMaxPooling2D()(x)
        encoder = keras.Model(self.encoder_input, encoder_output, name="encoder")
        # encoder.summary()
        x = layers.Reshape((4, 4, 1))(encoder_output)
        x = l[0](16, 3, activation="relu")(x)
        x = l[1](32, 3, activation="relu")(x)
        x = l[2](3)(x)
        x = l[3](16, 3, activation="relu")(x)
        decoder_output = l[4](1, 3, activation="relu")(x)
        autoencoder = keras.Model(self.encoder_input, decoder_output, name="autoencoder")
        # autoencoder.summary()  
        self.decoder_output=decoder_output
        return self
    # todo All models are callable, just like layers https://keras.io/guides/functional_api/