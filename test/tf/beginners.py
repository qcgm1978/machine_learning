import tensorflow as tf,inspect
from tensorflow.python.keras.engine.training import _is_scalar
mnist = tf.keras.datasets.mnist

class Beginners(object):
  def __init__(self,cols,*args):
    self.cols=cols
    names=('x_train', 'y_train','x_test', 'y_test')
    if len(args)==len(names):
      for name,arg in zip(names,args):
        setattr(self, name,arg)
  def prepare_data(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #     instead of modifying internal properties and returning a reference to the same object, the object is instead cloned, with properties changed on the cloned object, and that object returned.
    # https://en.wikipedia.org/wiki/Fluent_interface
    return Beginners(self.cols,x_train, y_train,x_test, y_test)
  def get_model(self):
    self.model=tf.keras.models.Sequential([ tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(self.cols) ])
    return self
  def op_model(self):
    model=self.model
    self.predictions = model(self.x_train[:1]).numpy()
    self.probabilities=tf.nn.softmax(self.predictions).numpy()
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.untrain_prob=self.loss_fn(self.y_train[:1],self. predictions).numpy()
    self.negative_log_prob=-tf.math.log(1/self.cols)
    model.compile(optimizer='adam',
                  loss=self.loss_fn,
                  metrics=['accuracy'])
    history=model.fit(self.x_train,self. y_train, epochs=5,verbose=0)
    self.is_history=(history,tf.python.keras.callbacks.History)
    self.sc=model.evaluate(self.x_test, self. y_test, verbose=2)
    self.is_s=_is_scalar(self.sc)
    probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
    ])
    self.p_m=probability_model(self.x_test[:5])
    return self