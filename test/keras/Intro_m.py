import os
import keras

# from tensorflow.python.framework.tensor_util import is_tensor
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from utilities import getPath,parseNumber
import numpy as np,tensorflow as tf
class Intro(np.ndarray,tf.data.Dataset,tf.keras.utils.Sequence):
    def __init__(self,l):
        super(tf.data.Dataset)  
    def get_keras(self):
        return tf.keras
    def getDataset(self):
        return tf.data.Dataset
    def getSequence(self):
        return tf.keras.utils.Sequence
    def getNp(self):
        return np
    def get_tf(self):
        return tf
    def isValidData(self,data):
        return isinstance(data,Intro) or isinstance(data,tf.data.Dataset) or isinstance(data,tf.keras.utils.Sequence)
    def isLayer(self,layer):
        return isinstance(layer,tf.keras.layers.Layer)
    def is_tensor(self,tensor):
        return isinstance(tensor,tf.Tensor)
    def is_model(self,model):
        return isinstance(model,keras.Model)