import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from utilities import getPath,parseNumber
import numpy as np,tensorflow as tf
class Intro(np.ndarray,tf.data.Dataset):
    def __init__(self,l):
        super(tf.data.Dataset)  
    def getDataset(self):
        return tf.data.Dataset
    def getNp(self):
        return np
    def isValidData(self,data):
        return isinstance(data,Intro) or isinstance(data,tf.data.Dataset)
