import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utilities import getPath,parseNumber
import numpy as np,tensorflow as tf
# class Intro1(tf._api.v2.data):
#     pass
# class Intro2(np.ndarray):
#     pass
class Intro(np.ndarray,tf.data.Dataset):
    def __init__(self,l):
        super(tf.data.Dataset)  
    def isValidData(self,data):
        return isinstance(data,Intro) 
