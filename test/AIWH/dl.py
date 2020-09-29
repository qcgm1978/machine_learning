#Import requires packages
import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# Import Keras, Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from .base import Base
class DL(Base):
    def preprocessStratified(self):
        #Sort the dataset by "class" to apply stratified sampling
        self.df.sort_values(by='Class', ascending=False, inplace=True)
        return self
