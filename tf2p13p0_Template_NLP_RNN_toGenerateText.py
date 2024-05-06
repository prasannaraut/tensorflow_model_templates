#numpy==1.24.3
#pandas==2.0.3
#Pillow==10.0.0
#scipy==1.10.1
#tensorflow==2.13.0
#tensorflow-datasets==4.9.2



# when compared with deep neural networks RNN have memory cell to retain information from the sequence data
# LSTM can retain memory over long distance when compared with RNN, tackling the vanishing gradient problem.
#       LSTM have Input, Forget, Output gates and cell state (which can remember information to be used later in sequence)
# BiLSTM (Bidirectional LSTM)
#       Unlike LSTM it processes information from start to end and then from end to start
# GRU (Gated Recurrent Unit)
#       Simpler than LSTM but avoids vanishing gradient problem of RNN.
#       GRU have Update gate and Reset gate
#







############ Text classification using AG news dataset
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import make_column_selector
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import re
import string
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import tensorflow_datasets as tfds #pip3 install tensorflow_datasets==4.9.2


pd.set_option('display.max_columns', 500)

