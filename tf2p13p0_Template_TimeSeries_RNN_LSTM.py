#numpy==1.24.3
#pandas==2.0.3
#Pillow==10.0.0
#scipy==1.10.1
#tensorflow==2.13.0
#tensorflow-datasets==4.9.2


# if developing CNN networks for timeseries forecasting use, always use padding as 'causal'


# Characteristics of time series
#   Trend: general direction in which time series is moving over the long term
#   Seasonality: repetative cycles occuring at regular intervals over a specified period
#   Cyclicity: irregular cycles that occure in a time series ovre a long period of time
#   Autocorrelation: correlation between a time series and a lagged version of itself
#   Noise: random flacutations in data points


# Types of time series data
#   Stationary and non-stationary time series: stationary when (mean, variance and autocorrelation) remain constant over time
#   Univariate and multivariate time series: univariate when we track just one metric over time


# Applications of time series
#   Forecasting
#   Imputed data
#   Anomaly detection
#   Trend analysis
#   Seasonality analysis


# Techniques for forecasting time series
#   ARIMA (Autoregressive Integrated Moving Average)
#   STL (seasonal and trend decomposing using LOESS)
#   Nieve forecasting
#   Moving average
#   DNN, RNN, LSTM


# Data partitioning methods for train and validation split
#   fixed partitioning
#   roll-forward partitioning


# Types of moving average
#   Centered moving average: calculates average around a central point. Not good for forecasting as future data is required
#   Trailing moving average: calculate average using most recent n data points


# Lambda Layer: The Lambda layer exists so that arbitrary expressions can be used as a Layer when constructing Sequential and Functional API models


#load libraries
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



# load data
#source: https://github.com/PacktPublishing/TensorFlow-Developer-Certificate-Guide/tree/main/Chapter%2012
df = pd.read_csv('datasets/sales_data.csv')
print(df.head())
print(df.describe())



# create plot
df.set_index('Date').plot()
plt.ylabel("Sales")
plt.title("Sales over time")
plt.xticks(rotation=90)
plt.show()


# split data as training and validation set
split_time = int(len(df)*0.8)

train_df = df[:split_time]
valid_df = df[split_time:]

plt.figure(figsize=(12, 9))
# Plotting the training data in green
plt.plot(train_df['Date'], train_df['Sales'], 'green',
label = 'Training Data')
plt.plot(valid_df['Date'], valid_df['Sales'], 'blue',
label = 'Validation Data')
plt.title('Fixed Partitioning')
plt.xlabel('Date')
plt.ylabel('Sales')
all_dates = np.concatenate([train_df['Date'],
valid_df['Date']])
plt.xticks(all_dates[::180], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


############################ Forecasting using Machine Leanring (Sales Data)

# extracting the data
time = pd.to_datetime(df["Date"])
sales = df["Sales"]

#split data
split_time = int(len(df) * 0.8)
time_train = time[:split_time]
time_valid = time[split_time:]

x_train = sales[:split_time]
x_valid = sales[split_time:]
x_valid = x_valid.reset_index()["Sales"]


# create windowed dataset simple function
def windowd_dataset_simpleFunc(series, window_size):
    x = []
    y = []
    for i in range (window_size, len(series)):
        x.append(list(series[i-window_size:i]))
        y.append(series[i])
    return x,y

window_size = 20
x_train_processed, y_train_processed = windowd_dataset_simpleFunc(x_train, window_size)
x_valid_processed, y_valid_processed = windowd_dataset_simpleFunc(x_valid, window_size)

x_train_processed = np.array(x_train_processed)
y_train_processed = np.array(y_train_processed)

x_valid_processed = np.array(x_valid_processed)
y_valid_processed = np.array(y_valid_processed)

# build a tensorflow dataset
batch_size = 4
buffer_size = 10000

train_data = tf.data.Dataset.from_tensor_slices((x_train_processed, y_train_processed))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).prefetch(1)



# Building model

model_dnn = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[window_size], kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1,  activation='relu'),
], name = "dnn")

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu', input_shape=(window_size,1)),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='causal', activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
    tf.keras.layers.Dense(1, activation='relu')

], name = "cnn")


model_rnn = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*100)
], name = "rnn")

model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1),
], name = "lstm")

# cnn captures the local patterns and lstm captures the temporal relationships
model_cnn_lstm = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='causal', input_shape=[window_size,1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
], name = "cnn_lstm")



model = model_dnn
#model = model_cnn
#model = model_rnn
#model = model_lstm
#model = model_cnn_lstm

# exponential decay
initial_learning_rate = 0.00001
lr_exponentialDecay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

# piecewise linear decay
lr_piecewise = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [100, 200, 300, 400, 500], [0.00001, 0.000005, 0.000001, 0.0000005, 0.0000005, 0.0000005])

# polynomial decay
lr_poly = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.00001,
    decay_steps=100,
    end_learning_rate=0.000001,
    power=1.0)


constant_learning_rate = 0.00001
if (model.name in ["dnn", "cnn", "rnn"]):
    constant_learning_rate = 0.000001
elif (model.name in ["lstm", "cnn_lstm"]):
    constant_learning_rate = 0.0001

optimizer=tf.keras.optimizers.Adam(learning_rate = constant_learning_rate)
#optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)

#loss_model = "mean_absolute_percentage_error"
#loss_model = 'mean_absolute_error'
loss_model ="mse"

metrics_to_be_used = [tf.keras.metrics.RootMeanSquaredError()]
metrics_to_be_used_name = ["root_mean_squared_error"]

model.compile(optimizer=optimizer,loss=loss_model, metrics=metrics_to_be_used)


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-4, verbose=1)
EarlyStoppingMonitory = EarlyStopping(patience=4)



history = model.fit(x=x_train_processed, y=y_train_processed, #x=train_data,
                    epochs = 300,
                    callbacks=[reduce_lr, EarlyStoppingMonitory]
                    )

model.summary()

variables_for_plot = ["loss"] + metrics_to_be_used_name

for var in metrics_to_be_used_name:

    loss_train = history.history["{}".format(var)]
    #loss_val = history.history['val_{}'.format(var)]
    epochs = range(1,len(history.history['loss'])+1)
    plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    #plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('{}'.format(var))
    plt.xlabel('Epochs')
    plt.ylabel(var)
    plt.legend()
plt.show()

####################################################### Evaluate Model #######################################
def eval_model(model):
    return model.evaluate(x_valid_processed)

all_models = [model]
for m in all_models:
    eval_model(m)



############################################ Post Process the data ############################################

y_pred = model.predict(x_valid_processed)
print(y_pred)

#y_true = np.array(x_valid)[:-1 * window_size]
y_true = y_valid_processed

print("r2_score: {}".format(sklearn.metrics.r2_score(y_true, y_pred)))
#print(sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred))
print("mean_absolute_error: {}".format(sklearn.metrics.mean_absolute_error(y_true, y_pred)))

plt.figure()
plt.plot(y_true, label="True")
plt.plot(y_pred, label="Predicted")
plt.title("Validatio set, true vs predicted")
plt.legend()
plt.show()