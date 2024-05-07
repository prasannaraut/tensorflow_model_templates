#numpy==1.24.3
#pandas==2.0.3
#Pillow==10.0.0
#scipy==1.10.1
#tensorflow==2.13.0
#tensorflow-datasets==4.9.2


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



############################ Naive forcecast
# assumes that future value to be same as current value
df["Naive_Forecast"] = df['Sales'].shift(1, fill_value=df["Sales"][0])
print(df.head())

# lets calculate error terms
naive_mse = tf.keras.metrics.mean_squared_error(df['Sales'], df["Naive_Forecast"]).numpy()
print("Mean squared error, Naive Forecaset : {}".format(naive_mse))

naive_mae = tf.keras.metrics.mean_absolute_error(df['Sales'], df["Naive_Forecast"]).numpy()
print("Mean absolute error, Naive Forecaset : {}".format(naive_mae))

naive_mape = tf.keras.metrics.mean_absolute_percentage_error(df['Sales'], df["Naive_Forecast"]).numpy()
print("Mean absolute percentage error, Naive Forecaset : {}".format(naive_mape))
print("\n\n")



#### Moving average
window = 30
df["Moving_Average_Forecast"] = df["Sales"].rolling(window=window).mean().shift(1)
for i in range(30):
    df.loc[i, "Moving_Average_Forecast"]= df["Sales"][0]

plt.figure(figsize=(12, 9))
# Plotting the training data in green
plt.plot(df['Date'], df['Sales'], 'green', label = 'Input Data')
plt.plot(df['Date'], df['Sales'], 'blue', label = 'Moving_Average_Forecast Data')
plt.title('Moving Average Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(df['Date'][::180], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

MovingAvg_mse = tf.keras.metrics.mean_squared_error(df['Sales'], df["Moving_Average_Forecast"]).numpy()
print("Mean squared error, MovingAvg : {}".format(MovingAvg_mse))

MovingAvg_mae = tf.keras.metrics.mean_absolute_error(df['Sales'], df["Moving_Average_Forecast"]).numpy()
print("Mean absolute error, MovingAvg : {}".format(MovingAvg_mae))

MovingAvg_mape = tf.keras.metrics.mean_absolute_percentage_error(df['Sales'], df["Moving_Average_Forecast"]).numpy()
print("Mean absolute percentage error, MovingAvg : {}".format(MovingAvg_mape))
print("\n\n")

##### Differencing
# Perform Seasonal Differencing
df["Differenciated_Sales"]=df["Sales"].diff(365).fillna(0)

plt.figure(figsize=(12, 9))
# Plotting the training data in green
plt.plot(df['Date'], df['Sales'], 'green', label = 'Input Data')
plt.plot(df['Date'], df['Differenciated_Sales'], 'blue', label = 'Differenciated Sales Data')
plt.title('Differenciation')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(df['Date'][::180], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
print("Note differenciated series is stationary")


# Restore trend and seasonality
df["Restored_Sales"] = df["Sales"].shift(365) + df["Differenciated_Sales"]

# Compute moving average on the restore data
df["Restore_Moving_Average_Forecast"] = df["Restored_Sales"].rolling(window=window).mean().shift(1)
for i in range(365+30):
    df.loc[i, "Restore_Moving_Average_Forecast"]= df["Sales"][0]

plt.figure(figsize=(12, 9))
# Plotting the training data in green
plt.plot(df['Date'], df['Sales'], 'green', label = 'Input Data')
plt.plot(df['Date'], df['Restore_Moving_Average_Forecast'], 'blue', label = "Restore Moving Average Forecast")
plt.title('Restore Moving Average Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(df['Date'][::180], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

RestoredMovingAvg_mse = tf.keras.metrics.mean_squared_error(df['Sales'], df["Restore_Moving_Average_Forecast"]).numpy()
print("Mean squared error, RestoredMovingAvg : {}".format(RestoredMovingAvg_mse))

RestoredMovingAvg_mae = tf.keras.metrics.mean_absolute_error(df['Sales'], df["Restore_Moving_Average_Forecast"]).numpy()
print("Mean absolute error, RestoredMovingAvg : {}".format(RestoredMovingAvg_mae))

RestoredMovingAvg_mape = tf.keras.metrics.mean_absolute_percentage_error(df['Sales'], df["Restore_Moving_Average_Forecast"]).numpy()
print("Mean absolute percentage error, RestoredMovingAvg : {}".format(RestoredMovingAvg_mape))
print("\n\n")


'''
####################################### Forecasting using Machine Learning (Sample Data)
temperature = np.arange(1,15)
window_size=3
batch_size=2
shuffle_buffer=10

dataset = tf.data.Dataset.from_tensor_slices(temperature)

dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
for window in dataset:
    window_data = " ".join([str(element.numpy()) for element in window])
    print(window_data)

# flattening
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
for element in dataset:
    print(element.numpy())

# shuffle the data
dataset =   dataset.shuffle(shuffle_buffer)
print("\n\nAfter shuffling the data")
for element in dataset:
    print(element.numpy())

# Mapping features and labels
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
print("\n\nAfter Map")
for x,y in dataset:
    print(print("x =", x.numpy(), "y =", y.numpy()))


# batching and prefetching
dataset = dataset.batch(batch_size).prefetch(1)
print("\nAfter batch and prefetch:")
for batch in dataset:
    print(batch)
'''

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


# creating windowed dataset
def windowd_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


window_size = 4
batch_size = 4
shuffle_buffer = 10
x_train_processed = windowd_dataset(x_train, window_size =window_size, batch_size = batch_size, shuffle_buffer = shuffle_buffer)
x_valid_processed = windowd_dataset(x_valid, window_size =window_size, batch_size = batch_size, shuffle_buffer = shuffle_buffer)


# Building model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[window_size]),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(1,  activation='relu'),
])


initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001)
optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)

#loss_model = "mean_absolute_percentage_error"
#loss_model = 'mean_absolute_error'
loss_model ="mse"

metrics_to_be_used = [tf.keras.metrics.RootMeanSquaredError()]
metrics_to_be_used_name = ["root_mean_squared_error"]

model.compile(optimizer=optimizer,loss=loss_model, metrics=metrics_to_be_used)


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-4, verbose=1)
EarlyStoppingMonitory = EarlyStopping(patience=4)


history = model.fit(x=x_train_processed,
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

y_true = np.array(x_valid)[:-1 * window_size]

print("r2_score: {}".format(sklearn.metrics.r2_score(y_true, y_pred)))
#print(sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred))
print("mean_absolute_error: {}".format(sklearn.metrics.mean_absolute_error(y_true, y_pred)))

plt.figure()
plt.plot(y_true, label="True")
plt.plot(y_pred, label="Predicted")
plt.title("Validatio set, true vs predicted")
plt.legend()
plt.show()
