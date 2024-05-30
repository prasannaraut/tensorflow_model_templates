# source : https://www.tensorflow.org/tutorials/structured_data/time_series

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False






class process_data_for_time_series_model():
    def __init__(self, history_points, future_points, gap, input_cols=None, label_columns=None, train_df=None, valid_df=None, test_df=None):
        self.all_cols = list(set(input_cols+label_columns))
        self.input_cols = input_cols
        self.label_columns = label_columns
        self.gap = gap
        # Store the raw data.
        if train_df is not None:
            self.train_df = train_df[self.all_cols]
        else:
            self.train_df = None

        if valid_df is not None:
            self.valid_df = valid_df[self.all_cols]
        else:
            self.valid_df = None

        if test_df is not None:
            self.test_df = test_df[self.all_cols]
        else:
            self.test_df = None


        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
              self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}

        if train_df is not None:
            self.column_indices = {name: i for i, name in
                                   enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = history_points
        self.label_width = future_points
        self.shift = gap+future_points

        self.total_window_size = history_points + future_points + gap

        self.input_slice = slice(0, history_points)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Input datasets are Train:{self.train_df is not None}, Valid:{self.valid_df is not None}, Test:{self.test_df is not None}',
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    # helper function used by make_dataset
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # helper function for final prediction in the future values
    def split_window_only_inputs(self, features):
        inputs = features[:, self.input_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])


        return inputs

    # convert dataframe to formatted tensor
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    # convert dataframe to formatted tensor only inputs
    def make_dataset_only_inputs(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width,
            sequence_stride=1,
            shuffle=False,
            #batch_size=32,
        )

        ds = ds.map(self.split_window_only_inputs)

        return ds


    def plot_train_data(self, cols=None):
        if cols is None:
            cols = self.input_cols

        if self.train_df is not None:
            plt.figure(figsize=(12, 8))
            for c in cols:
                plt.plot(self.train_df.index, self.train_df[c], label=c)

            plt.xlabel("Time")
            plt.title("Train data frame plots")
            plt.legend(loc="upper left")
            plt.plot()

    # note cross check the logic of plots once again
    def plot_train_data_withPredictions(self, train_output_predictions):

        train_df_toPlot = self.train_df.iloc[self.input_width-1:-1*self.label_width-1*self.gap, :]
        if self.train_df is not None:
            count_label=0
            for c_out in self.label_columns:
                plt.figure(figsize=(12, 8))
                plt.plot(train_df_toPlot.index, train_df_toPlot[c_out], label=c_out)
                for i in range(train_output_predictions.shape[1]):
                    plt.plot(train_df_toPlot.index, train_output_predictions[:, i, count_label], label="{}th prediction".format(i), linestyle='dashed')
                plt.xlabel("Time")
                plt.title("Train data frame plots for output column as {}".format(c_out))
                plt.legend(loc="upper left")
                plt.plot()
                count_label=count_label+1

    # note cross check the logic of plots once again
    def plot_valid_data_withPredictions(self, valid_output_predictions):

        valid_df_toPlot = self.valid_df.iloc[self.input_width-1:-1*self.label_width-1*self.gap, :]
        if self.valid_df is not None:
            count_label=0
            for c_out in self.label_columns:
                plt.figure(figsize=(12, 8))
                plt.plot(valid_df_toPlot.index, valid_df_toPlot[c_out], label=c_out)
                for i in range(train_output_predictions.shape[1]):
                    plt.plot(valid_df_toPlot.index, valid_output_predictions[:, i, count_label], label="{}th prediction".format(i), linestyle='dashed')
                plt.xlabel("Time")
                plt.title("Valid data frame plots for output column as {}".format(c_out))
                plt.legend(loc="upper left")
                plt.plot()
                count_label=count_label+1

    #note cross check the logic of plots once again
    def plot_test_data_withPredictions(self, test_output_predictions):

        test_df_toPlot = self.test_df.iloc[self.input_width-1:-1*self.label_width-1*self.gap, :]
        if self.valid_df is not None:
            count_label=0
            for c_out in self.label_columns:
                plt.figure(figsize=(12, 8))
                plt.plot(test_df_toPlot.index, test_df_toPlot[c_out], label=c_out)
                for i in range(train_output_predictions.shape[1]):
                    plt.plot(test_df_toPlot.index, test_output_predictions[:, i, count_label], label="{}th prediction".format(i), linestyle='dashed')
                plt.xlabel("Time")
                plt.title("Test data frame plots for output column as {}".format(c_out))
                plt.legend(loc="upper left")
                plt.plot()
                count_label=count_label+1


    #note cross check the logic of plots once again
    def plot_future_predictions(self, future_predictions, train_mean, train_std):

        df = pd.concat([self.train_df, self.valid_df, self.test_df])
        df = df[self.label_columns]

        df_len = len(df)

        for i in range(future_predictions.shape[1]):
            df.loc[df_len+i]=future_predictions[0,i,:]
        df=df.reset_index()

        if df is not None:
            count_label=0
            for c_out in self.label_columns:
                mean = train_mean[c_out]
                std = train_std[c_out]
                plt.figure(figsize=(12, 8))
                plt.plot(df.index[:-3], df[c_out][:-3]*std+mean, label=c_out)
                plt.plot(df.index[-3:], df[c_out][-3:]*std+mean, label=c_out)
                plt.xlabel("Time")
                plt.title("Final Prediction output column as {}".format(c_out))
                plt.legend(loc="upper left")
                plt.plot()
                count_label=count_label+1

##############################################################################################################################
# df = pd.read_csv("datasets/PET_PRI_GND_DCUS_NUS_W.csv", infer_datetime_format=True, index_col='Date', header=0)
df = pd.read_csv("datasets/PET_PRI_GND_DCUS_NUS_W.csv", index_col='Date')
df.index = np.linspace(1,len(df),len(df))

# df = pd.read_csv("datasets/jena_climate_2009_2016_v2.csv",  index_col='Date Time')
# df.index = np.linspace(1,len(df),len(df))
# cols = df.columns
# df["T"] = df["T (degC)"]
# df["P"] = df["p (mbar)"]
# df=df.drop(cols, axis=1)

# #df2 = pd.read_csv("datasets/simple_sin_func.csv", infer_datetime_format=True, index_col='T', header=0)
# #df.index = df["Date Time"]
# #df = df.drop(["Date Time"], axis=1)
# print(df.columns)

'''
x = np.linspace(1,2000,2000)
y = np.sin(x*3.1428/180)

df = pd.DataFrame()
df["x"] = x
df["y"] = y
'''


### Define Parameters
history_points=10
future_points=3
gap=0
input_cols = ["A1","A2","A3"]
output_cols = ["A1","A2"]


total_length = len(df)
train_df = df[ : int(total_length*0.7)]
valid_df = df[int(total_length*0.7) : int(total_length*0.9)]
test_df = df[int(total_length*0.9) : ]
future_prediction_df = df[total_length-history_points+1 :]



## Normalization of data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
valid_df = (valid_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
future_prediction_df = (future_prediction_df - train_mean) / train_std


# Process the data
w1 = process_data_for_time_series_model(history_points=history_points, future_points=future_points, gap=gap, input_cols=input_cols,  label_columns=output_cols, train_df=train_df, valid_df=valid_df, test_df=test_df)
#data = w1.get_data()






# Get Tensors
if train_df is not None:
    train_ds = w1.make_dataset(train_df)
if valid_df is not None:
    valid_ds = w1.make_dataset(valid_df)
if test_df is not None:
    test_ds = w1.make_dataset(test_df)

if future_prediction_df is not None:
    future_prediction_ds = w1.make_dataset_only_inputs(future_prediction_df)





########## Create models
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(4),
    tf.keras.layers.Dense(4),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(future_points * len(output_cols),
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([future_points, len(output_cols)])
])

model = lstm_model

model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=1e-6, verbose=1)

early_stopping = EarlyStopping(patience=15)


history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=1000,
                    callbacks=[reduce_lr, early_stopping])

metrics_to_be_used = []
variables_for_plot = ["loss"] + metrics_to_be_used
for var in variables_for_plot:

    loss_train = history.history["{}".format(var)]
    loss_val = history.history['val_{}'.format(var)]
    epochs = range(1,len(history.history['loss'])+1)
    plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('{}'.format(var))
    plt.xlabel('Epochs')
    plt.ylabel(var)
    plt.legend()
    plt.show()

# get predictions

train_output_predictions = model.predict(train_ds)
valid_output_predictions = model.predict(valid_ds)
test_output_predictions = model.predict(test_ds)
future_predictions = model.predict(future_prediction_ds)

# create prediction plots
w1.plot_train_data()
w1.plot_train_data_withPredictions(train_output_predictions)
w1.plot_valid_data_withPredictions(valid_output_predictions)
w1.plot_test_data_withPredictions(test_output_predictions)
w1.plot_future_predictions(future_predictions, train_mean, train_std)

print("########################## Final Prediction is##########################")
print(future_predictions)
