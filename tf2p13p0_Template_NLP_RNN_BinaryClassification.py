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


#### Load the data
#dataset, info = tfds.load('ag_news_subset', with_info=True, as_supervised=True)
#train_dataset, test_dataset = dataset['train'], dataset['test']

# source: https://www.kaggle.com/datasets/hoshi7/news-sentiment-dataset

doc_link = "datasets/Sentiment_dataset.csv"
df = pd.read_csv(doc_link)

# just keep sentiment and text columns
df=df[["text","sentiment"]]

df['text'] = df['text'].str.lower()


def remove_special_characters(text):
    text = re.sub(r'http\S+', ' ', text )
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bhttps?://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d', ' ', text)  # Corrected line
    text= re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    return text

df['text'] = df['text'].apply(remove_special_characters)


################################################ Stratified train test split ######################################
# stratified train test
#df["temp"]=df["graduate"].astype(str) + df["Qualification"].astype(str)
df["temp"]=df["sentiment"].astype(str)

#train test split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["temp"]) #stratify=df["temp"],

print("Training set size: {}".format(train_set.shape))
print("Testing set size: {}".format(test_set.shape))


train_inputs = train_set.drop(['temp','sentiment'], axis=1)
train_labels = train_set['sentiment'].copy()

test_inputs = test_set.drop(['temp','sentiment'], axis=1)
test_labels = test_set['sentiment'].copy()



vocab_size = 10000
embedding_dim = 128
max_length = 2048
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
num_epochs = 50


#get list of all titles and abstracts
train_list = []
test_list = []

for v in train_inputs.iterrows():
    train_list.append(v[1][0])

for v in test_inputs.iterrows():
    test_list.append(v[1][0])


# word embedding
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_tok, num_words = vocab_size) # define out of vocabulary words token

tokenizer.fit_on_texts(train_list)
print("Showing tokenizers - Words")
print(tokenizer.word_index)
print("\n\n\n")


# convert to sequence
train_sequence = tokenizer.texts_to_sequences(train_list)


test_sequence = tokenizer.texts_to_sequences(test_list)


# pad and truncate
padded_train_sequence = tf.keras.preprocessing.sequence.pad_sequences(train_sequence, padding=padding_type, truncating=trunc_type, maxlen=max_length)
train_data_array = padded_train_sequence
train_labels_array = np.array(train_labels)

padded_test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, padding=padding_type, truncating=trunc_type, maxlen=max_length)
test_data_array = padded_test_sequence
test_labels_array = np.array(test_labels)

#train_data_array = np.concatenate((padded_train_sequence, padded_sequence_abstract), axis=1)


###################################### Prepare Tensorflow Model ####################################################
model_dnn =  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


model_cnn=  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model_lstm=  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


model_bilstm=  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


#model = model_dnn
#model = model_cnn
#model = model_lstm
model = model_bilstm

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
#optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)

loss_model = "binary_crossentropy"
#loss_model = 'categorical_crossentropy'
#loss_model = 'mean_absolute_error'

metrics_to_be_used = ["accuracy"]

model.compile(optimizer=optimizer,loss=loss_model, metrics=metrics_to_be_used)
model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-4, verbose=1)

EarlyStoppingMonitory = EarlyStopping(patience=4)



history = model.fit(x=train_data_array,y=train_labels_array,
                    #validation_data = (test_inputs_array,test_labels_array),
                    validation_split = 0.2,
                    epochs = 10,
                    shuffle = True,
                    callbacks=[reduce_lr,EarlyStoppingMonitory])

print(model.summary())

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

########################### Evaluate model
def eval_model(model):
    return model.evaluate(test_data_array, test_labels_array)

all_models = [model]
for m in all_models:
    eval_model(m)


y_pred = model.predict(test_data_array)
y_pred = np.round(y_pred).astype('int')

print(y_pred)



#Generating the confusion matrix
print("Confusion matrix")
eval = confusion_matrix(test_labels_array, y_pred)
print("######### Confusion Matrix #############")
print(eval)
print()

print("############# Classification Report##############")
class_names = [ 'Wrong', 'Correct']
print("Classification Report")
print(classification_report(test_labels_array, y_pred, target_names=class_names))
print()
