#numpy==1.24.3
#pandas==2.0.3
#Pillow==10.0.0
#scipy==1.10.1
#tensorflow==2.13.0
#tensorflow-datasets==4.9.2
#matplotlib=3.7.2
#scikit-learn=1.3.0
#nltk=3.8.1


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



########## read stories data
text = open('datasets/stories.txt').read().lower()


def remove_special_characters(text):
    text = re.sub(r'http\S+', ' ', text )
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bhttps?://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d', ' ', text)  # Corrected line
    text= re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    return text

#text = remove_special_characters(text)


vocab_size = 10000
embedding_dim = 200
trunc_type='post'
padding_type='pre'
oov_tok = "<OOV>"
num_epochs = 50


# word embedding
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_tok, num_words = vocab_size) # define out of vocabulary words token

tokenizer.fit_on_texts([text])
print("Showing tokenizers - Words")
print(tokenizer.word_index)
print("\n\n\n")


total_words = len(tokenizer.word_index) + 1

# convert to sequence
input_sequence = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]


    for i in range(1, len(token_list)):
        seq = token_list[:i+1]
        input_sequence.append(seq)


## pad the sequence
max_sequence_len = max([len(x) for x in input_sequence])
input_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(input_sequence,
                                                               padding=padding_type,
                                                               #truncating=trunc_type,
                                                               maxlen=max_sequence_len)


########## create training data
predictors, labels = input_sequence_padded[:,:-1], input_sequence_padded[:,-1]

labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)



model_bilstm=  tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, embedding_dim, input_length = max_sequence_len-1),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)),
    tf.keras.layers.Dense(total_words, activation='softmax'),
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

#loss_model = "binary_crossentropy"
loss_model = 'categorical_crossentropy'
#loss_model = 'mean_absolute_error'

metrics_to_be_used = ["accuracy"]

model.compile(optimizer=optimizer,loss=loss_model, metrics=metrics_to_be_used)
model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-4, verbose=1)

EarlyStoppingMonitory = EarlyStopping(patience=4)



history = model.fit(x=predictors,y=labels,
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



############ define a function to make prediction
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list],
                                                               padding=padding_type,
                                                               #truncating=trunc_type,
                                                               maxlen=max_sequence_len-1)

        # Get the predictions
        predictions = model.predict(token_list)

        # Get the index with maximum prediction value
        predicted = np.argmax(predictions)

        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


input_text = "In the hustle and bustle of ipoti"
generated_text = generate_text(input_text, 50, model, max_sequence_len)
print(generated_text)
