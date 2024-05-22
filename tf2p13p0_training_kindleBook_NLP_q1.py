# Book: Tensorflow developer certificate exam practice tests 2024 made easy (https://read.amazon.in/?asin=B0CV19NDD1&ref_=kwl_kr_iv_rec_1)
# slightly modifying the question, take language sentence from here and classify
# https://github.com/elisiojsj/NLP-language-classification-generation-translation/tree/master/data/text_samples

import json
import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.models import load_model
from keras.src.preprocessing.text import Tokenizer

from keras.src.utils import pad_sequences
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def get_lines_from_text_file(filepath):
    file = open(filepath, 'r', encoding="utf8" )
    lines_raw = file.readlines()
    lines = []
    for l in lines_raw:
        if l != "\n":
            lines.append(l)
    return lines



en_path = "datasets/languages/english.txt"
sp_path = "datasets/languages/spanish.txt"
ge_path = "datasets/languages/german.txt"
fr_path = "datasets/languages/french.txt"

map_dict = {0:"English", 1:"Spanish", 2:"German", 3:"French"}

# get all the data in one single list
x = []
y = []

#English
lines = get_lines_from_text_file(en_path)
for l in lines:
    x.append(l)
    y.append(0)

#Spanish
lines = get_lines_from_text_file(sp_path)
for l in lines:
    x.append(l)
    y.append(1)

#German
lines = get_lines_from_text_file(ge_path)
for l in lines:
    x.append(l)
    y.append(2)

#French
lines = get_lines_from_text_file(fr_path)
for l in lines:
    x.append(l)
    y.append(3)


#generate dataframe
df = pd.DataFrame(x,columns=["text"])


##### process data

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
df['label']=y



# stratified train test
#df["temp"]=df["graduate"].astype(str) + df["Qualification"].astype(str)
df["temp"]=df["label"].astype(str)

#train test split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["temp"])

train_inputs = train_set.drop(['temp','label'], axis=1)
train_labels = train_set['label'].copy()

test_inputs = test_set.drop(['temp','label'], axis=1)
test_labels = test_set['label'].copy()




# preprocess and then prepare model

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
    tf.keras.layers.Dense(4, activation='softmax'),
])


model_cnn=  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model_lstm=  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])


model_bilstm=  tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
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
#loss_model = 'categorical_crossentropy'
#loss_model = 'mean_absolute_error'

loss_model = tf.keras.losses.sparse_categorical_crossentropy

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
                    epochs = 100,
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

# converting y_pred from one-hot to single value

y_true = np.array(pd.get_dummies(test_labels_array)).astype(int)



#Generating the confusion matrix
print("Confusion matrix")
for i in range(4):
    print("For {}".format(map_dict[i]))
    eval = confusion_matrix(y_true[:,i], y_pred[:,i])
    print("######### Confusion Matrix #############")
    print(eval)
    print()

print("############# Classification Report##############")
for i in range(4):
    print("For {}".format(map_dict[i]))
    class_names = [ 'Wrong', 'Correct']
    print("Classification Report")
    print(classification_report(y_true[:,i], y_pred[:,i], target_names=class_names))
    print()


filepath = "NLP_language_cat.h5"
model.save(filepath)

saved_model = load_model(filepath)

saved_model.summary()
