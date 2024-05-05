# Types of classification
    # Binary Classification --> only two classes like span or not_spam --> Sigmoid Activation Function
    # Multi Class Classification --> multiple classes like breeds of a dog --> Softmax Activation Function
    # Multi Label Classification --> multiple labels like Avenger movie is scifi, superhero, epic, fantasy --> Sigmoid Activation Function

# Confusion Matrix for binary classification
    # True Positive, True Negative, False Positive, False Negative
    # Accuracy --> how many are labelled correctly (TP+TN)/(TP+TN+FP+FN)
    # Recalll --> Out of actual(ground truth) positive labels how many are predicted positive by model --> (TP)/(TP+FN)
    # Precision --> Out of positive classes predicted by model how many are actually positive --> (TP)/(TP+FP)
    # F1 Score --> Harmonic mean of precision and recall --> 2*Precision*Recall(Precision+Recall)


################################ Import Libraries ##########################
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


pd.set_option('display.max_columns', 500)



############################## Variables ####################################

output_column = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
columns_to_drop = ["ID"]



################# Read data as pandas dataframe ############################
df=pd.read_csv('datasets/paper_abstracts/train.csv') #https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset
df_to_predict=pd.read_csv('datasets/paper_abstracts/test.csv') #https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset
cols = df.columns
print("All numeric columns are: {}".format(list(df.select_dtypes(include="number").columns)))
print("All categorical columns are: {}".format(list(df.select_dtypes(include="object").columns)))
print("Selected output column are: {}".format(output_column))
print("Columns to drop are: {}".format(columns_to_drop))

print()

print("Details of the dataset")
print(df.info())

################### Data Preprocessing #########################################
# Create new columns as required
#df['Year_of_Birth'] = pd.DatetimeIndex(df['Date_Of_Birth']).year
#df["year"]=df["year"].apply(str)



#drop columns that are not required
df.drop(columns_to_drop,inplace=True, axis=1)
df_to_predict.drop(columns_to_drop,inplace=True, axis=1)




################################## Processing NLP Text #############################################################



df['TITLE'] = df['TITLE'].str.lower()
df['ABSTRACT'] = df['ABSTRACT'].str.lower()

df_to_predict['TITLE'] = df_to_predict['TITLE'].str.lower()
df_to_predict['ABSTRACT'] = df_to_predict['ABSTRACT'].str.lower()


def remove_special_characters(text):
    text = re.sub(r'http\S+', ' ', text )
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bhttps?://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d', ' ', text)  # Corrected line
    text= re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    return text

df['TITLE'] = df['TITLE'].apply(remove_special_characters)
df['ABSTRACT'] = df['ABSTRACT'].apply(remove_special_characters)

df_to_predict['TITLE'] = df_to_predict['TITLE'].apply(remove_special_characters)
df_to_predict['ABSTRACT'] = df_to_predict['ABSTRACT'].apply(remove_special_characters)




#df['TITLE'] = df['TITLE'].apply(word_tokenize)
#df['ABSTRACT'] = df['ABSTRACT'].apply(word_tokenize)





################################################ Stratified train test split ######################################
# stratified train test
#df["temp"]=df["graduate"].astype(str) + df["Qualification"].astype(str)
#df["temp"]=df["cat1"].astype(str) + df["cat2"].astype(str) + df["cat3"].astype(str)

#train test split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #stratify=df["temp"],

print("Training set size: {}".format(train_set.shape))
print("Testing set size: {}".format(test_set.shape))

#train_set.drop("temp", axis=1, inplace=True)
#test_set.drop("temp", axis=1, inplace=True)
#df.drop("temp", axis=1, inplace=True)

# Prepare data for model
#divide train and test sets into inputs and labels
train_inputs = train_set.drop(output_column, axis=1)
train_labels = train_set[output_column].copy()

test_inputs = test_set.drop(output_column, axis=1)
test_labels = test_set[output_column].copy()


#################### PreProcess
# train_inputs, train_labels (TITLE, ABSTRACT)
# test_inputs, test_labels
# using NGrams
'''
vec_Title = TfidfVectorizer(ngram_range=(1, 1),
                      min_df=3,
                      max_df=0.9,
                      strip_accents='unicode',
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1,
                      stop_words='english',
                      max_features = 1000)

vec_Abstract = TfidfVectorizer(ngram_range=(1, 1),
                      min_df=3,
                      max_df=0.9,
                      strip_accents='unicode',
                      use_idf=1,
                      smooth_idf=1,
                      sublinear_tf=1,
                      stop_words='english',
                      max_features = 1000)

train_inputs_vector_Title = vec_Title.fit_transform(train_inputs['TITLE'])
train_inputs_vector_Abstract = vec_Abstract.fit_transform(train_inputs['ABSTRACT'])

test_inputs_vector_Title = vec_Title.transform(test_inputs['TITLE'])
test_inputs_vector_Abstract = vec_Abstract.transform(test_inputs['ABSTRACT'])

df_to_predict_vector_Title = vec_Title.transform(df_to_predict['TITLE'])
df_to_predict_vector_Abstract = vec_Abstract.transform(df_to_predict['ABSTRACT'])


#train_inputs_array = train_inputs_vector_Abstract
train_inputs_array = hstack((train_inputs_vector_Title, train_inputs_vector_Abstract))
train_labels_array = np.array(train_labels)

#test_inputs_array = test_inputs_vector_Abstract
test_inputs_array = hstack((test_inputs_vector_Title, test_inputs_vector_Abstract))
test_labels_array = np.array(test_labels)

df_to_predict_array = hstack((df_to_predict_vector_Title, df_to_predict_vector_Abstract))


#train_inputs_array = train_inputs_array.toarray()
#test_inputs_array = test_inputs_array.toarray()

#train_inputs_array_3d = train_inputs_array.reshape(train_inputs_array.shape[0], train_inputs_array.shape[1], 1)
#test_inputs_array_3d = test_inputs_array.reshape(test_inputs_array.shape[0], test_inputs_array.shape[1], 1)

'''

# Using Embedding

# remove special characters and use lowercase


# define parameters
vocab_size = 10000
embedding_dim_title = 128
embedding_dim_abstract = 1024
max_length_title = 256
max_length_abstract = 2048
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
num_epochs = 500


#get list of all titles and abstracts
title_list = []
abstract_list = []
test_title_list = []
test_abstract_list = []

for v in train_inputs.iterrows():
    title_list.append(v[1][0])
    abstract_list.append(v[1][1])

for v in test_inputs.iterrows():
    test_title_list.append(v[1][0])
    test_abstract_list.append(v[1][1])

# word embedding
tokenizer_title = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_tok, num_words = vocab_size) # define out of vocabulary words token
tokenizer_abstract = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_tok, num_words = vocab_size) # define out of vocabulary words token

tokenizer_title.fit_on_texts(title_list)
print("Showing tokenizers - Words")
print(tokenizer_title.word_index)
print("\n\n\n")

tokenizer_abstract.fit_on_texts(abstract_list)
print("Showing tokenizers - Words")
print(tokenizer_abstract.word_index)
print("\n\n\n")

# convert to sequence
sequence_title = tokenizer_title.texts_to_sequences(title_list)
sequence_abstract = tokenizer_abstract.texts_to_sequences(abstract_list)

test_sequence_title = tokenizer_title.texts_to_sequences(test_title_list)
test_sequence_abstract = tokenizer_abstract.texts_to_sequences(test_abstract_list)

# pad and truncate
padded_sequence_title = tf.keras.preprocessing.sequence.pad_sequences(sequence_title, padding=padding_type, truncating=trunc_type, maxlen=max_length_title)
padded_sequence_abstract = tf.keras.preprocessing.sequence.pad_sequences(sequence_abstract, padding=padding_type, truncating=trunc_type, maxlen=max_length_abstract)

padded_test_sequence_title = tf.keras.preprocessing.sequence.pad_sequences(test_sequence_title, padding=padding_type, truncating=trunc_type, maxlen=max_length_title)
padded_test_sequence_abstract = tf.keras.preprocessing.sequence.pad_sequences(test_sequence_abstract, padding=padding_type, truncating=trunc_type, maxlen=max_length_abstract)

train_data_array = np.concatenate((padded_sequence_title, padded_sequence_abstract), axis=1)
train_labels_array = np.array(train_labels)

test_data_array = np.concatenate((padded_test_sequence_title, padded_test_sequence_abstract), axis=1)
test_labels_array = np.array(test_labels)

###################################### Prepare Tensorflow Model ####################################################
model_multiLabelClassification_LSTM = tf.keras.Sequential([
    #tf.keras.layers.Dense(21, activation='relu', input_shape=(13,)),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Embedding(vocab_size, embedding_dim_title+embedding_dim_abstract, input_length = max_length_title+max_length_abstract),
    #tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2),
    #tf.keras.layers.Dropout(0.3),
    #tf.keras.layers.LSTM(5),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(30, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid'),
])

model_multiLabelClassification_Dense = tf.keras.Sequential([
    #tf.keras.layers.Dense(21, activation='relu', input_shape=(13,)),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Embedding(vocab_size, embedding_dim_title+embedding_dim_abstract, input_length = max_length_title+max_length_abstract),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(50, activation='relu'),
    #tf.keras.layers.LSTM(15),
    tf.keras.layers.Dropout(0.3),
    #tf.keras.layers.LSTM(5),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid'),
])


model = model_multiLabelClassification_LSTM
#model = model_multiLabelClassification_Dense

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001)
#optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)

#loss_model = "binary_crossentropy"
loss_model = 'categorical_crossentropy'
#loss_model = 'mean_absolute_error'

metrics_to_be_used = ["accuracy"]

model.compile(optimizer=optimizer,loss=loss_model, metrics=metrics_to_be_used)


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
model.summary()
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



####################################################### Evaluate Model #######################################
def eval_model(model):
    return model.evaluate(test_data_array, test_labels_array)

all_models = [model]
for m in all_models:
    eval_model(m)





############################################ Post Process the data ############################################

y_pred = model.predict(test_data_array)
y_pred = np.round(y_pred).astype('int')

print(y_pred)


#Generating the confusion matrix
for i in range(6):
    print("Confusion matrix for {}th category".format(i))
    eval = confusion_matrix(test_labels_array[:,i], y_pred[:,i])
    print("######### Confusion Matrix #############")
    print(eval)
    print()

print("############# Classification Report##############")
class_names = [ 'Wrong', 'Correct']
for i in range(6):
    print("Classification Report for {}th category".format(i))
    print(classification_report(test_labels_array[:,i], y_pred[:,i], target_names=class_names))
    print()

print("accuracy: {}".format(sklearn.metrics.accuracy_score(test_labels_array, y_pred)))
#print(sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred))
#print("mean_absolute_error: {}".format(sklearn.metrics.mean_absolute_error(test_labels_array, y_pred)))

plt.figure()
plt.plot(test_labels_array)
plt.plot(y_pred)
plt.show()


########################################## Save the model ######################################################
# model.save("model_name.keras")
