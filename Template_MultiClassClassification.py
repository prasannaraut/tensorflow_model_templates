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


pd.set_option('display.max_columns', 500)

############################## Variables ####################################
output_column_from_input_dataset = ["category"]
output_column = ["cat1","cat2","cat3"]
columns_to_drop = ["category"]



################# Read data as pandas dataframe ############################
df=pd.read_csv('datasets/iris.csv') #https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
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

# Encode the categorical output column
encoder = LabelEncoder()
Y = np.array(df[output_column_from_input_dataset[0]])
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y, num_classes=3)
df["cat1"]=dummy_y[:,0]
df["cat2"]=dummy_y[:,1]
df["cat3"]=dummy_y[:,2]

# Impute the data as required
# Imputing for numeric columns
for col in df.select_dtypes(include="number").columns:
    val = df[col].median()
    #val =df[col].mean()
    df[col].fillna(val, inplace=True)

# Imputing for categorical columns
for col in df.select_dtypes(include="object").columns:
    val = df[col].mode()
    df[col].fillna(val, inplace=True)


#drop columns that are not required
df.drop(columns_to_drop,inplace=True, axis=1)

#convert numeric colums to float32
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].astype(float)

#add a small number to numeric colums to avoid divide by 0 error
for col in df.select_dtypes(include="number").columns:
    if (col not in output_column):
        df[col] = df[col]+1e-10


###################### Data visuzlization #######################
#extract informatoin from dataset
print("Description of the dataset")
print(df.describe())
print()

print("Categorical column all - value counts")
#print(df.select_dtypes(include="object").value_counts())
print()


print("Value counts for each inidividual categorical columns")
for col in df.select_dtypes(include="object").columns:
    print(df[col].value_counts())




#visualize data
df.hist(bins=50, figsize=(12,8))
plt.show()


#check for correlations
corr_matrix = df.select_dtypes(include="number").corr()
for col in output_column:
    print("For column {}".format(col))
    print(corr_matrix[col].sort_values(ascending=False))
    print()


#create scatter matrix
attributes = df.columns
scatter_matrix(df[attributes], figsize=(12,8))
plt.show()


################################################ Stratified train test split ######################################
# stratified train test
#df["temp"]=df["graduate"].astype(str) + df["Qualification"].astype(str)
df["temp"]=df["cat1"].astype(str) + df["cat2"].astype(str) + df["cat3"].astype(str)

#train test split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #stratify=df["temp"],

print("Training set size: {}".format(train_set.shape))
print("Testing set size: {}".format(test_set.shape))

train_set.drop("temp", axis=1, inplace=True)
test_set.drop("temp", axis=1, inplace=True)
df.drop("temp", axis=1, inplace=True)

# Prepare data for model
#divide train and test sets into inputs and labels
train_inputs = train_set.drop(output_column, axis=1)
train_labels = train_set[output_column].copy()

test_inputs = test_set.drop(output_column, axis=1)
test_labels = test_set[output_column].copy()


################################################## Some custom classes and functions to prepare data for tensorflow model #################################
# Write custom class to detect Cluster Similarity

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, n_init=3, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# custom functions for ratio pipeline

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


# log pipeline

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# cluster_simil
cluster_simil = ClusterSimilarity(n_clusters=3, n_init=3, gamma=1., random_state=42)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing_complex = ColumnTransformer([
    #("Age_by_Experience", ratio_pipeline(), ["Fare", "Pclass"]),
    #("AgeBySibSp", ratio_pipeline(), ["Age", "SibSp"]),
    #("AgeByParch", ratio_pipeline(), ["Age", "Parch"]),
    #("log", log_pipeline, ["Age", "Fare"]),
    #("geo", cluster_simil, ["Year_of_Birth"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),

],
remainder=default_num_pipeline)



######################################### Prepare training data
#prepare trining data
train_inputs_processed = preprocessing_complex.fit_transform(train_inputs)
test_inputs_processed = preprocessing_complex.transform(test_inputs)

print("Training data input shape: {}".format(train_inputs_processed.shape))
print("Testing data input shape: {}".format(test_inputs_processed.shape))
print("Preprocessing complex class features: {}".format(preprocessing_complex.get_feature_names_out()))


train_inputs_array = train_inputs_processed
train_labels_array = np.array(train_labels)


test_inputs_array = test_inputs_processed
test_labels_array = np.array(test_labels)


print(train_inputs_array)
print(train_labels_array)

print(test_inputs_array)
print(test_labels_array)


'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(21, activation='relu', input_shape=(13,)))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
#model.add(tf.keras.layers.Dense(1024, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))
'''

###################################### Prepare Tensorflow Model ####################################################
model_multiClassClassification = tf.keras.Sequential([
    #tf.keras.layers.Dense(21, activation='relu', input_shape=(13,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])



model = model_multiClassClassification

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


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-4, verbose=1)

history = model.fit(x=train_inputs_array,y=train_labels_array,
                    validation_data = (test_inputs_array,test_labels_array),
                    epochs = 500,
                    shuffle = True,
                    callbacks=[reduce_lr])
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
    return model.evaluate(test_inputs_array, test_labels_array)

all_models = [model]
for m in all_models:
    eval_model(m)





############################################ Post Process the data ############################################

y_pred = model.predict(test_inputs_array)
y_pred = np.round(y_pred).astype('int')

print(y_pred)

#Generating the confusion matrix
for i in range(3):
    print("Confusion matrix for {}th category".format(i))
    eval = confusion_matrix(test_labels_array[:,i], y_pred[:,i])
    print("######### Confusion Matrix #############")
    print(eval)
    print()

print("############# Classification Report##############")
class_names = [ 'Wrong', 'Correct']
for i in range(3):
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
