#numpy==1.24.3
#pandas==2.0.3
#Pillow==10.0.0
#scipy==1.10.1
#tensorflow==2.13.0
#tensorflow-datasets==4.9.2
#matplotlib=3.7.2
#scikit-learn=1.3.0
#nltk=3.8.1

######## Improving the model
#Early stopping - callbacks
#Number of hidden layers and neurons per layer
#More Epochs for training
#Different optiimizes, usually Adam is best, for very deep neural networks such as RNN RMSprop works well
#Changing learning rate, use learning rate scheduler

# CNN - Kernals,Valid/Same Padding, Max/Average/GlobalMax/GlobalAverage Pooling, Strides
# CNN - Translational Invariance

# For image classification we typically use 2Dconvolution, for audio processing we use 1Dconvolution, and for videos we use 3D convolution

# Output feature size = ((W-F+2P)/S)+1
#   W = size of input image, F = Filter size, P=Padding, S=Stride

# try to capture,
#   image data augmentation techniques - done
#   See how CNN is learning about images in each layer - done
#   loading pre-trained CNN models - in next Template of transfer learning
#   fine tunning with early stopping and learning rate scheduler - done

# how to reduce overfitting
#   reduce number of epochs, use callbacks to stop training early, basically track training accuracy and validation accuracy both should be similar
#   use simple models, big and complex models capture the noise from input data
#   use L1(lasso regression) or L2 regularization, L1 will cause some of the coefficient to become 0, the penalty term L2 reduces the magnitude of coefficient preventing them from becoming too large
#       tf.keras.layers.Dense(1050, activation="relu", kernel_regularizer=regularizers.l2(0.01))
#   use dropouts
#   adjust learning rate
#   perform error analysis, check the cases where model misclassifid information
#   use data augmentation




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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, ResNet50


pd.set_option('display.max_columns', 500)




################################## Load image data
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

############################################## Load Images
# define saling

batch_size = 32
img_height = 224
img_width = 224


# Load images
train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=3,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

valid_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=3,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


print("Class names are: {} \n\n ".format(train_data.class_names))
classes = train_data.class_names
n_classes = len(classes)


#################################################################### Modelling from Scratch
model_CNN_scratch = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])


########### VGG16 Model, MobileNet,
# Instantiate the VGG16 model
vgg16 = VGG16(weights='imagenet', include_top = False, input_shape=(224,224,3))
mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


imported = vgg16

# Freeze all the layers in  Model
for layer in imported.layers:
    layer.trainable = False


# Unfreeze last 3 layers in  Model
for layer in imported.layers[-3: ]:
    layer.trainable = True

# create a new model by adding on top of vgg16
imported_newModel = tf.keras.models.Sequential()
imported_newModel.add(tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
imported_newModel.add(imported)
imported_newModel.add(tf.keras.layers.Flatten())
imported_newModel.add(tf.keras.layers.Dense(1024, activation='relu'))
imported_newModel.add(tf.keras.layers.Dropout(0.5))
imported_newModel.add(tf.keras.layers.Dense(n_classes, activation='softmax'))


model = imported_newModel
#model=model_CNN_scratch




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

#loss_model=tf.keras.losses.categorical_crossentropy # use if output is one hot encoder
loss_model=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # use if output is single digit specifying the class name

metrics_to_be_used = ["accuracy"]

model.compile(optimizer=optimizer,loss=loss_model, metrics=metrics_to_be_used)


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-4, verbose=1)

EarlyStoppingMonitory = EarlyStopping(patience=4)


#y=y_train,
# validation_data = (x_test,y_test),
history = model.fit(x=train_data,
                    validation_data=valid_data,
                    #validation_split=0.2,
                    epochs = 1,
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

output = model.predict(valid_data)
