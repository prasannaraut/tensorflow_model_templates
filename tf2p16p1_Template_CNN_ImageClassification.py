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


pd.set_option('display.max_columns', 500)


################################## Load image data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Number of training images : {}".format(len(x_train)))
print("Number of test images : {}".format(len(x_test)))

### Reshaping
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

y_train_original = y_train

# Normalize the pixel value
x_train = x_train/255
x_test = x_test/255


# convert y values to one-hot encoding (categorical)
n_labels = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train,n_labels)
y_test = tf.keras.utils.to_categorical(y_test,n_labels)


################################################################################
# plot images

nums = [10,23,24,66,533,2323,12,124]
figure = plt.figure(figsize=(12,12))

i=0
for n in nums:
    data = x_train[n].reshape(28, 28)
    ax = figure.add_subplot(4,2, i+1, xticks=[], yticks=[])
    ax.imshow(data, cmap='gray')
    ax.set_title(y_train_original[n])

    i = i+1
figure.show()

'''
############ Reading images from dictionary and preprocess

# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# Import data from directories and turn it into batches

train_dir="path_of_directory"
val_dir="path_of_directory"
test_dir="path_of_directory"

train_data = train_datagen.flow_from_directory(train_dir,
    batch_size=64, # number of images to process at a time
    target_size=(224,224), # convert all images to be 224 x 224
    class_mode="categorical")
valid_data = valid_datagen.flow_from_directory(val_dir,
    batch_size=64,
    target_size=(224,224),
    class_mode="categorical")
test_data = test_datagen.flow_from_directory(test_dir,
    batch_size=64,
    target_size=(224,224),
    class_mode="categorical")
'''

# image pre-processing with ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    interpolation_order=1,
    dtype=None
)


# perform transformations
n=10

original_image = x_train[n].reshape(28, 28,1)

figure = plt.figure(figsize=(12,12))
data = x_train[n].reshape(28, 28)
ax = figure.add_subplot(2,2, 1, xticks=[], yticks=[])
ax.imshow(original_image, cmap='gray')
ax.set_title("Original Image")

transform_parameters ={
    'theta':0,
    'tx':0,
    'ty':0,
    'sheer':0,
    #'zx':0,
    #'zy':0,
    'flip_horizontal':0,
    'flip_vertical':0,
    #'channel_shift_intensity':0.9,
    #'brightness':0.5
}
modified_image = datagen.apply_transform(original_image, transform_parameters)


ax = figure.add_subplot(2,2, 2, xticks=[], yticks=[])
ax.imshow(modified_image, cmap='gray')
ax.set_title("Modified Image")

tf_rand = datagen.get_random_transform((28,28,1),seed=33)
#modified_image_rand = datagen.random_transform(original_image, seed=43)
modified_image_rand = datagen.apply_transform(original_image, tf_rand)
ax = figure.add_subplot(2,2, 3, xticks=[], yticks=[])
ax.imshow(modified_image_rand, cmap='gray')
ax.set_title("Random Modified Image")

figure.show()

################################## Image resizing

n=10

original_image = x_train[n].reshape(28, 28,1)

figure = plt.figure(figsize=(12,12))
ax = figure.add_subplot(2,2, 1, xticks=[], yticks=[])
ax.imshow(original_image, cmap='gray')
ax.set_title("Original Image Before Resizing")

x_train_reshaped = tf.keras.preprocessing.image.smart_resize(
    x_train,
    (64,64),
    interpolation='bilinear',
    data_format='channels_last',
    backend_module=None
)

resized_image = x_train_reshaped[n]

ax = figure.add_subplot(2,2, 2, xticks=[], yticks=[])
ax.imshow(resized_image, cmap='gray')
ax.set_title("Resized Image")
figure.show()



################################################################# Build Sequencial model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])



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



history = model.fit(x=x_train,y=y_train,
                    #validation_data = (x_test,y_test),
                    validation_split=0.2,
                    epochs = 10,
                    shuffle = True,
                    callbacks=[reduce_lr,EarlyStoppingMonitory])

variables_for_plot = ["loss"] + metrics_to_be_used

model.summary()

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


### Evaluate the model
print("Test score")
score = model.evaluate(x_test,y_test)

#### predict for test images

n = 10
output = model.predict(x_test[n].reshape(1,28,28,1)).argmax()
print("Prediction class for {}th image in test set is {}".format(n, output))


######################### Checking what is the output from a layer
desiredLayer=0
desiredOutput = model.layers[desiredLayer].output
newModel = tf.keras.Model(model.inputs, desiredOutput)

nth_image = 10
desiredOutputData = newModel.predict(x_test[nth_image].reshape(1,28,28,1))

plt.imshow(x_train[nth_image].reshape(28,28), cmap='gray')
plt.title("Original image considered")
plt.show()

figure = plt.figure(figsize=(12,12))
for i in range(1,4):
    data = desiredOutputData[0,:,:,i]
    ax = figure.add_subplot(2,2, i, xticks=[], yticks=[])
    ax.imshow(data, cmap='gray')
    ax.set_title("Conv2D output from {}th channel".format(i))

figure.show()
