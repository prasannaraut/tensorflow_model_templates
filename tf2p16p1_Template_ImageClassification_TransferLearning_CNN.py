##### Comments





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
from PIL import Image
import matplotlib.image as mpimg
#Import necessary libraries
import os
import random
from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, ResNet50



pd.set_option('display.max_columns', 500)


################################## Load image data
# data is available at https://storage.googleapis.com/x_ray_dataset/dataset.zip

root_dir = "C:/Users/PRAU4KBR/Documents/PyCharm_Projects/TensorFlow_Certificate/datasets/pneumonia_classifier/dataset"

for dirpath, dirnames, filenames in os.walk(root_dir):
    print(f"Directory: {dirpath}")
    print(f"Number of images: {len(filenames)}")
    print()


# view random images
def view_random_images(target_dir, num_images):
    """
    View num_images random images from the subdirectories of
    target_dir as a subplot.
    """
    # Get list of subdirectories
    subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    # Select num_images random subdirectories
    random.shuffle(subdirs)
    selected_subdirs = subdirs[:num_images]
    # Create a subplot
    fig, axes = plt.subplots(1, num_images, figsize=(12,12))
    for i, subdir in enumerate(selected_subdirs):
        # Get list of images in subdirectory
        image_paths = [f for f in os.listdir( os.path.join(target_dir, subdir))]
        # Select a random image
        image_path = random.choice(image_paths)
        # Load image
        image = plt.imread(os.path.join(target_dir, subdir, image_path))
        # Display image in subplot
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(subdir)
        print(f"Shape of image: {image.shape}")
        #width,height, colour chDNNels
    plt.show()


view_random_images(target_dir="C:/Users/PRAU4KBR/Documents/PyCharm_Projects/TensorFlow_Certificate/datasets/pneumonia_classifier/dataset/train",num_images=4)



############################################## Load Images
# define saling

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# setup directories
train_dir = 'C:/Users/PRAU4KBR/Documents/PyCharm_Projects/TensorFlow_Certificate/datasets/pneumonia_classifier/dataset/train'
valid_dir = 'C:/Users/PRAU4KBR/Documents/PyCharm_Projects/TensorFlow_Certificate/datasets/pneumonia_classifier/dataset/val'
test_dir = 'C:/Users/PRAU4KBR/Documents/PyCharm_Projects/TensorFlow_Certificate/datasets/pneumonia_classifier/dataset/test'


train_data = train_datagen.flow_from_directory( train_dir, target_size=(224,224), class_mode="binary")
valid_data = valid_datagen.flow_from_directory( valid_dir, target_size=(224,224), class_mode="binary", shuffle=False)
test_data = test_datagen.flow_from_directory( test_dir, target_size=(224,224), class_mode="binary", shuffle=False)

print("Class Indices are: {} \n\n ".format(train_data.class_indices))

print("Train data filenames")
print(train_data.filenames)
print("\n\n")

print("Train data labels")
print(train_data.labels)
print("\n\n")


#################################################################### Modelling from Scratch
model_CNN_scratch = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=train_data.image_shape, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
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
imported_newModel.add(imported)
imported_newModel.add(tf.keras.layers.Flatten())
imported_newModel.add(tf.keras.layers.Dense(1024, activation='relu'))
imported_newModel.add(tf.keras.layers.Dropout(0.5))
imported_newModel.add(tf.keras.layers.Dense(1, activation='sigmoid'))


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

loss_model = "binary_crossentropy"
#loss_model = 'categorical_crossentropy'
#loss_model = 'mean_absolute_error'

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

output = model.predict(test_data)
