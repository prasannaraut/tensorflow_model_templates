# grey scale model

from keras import Sequential
from keras.datasets import mnist
from keras.saving import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, ResNet50

def my_model():
    dataset = mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Normalize the input data
    #x_train =  x_train.astype('float32')/255.0
    #x_test = x_test.astype('float32')/255.0

    img_height = x_train[0].shape[0]
    img_width = x_train[0].shape[1]
    img_height_new = 75
    img_width_new = 75

    n_classes = len(set(y_train))

    train_data = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
    train_labels = y_train
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomHeight(0.2),
            tf.keras.layers.RandomWidth(0.2),
            tf.keras.layers.Resizing(img_height_new, img_width_new),
        ]
    )

    model_CNN_scratch = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height_new, img_height_new)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    '''
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height_new, img_width_new, 1))
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height_new, img_width_new, 1))
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height_new, img_width_new, 1))

    imported = inceptionv3

    # Freeze all the layers in  Model
    for layer in imported.layers:
        layer.trainable = False

    # Unfreeze last 3 layers in  Model
    for layer in imported.layers[-3:]:
        layer.trainable = True

    # create a new model by adding on top of vgg16
    imported_newModel = tf.keras.models.Sequential()
    imported_newModel.add(data_augmentation)
    imported_newModel.add(tf.keras.layers.Rescaling(1. / 255, input_shape=(img_width_new, img_width_new, 1)))
    imported_newModel.add(imported)
    imported_newModel.add(tf.keras.layers.Flatten())
    imported_newModel.add(tf.keras.layers.Dense(1024, activation='relu'))
    imported_newModel.add(tf.keras.layers.Dropout(0.5))
    imported_newModel.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    '''
    
    #model = imported_newModel
    model=model_CNN_scratch


    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)

    # loss_model = "binary_crossentropy"
    # loss_model = 'categorical_crossentropy'
    # loss_model = 'mean_absolute_error'

    # loss_model=tf.keras.losses.categorical_crossentropy # use if output is one hot encoder
    loss_model = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),  # use if output is single digit specifying the class name

    metrics_to_be_used = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss_model, metrics=metrics_to_be_used)

    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=3, min_lr=1e-8, verbose=1)

    EarlyStoppingMonitory = EarlyStopping(patience=4)

    # y=y_train,
    # validation_data = (x_test,y_test),
    history = model.fit(x=train_data, y=train_labels,
                        validation_data=(x_test, y_test),
                        #validation_split=0.2,
                        epochs=3,
                        shuffle=True,
                        callbacks=[reduce_lr, EarlyStoppingMonitory])

    print(model.summary())

    variables_for_plot = ["loss"] + metrics_to_be_used

    for var in variables_for_plot:
        loss_train = history.history["{}".format(var)]
        loss_val = history.history['val_{}'.format(var)]
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, loss_train, 'g', label='Training loss')
        plt.plot(epochs, loss_val, 'b', label='Validation loss')
        plt.title('{}'.format(var))
        plt.xlabel('Epochs')
        plt.ylabel(var)
        plt.legend()
        plt.show()

    return model


if __name__ == '__main__':
    my_model = my_model()
    filepath = "grayscale_model_1.h5"
    my_model.save(filepath)

    saved_model = load_model(filepath)

    saved_model.summary()
