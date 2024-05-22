# you have been given two arrays: x_array and y_array, each containing a number of interge values
# x contain inputs, y contain outputs

# write deep learning model

import numpy as np
from keras import Sequential
from keras.saving import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

def regression_model():
    x_array = np.array([-1, 0, 1,2,3,4,5,6], dtype=int)
    y_array = np.array([-5, 5, 15, 25, 35, 45, 55, 65], dtype=int)

    model = Sequential([
        tf.keras.layers.Dense(32, input_dim=1, activation='linear'),
        tf.keras.layers.Dense(8, activation='linear'),
        tf.keras.layers.Dense(1,  activation='linear'),
    ])

    #compile model

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)

    # loss_model = "mean_absolute_percentage_error"
    loss_model = 'mean_absolute_error'
    #loss_model = 'categorical_crossentropy'
    # loss_model = tf.keras.losses.CosineSimilarity

    metrics_to_be_used = [tf.keras.metrics.RootMeanSquaredError()]
    metrics_to_be_used_name = ["root_mean_squared_error"]
    #metrics_to_be_used = ["accuracy"]
    #metrics_to_be_used_name = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss_model, metrics=metrics_to_be_used)

    #Train model

    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=1e-12, verbose=1)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 15)

    history = model.fit(x=x_array, y=y_array,
                        # validation_data = (test_inputs_array,test_labels_array),
                        validation_split=0.2,
                        epochs=5000,
                        shuffle=True,
                        callbacks=[reduce_lr, early_stop])

    model.summary()



    for var in metrics_to_be_used_name:
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




#if __name__ == '_main_':
my_model = regression_model()
filepath = 'regression_model_1.h5'
my_model.save(filepath)

saved_model = load_model(filepath)

print(saved_model.summary())

#test model

x_test = np.array([7, 8, 9, 10])
y_test = np.array([75, 85, 95, 105])
predictions = saved_model.predict(x_test)

for i in range(len(x_test)):
    print("x = {:.0f}, expected y={:.0f}, predicted y={:.0f}".format(x_test[i], y_test[i], predictions[i][0]))

test_loss = saved_model.evaluate(x_test, y_test)
print("Test loss: {}".format(test_loss))
