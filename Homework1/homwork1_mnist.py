import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
import datetime
from tensorboard.plugins.hparams import api as hp

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def createModel(l1=128, l2=64, epochs=16):
    """Import data"""

    """Show example image"""

    print(x_train[0])

    plt.imshow(x_train[0], cmap='gray')
    plt.show()
    print("Label: ", y_train[0])
    # Sets up a timestamped log directory.
    logdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        # Don't forget to reshape.
        images = np.reshape(x_train[0:25], (-1, 28, 28, 1))
        tf.summary.image("25 training data examples", images, max_outputs=25, step=0)
    input_dimension = 784  # 28*28

    model = Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # turn image into 1D numpy array

    model.add(Dense(l1,
                    activation='relu',
                    kernel_regularizer=L2(0.02)))  # add L2 regularization

    model.add(Dense(l2,
                    activation='relu'))

    model.add(Dense(10,
                    activation='softmax'))  # final layer is softmax

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    """Summary of model"""

    model.summary()

    """One Hot encode y data"""

    y_One_Hot = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
    print(y_One_Hot)

    """train model

    32 epochs

    batch size of 64

    20% of data for validation
    """
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train, y_One_Hot, epochs=epochs, batch_size=64, validation_split=0.2,
                        callbacks=[tensorboard_callback])

    model.save(os.getcwd() + '/saved_model/saved_model')

    """graph train and validation accuracy and loss

    how to graph from website:
    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

    """

    print(history.history.keys())
    # graph accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
    # graph loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.show()


"""Based on the two graphs the model seems vary accurate because of the high accuracy and low loss for both the train and validation

Now evaluate the test data:
"""


def eval():
    model = tf.keras.models.load_model(os.getcwd() + '/saved_model/saved_model')
    score = model.evaluate(x_test, OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray(), verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    """the low test loss and high accuracy means that the model is accurate

    Now test precision and recall
    """

    # from sklearn.metrics import precision_score, recall_score
    y_predict = model.predict(x_test, verbose=0)
    # classification = []
    # for c in y_predict:
    #     classification.append(np.argmax(c))
    #     p= precision_score(y_test, classification, average='micro') 
    #     print("precision:"+ str(p))
    #     r= recall_score(y_test, classification, average='micro') 
    #     print("recall:"+ str(r))

    # print(y_test)

    arg_y = np.argmax(y_predict, axis=-1)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, arg_y, target_names=[str(i) for i in range(10)]
                                ))

    """confusion matrix"""

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, arg_y)

    """the matrix shows how many times the model predicted the correct class and place all of its results on the graph. Since we see that the diagonal  is much brightewr and much has much larger numbers then we can say that the model is accurate"""

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    """normalized matrix"""

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round_(cm, 2)  # rounded to 2 decimal points for simplicity
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def tune():
    y_One_Hot = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
    Layer1 = hp.HParam('Layer1', hp.Discrete([64, 128, 256]))
    Layer2 = hp.HParam('Layer2', hp.Discrete([16, 32, 64]))
    Epochs = hp.HParam('epochs', hp.Discrete([8, 16, 32]))
    for l1 in Layer1.domain.values:
        for l2 in Layer2.domain.values:
            for e in Epochs.domain.values:
                params = {
                    Layer1: l1,
                    Layer2: l2,
                    Epochs: e
                }
                print(params)
                input_dimension = 784  # 28*28

                model = Sequential()

                model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # turn image into 1D numpy array

                model.add(Dense(l1,
                                activation='relu',
                                kernel_regularizer=L2(0.02)))  # add L2 regularization

                model.add(Dense(l2,
                                activation='relu'))

                model.add(Dense(10,
                                activation='softmax'))  # final layer is softmax

                model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])
                log_dir = "logs/hparam_tuning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                hpcall = hp.KerasCallback(log_dir, params)
                history = model.fit(x_train, y_One_Hot, epochs=e, batch_size=64, validation_split=0.2,
                                    callbacks=[tensorboard_callback, hpcall])


def test(img):
    model = tf.keras.models.load_model(os.getcwd() + '/saved_model/saved_model')
    scores = model.predict(img)
    print("image is ", np.argmax(scores))
