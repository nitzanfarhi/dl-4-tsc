# resnet model 
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
from utils.utils import save_test_duration

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils import save_logs
from utils.utils import calculate_metrics


class Classifier_2DRESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        self.epoch_num = 5

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return
    
    
    def reset_model(self):
        self.model.load_weights(self.output_directory + 'model_init.hdf5')
        
        
    def build_model(self, input_shape, nb_classes):
        import tensorflow
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Flatten
        from tensorflow.keras.layers import Conv2D, MaxPooling2D

        # Model configuration
        img_width, img_height = input_shape[0], input_shape[1]
        batch_size = 250
        no_epochs = self.epoch_num
        no_classes = nb_classes
        validation_split = 0.2
        verbosity = 1

        # Load MNIST dataset
        #(input_train, target_train), (input_test, target_test) = mnist.load_data()

        # Reshape data 
        #input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
        #input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
        input_shape = (img_width, img_height, 1) 
        # Parse numbers as floats
        #input_train = input_train.astype('float32')
        #input_test = input_test.astype('float32')

        # Convert into [0, 1] range.
        #input_train = input_train / 255
        #input_test = input_test / 255

        # Convert target vectors to categorical targets
        #target_train = tensorflow.keras.utils.to_categorical(target_train, no_classes)
        #target_test = tensorflow.keras.utils.to_categorical(target_test, no_classes)

        # Create the model
        

        import tensorflow.keras as K
        model = K.applications.MobileNetV2(weights=None,input_shape=input_shape,classes = nb_classes)

        """        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(no_classes, activation='softmax'))"""

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=1000)
        self.callbacks = [reduce_lr, model_checkpoint,early_stop]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = self.epoch_num

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        #duration = time.time() - start_time

        #self.model.save(self.output_directory + 'last_model.hdf5')

        #y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
        #                      return_df_metrics=False)

        # save predictions
        #np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        #y_pred = np.argmax(y_pred, axis=1)

        #df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return hist.history['accuracy'][-1]
    
    def fit_generator(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics
    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
