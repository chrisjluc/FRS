import consts
import image_processing as ip
from storage import Writer, Reader

import numpy as np


class Model(object):

    def __init__(self, name, ids, X_train=None, Y_train=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.ids = ids
        self.name = name
        self.nb_classes = len(ids) if ids else 0
        self.id_to_idx = dict([(x[1], x[0]) for x in enumerate(ids)]) if ids else None

    def _reshape_data(self, X, y):
        from keras.utils import np_utils
        return  (X.reshape(X.shape[0], 1, self.input_shape[1], self.input_shape[2]),
        np_utils.to_categorical(y, self.nb_classes))

    def train(self):
        raise NotImplementedError

    def get_activations(self, data):
        raise NotImplementedError

    def save_activations(self):
        self._save_activations()

    def _save_activations(self):
        activations = self.get_activations(self.X_train)
        w = Writer()
        w.save_activations(activations, self.name)

    def save(self):
        w = Writer()
        w.save_model(self.model, self.name)

    def load(self):
        r = Reader()
        self.model = r.get_model(self.name)


class AutoEncoderModel(Model):

    def __init__(self, name, input_size=None, encoding_size=None, X_train=None):
        super(AutoEncoderModel, self).__init__(
                name,
                None,
                X_train,
                None
        )
        self.input_size = input_size
        self.encoding_size = encoding_size
        if input_size is not None and encoding_size is not None:
            self.autoencoder, self.encoder = _initAutoEncoder(self.input_size, self.encoding_size)

    def train(self):
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        self.autoencoder.fit(
                self.X_train,
                self.X_train,
                nb_epoch=consts.sae_nb_epoch,
                batch_size=consts.sae_batch_size,
                shuffle=True,
                validation_split=consts.sae_validation_split)

    def get_activations(self, data):
        activations = self.encoder.predict(data)
        assert activations.shape[1] == self.encoding_size
        return activations

    def save(self):
        w = Writer()
        w.save_model(self.autoencoder, self.name + consts.autoencoder_ext)
        w.save_model(self.encoder, self.name + consts.encoder_ext)

    def load(self):
        r = Reader()
        self.autoencoder = r.get_model(self.name + consts.autoencoder_ext)
        self.encoder = r.get_model(self.name + consts.encoder_ext)


def _initAutoEncoder(input_size, encoding_size):
    from keras.models import Model
    from keras.layers import Dense, Input

    _input = Input(shape=(input_size,))
    encoded = Dense(encoding_size, activation='sigmoid')(_input)
    decoded = Dense(input_size, activation='sigmoid')(encoded)

    autoencoder = Model(input=_input, output=decoded)
    encoder = Model(input=_input, output=encoded)

    return autoencoder, encoder


class CNNModel(Model):

    def _train(self):
        from keras.optimizers import SGD

        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model.fit(self.X_train, self.Y_train, batch_size=consts.cnn_batch_size, nb_epoch=consts.cnn_nb_epoch,
                verbose=1, shuffle=True, validation_split=consts.cnn_validation_split)

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
        self.model.fit(self.X_train, self.Y_train, batch_size=consts.cnn_batch_size, nb_epoch=consts.cnn_nb_epoch,
                verbose=1, shuffle=True, validation_split=consts.cnn_validation_split)

    def score(self, user_id, image):
        #X_test, _ = self._process_data([[image]])
        X_test, _ = self._reshape_data(X_test, _)
        proba = self.model.predict_proba(X_test)
        return proba[0][self.id_to_idx[user_id]]

    def train(self):
        self.X_train, self.Y_train = self._reshape_data(self.X_train, self.Y_train)
        self._train()

    def save_activations(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.input_shape[1], self.input_shape[2])
        self._save_activations()

    def get_activations(self, data):
        activations = None
        batch_size = consts.cnn_activation_batch_size
        for i in range(int(data.shape[0] / batch_size) + 1):
            a = self._get_activations_batch(data[batch_size * i:batch_size * (i + 1)])
            if activations is None:
                activations = a
            else:
                activations = np.concatenate((activations, a))
        return activations

    def _get_activations_batch(self, batch):
        from keras import backend as K
        fn = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-2].output,])
        activations = fn([batch,0])
        return np.array(activations)[0]


class NN1Model(CNNModel):

    def __init__(self, name, ids, X_train=None, Y_train=None):
        super(NN1Model, self).__init__(
                name,
                ids,
                X_train,
                Y_train
        )
        self.input_shape = consts.nn1_input_shape
        self.model = _NN1(self.nb_classes, self.input_shape)


class NN2Model(CNNModel):

    def __init__(self, name, ids, X_train=None, Y_train=None):
        super(NN2Model, self).__init__(
                name,
                ids,
                X_train,
                Y_train
        )
        self.input_shape = consts.nn2_input_shape
        self.model = _NN2(self.nb_classes, self.input_shape)


def _NN1(nb_classes, input_shape):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D((1,1)))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def _NN2(nb_classes, input_shape):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D((1,1)))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

class InvalidModelException(Exception):
    pass

class InvalidDataException(Exception):
    pass
