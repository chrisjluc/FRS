import consts
import image_processing as ip
from storage import Writer

import numpy as np


class Model(object):

    def __init__(self, images, ids, name):
        self.X_train = None
        self.Y_train = None
        self.nb_classes = len(ids)
        self.images = images
        self.ids = ids
        self.name = name
        self.id_to_idx = dict([(x[1], x[0]) for x in enumerate(ids)])

    def _reshape_data(self, X, y):
        from keras.utils import np_utils
        return  (X.reshape(X.shape[0], 1, self.input_shape[1], self.input_shape[2]),
        np_utils.to_categorical(y, self.nb_classes))

    def _train(self):
        raise NotImplementedError

    def _get_image_window_index(self):
        raise NotImplementedError

    def train(self):
        self.X_train, self.Y_train = self._reshape_data(X, y)
        self._train()

    def score(self, user_id, image):
        raise NotImplementedError

    def get_highest_score_user(self, image):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def _train(self):
        from keras.optimizers import SGD

        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model.fit(self.X_train, self.Y_train, batch_size=consts.batch_size, nb_epoch=consts.nb_epoch,
                verbose=1, shuffle=True, validation_split=consts.validation_split)

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
        self.model.fit(self.X_train, self.Y_train, batch_size=consts.batch_size, nb_epoch=consts.nb_epoch,
                verbose=1, shuffle=True, validation_split=consts.validation_split)

    def score(self, user_id, image):
        X_test, _ = self._process_data([[image]])
        X_test, _ = self._reshape_data(X_test, _)
        proba = self.model.predict_proba(X_test)
        return proba[0][self.id_to_idx[user_id]]

    def get_highest_score_user(self, image):
        X_test, _ = self._process_data([[image]])
        X_test, _ = self._reshape_data(X_test, _)
        proba = self.model.predict_proba(X_test)
        max_prob = 0
        max_idx = None
        for idx, prob in proba:
            if prob > max_prob:
                max_prob = prob
                max_idx = idx
        return self.ids[idx]

    def save(self):
        w = Writer()
        w.save_model(self.model, self.name)

class NN1Model(Model):

    def __init__(self, images, ids, name):
        super(NN1Model, self).__init__(
                images,
                ids,
                name
        )
        self.input_shape = consts.nn1_input_shape
        self.model = _NN1(self.nb_classes, self.input_shape)


class NN2Model(Model):

    def __init__(self, images, ids, name):
        super(NN2Model, self).__init__(
                images,
                ids,
                name
        )
        self.input_shape = consts.nn2_input_shape
        self.model = _NN2(self.nb_classes, self.input_shape)


class CNNH1Model(NN2Model):

    def _get_image_window_index(self):
        return 4

class CNNP1Model(NN1Model):

    def _get_image_window_index(self):
        return 0


class CNNP2Model(NN1Model):

    def _get_image_window_index(self):
        return 1


class CNNP3Model(NN1Model):

    def _get_image_window_index(self):
        return 2


class CNNP4Model(NN1Model):

    def _get_image_window_index(self):
        return 3


class CNNP5Model(NN1Model):

    def _get_image_window_index(self):
        return 4


class CNNP6Model(NN1Model):

    def _get_image_window_index(self):
        return 5


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
