import consts
import image_processing as ip
from storage import Writer

import numpy as np


class Model(object):

    def __init__(self, images, ids, name):
        self.nb_classes = len(ids)
        self.images = images
        self.ids = ids
        self.name = name
        self.id_to_idx = dict([(x[1], x[0]) for x in enumerate(ids)])

    def _process_data(self):
        raise NotImplementedError

    def _shuffle_data(self):
        from keras.utils import np_utils

        zipped = np.array(zip(self.X_train, self.y_train))
        np.random.shuffle(zipped)
        X_train = np.array([x[0] for x in zipped])
        y_train = np.array([x[1] for x in zipped])
        self.X_train = X_train.reshape(X_train.shape[0], 1, self.input_shape[1], self.input_shape[2])
        self.Y_train = np_utils.to_categorical(y_train, self.nb_classes)

    def _train(self):
        raise NotImplementedError

    def train(self):
        self.X_train, self.y_train = self._process_data(self.images)
        self._shuffle_data()
        self._train()

    def score(self, user_id, image):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class NN2Model(Model):

    def __init__(self, images, ids, name):
        super(NN2Model, self).__init__(
                images,
                ids,
                name
        )
        self.input_shape = consts.nn2_input_shape
        self.model = _NN2(self.nb_classes, self.input_shape)

    def _process_data(self, data):
        data = [[ip.get_image_window(
                    image.image,
                    self.input_shape,
                    image.landmark_points[4])
                for image in images]
                for images in data]

        X = [image
                for images in data
                for image in images
                if image.shape == self.input_shape
                ]

        y = [self.ids[images[0]]
                for images in enumerate(data)
                for image in images[1]
                if image.shape == self.input_shape
                ]

        return X, y

    def _train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')
        self.model.fit(X_train, Y_train, batch_size=32, nb_epoch=10,
                        show_accuracy=True, verbose=1, shuffle=True, validation_split=.15)

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001))
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=10,
                        show_accuracy=True, verbose=1, shuffle=True, validation_split=.15)

    def score(self, user_id, image):
        X_test, _ = self._process_data([[image]])
        proba = model.predict_proba(X_test)
        return proba[0][self.id_to_idx[user_id]]

    def save(self):
        w = Writer()
        w.save(self.model, self.name)


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
