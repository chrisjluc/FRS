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

    def _shuffle_data(self, X, y):
        zipped = np.array(zip(X, y))
        np.random.shuffle(zipped)
        return (np.array([x[0] for x in zipped]),
                np.array([x[1] for x in zipped]))

    def _reshape_data(self, X, y):
        from keras.utils import np_utils
        return  (X.reshape(X.shape[0], 1, self.input_shape[1], self.input_shape[2]),
        np_utils.to_categorical(y, self.nb_classes))

    def _train(self):
        raise NotImplementedError

    def train(self):
        X, y = self._process_data(self.images)
        X, y = self._shuffle_data(X, y)
        self.X_train, self.Y_train = self._reshape_data(X, y)
        self._train()

    def score(self, user_id, image):
        raise NotImplementedError

    def get_highest_score_user(self, image):
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
        image_shape = self.input_shape[1], self.input_shape[2]
        data = [[ip.get_image_window(
                    image.image,
                    image_shape,
                    image.landmark_points[4])
                for image in images]
                for images in data]

        X = np.array([image
                for images in data
                for image in images
                if image.shape == image_shape
                ])

        y = np.array([images[0]
                for images in enumerate(data)
                for image in images[1]
                if image.shape == image_shape
                ])

        return X, y

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
