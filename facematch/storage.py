import consts

from image import Image

import fnmatch
import numpy as np
import shutil
import os
import uuid

class Storage(object):

    def __init__(self):
        self.create_directory(consts.data_path)
        self.create_directory(consts.image_path)
        self.create_directory(consts.model_path)

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def remove_directory(self, path):
        shutil.rmtree(path)


class Writer(Storage):

    def save_image(self, user_id, image):
        image.assert_valid_state()
        user_path = os.path.join(consts.image_path, user_id)
        self.create_directory(user_path)
        image_id = str(uuid.uuid4())
        image_path = os.path.join(user_path, image_id)
        np.save(image_path + consts.image_ext, image.image)
        np.save(image_path + consts.landmarks_ext, image.landmark_points)
        np.save(image_path + consts.features_ext, image.feature_points)

    def save_model(self, model, name):
        """
        params:
        model: keras model
        name: file prefix
        """
        model_path = os.path.join(consts.model_path, name)
        json = model.to_json()
        open(model_path + consts.json_ext, 'w').write(json)
        model.save_weights(model_path + consts.h5_ext, overwrite=True)

    def save_activations(self, activations, model_name):
        pass


class Reader(Storage):

    def get_images(self, user_id):
        user_path = os.path.join(consts.image_path, user_id)
        if not os.path.exists(user_path):
            return None

        image_ids = [f.replace(consts.image_ext, '')
                for f in os.listdir(user_path)
                if fnmatch.fnmatch(f, '*' + consts.image_ext)]

        return [self._get_image(image_id, user_path) for image_id in image_ids]

    def _get_image(self, image_id, user_path):
        image_path = os.path.join(user_path, image_id)
        image = Image()
        image.image = np.load(image_path + consts.image_ext)
        image.landmark_points = np.load(image_path + consts.landmarks_ext)
        image.feature_points = np.load(image_path + consts.features_ext)
        image.assert_valid_state()
        return image

    def get_user_ids(self):
        return [os.path.basename(os.path.normpath(x[0]))
            for x in os.walk(consts.image_path)
            if x[0] != consts.image_path]

    def get_model(self, model_name):
        from keras.models import model_from_json

        model_path = os.path.join(consts.model_path, model_name)
        model_file = model_path + consts.json_ext
        weight_file = model_path + consts.h5_ext
        if not os.path.isfile(model_file) or not os.path.isfile(weight_file):
            raise Exception('Model files do not exist')

        model = model_from_json(open(model_file).read())
        model.load_weights(weight_file)
        return model

    def load_activations(self, model_name):
        pass
