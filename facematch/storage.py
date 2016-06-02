import consts

from image import Image

import numpy as np
import os
import uuid


class Writer(object):

    def __init__(self):
        self._create_directory(consts.data_path)
        self._create_directory(consts.image_path)
        self._create_directory(consts.model_path)

    def save_image(self, user_id, image):
        image.assert_valid_state()
        user_path = os.path.join(image_path, user_id)
        self._create_directory(user_path)
        image_id = uuid.uuid4()
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

    def _create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


class Reader(object):

    def __init__(self):
        pass

    def get_images(self, user_id):
        user_path = os.path.join(consts.image_path, user_id)
        if not os.path.exists(user_path):
            return None

        image_ids = [f.replace(image_ext, '')
                for f in os.listdir(user_path)
                if fnmatch.fnmatch(f, '*' + image_ext)]

        return [_get_image(image_id, user_path) for image_id in image_ids]

    def _get_image(self, image_id, user_path):
        image_path = os.path.join(user_path, image_id)
        image = Image()
        image.image = np.load(image_path + consts.image_ext)
        image.landmark_points = np.load(image_path + consts.landmarks_ext)
        image.feature_points = np.load(image_path + consts.features_ext)
        image.assert_valid_state()
        return image

    def get_user_ids(self):
        return [x[0] for x in os.walk(consts.image_path)]

    def get_model(self, model_type):
        pass
