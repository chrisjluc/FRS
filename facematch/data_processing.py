import copy
import numpy as np


def merge(a, b):
    """
    args:
    a - list of list of images
    b - list of list of images

    Returns:
    Merged list of list of images
    """
    if len(a) != len(b):
        raise Exception('a and b should have the same length')

    merged = []
    for x, y in zip(a, b):
        merged.append(x + y)
    assert(len(merged) == len(a))
    return merged


def clone(images, factor):
    """
    images: list of list
    factor: int
    """
    cloned_images = []
    for _images in images:
        _cloned_images = []
        for i in range(factor):
            _cloned_images += copy.deepcopy(_images)
        cloned_images.append(_cloned_images)
    assert(len(images) == len(cloned_images))
    return cloned_images

def flatten(images):
    pass

def shuffle(images);
    zipped = np.array(zip(X, y))
    np.random.shuffle(zipped)
    return (np.array([x[0] for x in zipped]),
            np.array([x[1] for x in zipped]))

def _process_data(self, data):
    image_shape = self.input_shape[1], self.input_shape[2]
    data = [[ip.get_image_window(
                image.image,
                image_shape,
                image.landmark_points[
                    self._get_image_window_index()])
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


