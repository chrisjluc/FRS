import numpy as np

def _reflection(image):
    return np.array([list(reversed(row)) for row in image])

def partition(image, top_left, rows, cols):
    return np.array([row[top_left[1]:top_left[1] + cols] for row in image[top_left[0]:top_left[0] + rows]])

def clone_images(images):
    pass

def apply_reflection(images):
    pass

def apply_noise(images):
    pass

def apply_cloning(images, factor):
    """
    Clones in place by factor
    If factor is two, we double the image sample size

    params:
    factor: int
    """
    pass

def merge(a, b):
    """
    args:
    a - list of list of images
    b - list of list of images

    Returns:
    Merged list of list of images
    """
    pass

def process_images_for_cnn1(images):
    pass
