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

def get_image_window(image, size, point):
    """
    Assume image is grey image
    """
    top = int(point[1] - size[0] / 2)
    left = int(point[0] - size[1] / 2)
    return image[top:top + size[0], left:left + size[1]]
