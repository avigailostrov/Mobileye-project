from matplotlib import image as mpimg
try:
    import os
    import json
    import glob
    import argparse

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy import signal as sg, ndimage
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image
    from matplotlib.cbook import get_sample_data

    import scipy
    import scipy.ndimage.filters as filters

    from matplotlib.cbook import get_sample_data
    import imageio
    from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import glob
import argparse
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import cv2
from scipy import signal as sg
from scipy.ndimage.filters import maximum_filter
from PIL import Image
from matplotlib.cbook import get_sample_data
import imageio
import numpy as np
import scipy
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


def max_filter(img, threshold, neighborhood_size):
    data_max = filters.maximum_filter(img, neighborhood_size)
    maxima = (img == data_max)
    data_min = filters.minimum_filter(img, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)
    plt.autoscale(False)
    return x, y


def find_tfl_lights(img_path):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    image = Image.open(img_path)
    img = np.asarray(image)
    #img = cv2.imread(img_path)
    info = np.iinfo(img.dtype)  # Get the information of the incoming image type
    # one channel 2D
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    plt.imshow(Image.fromarray(r, 'L'))
    kernel = np.array([[-0.065, -0.065, -0.065, -0.065, -0.065],
                       [-0.065, 0.05, 0.4, 0.05, -0.065],
                       [-0.065, 0.4, 0.4, 0.4, -0.065],
                       [-0.065, 0.05, 0.4, 0.05, -0.065],
                       [-0.065, -0.065, -0.065, -0.065, -0.065]])
    g = g.astype(np.float64) / info.max
    r = r.astype(np.float64) / info.max
    img_b = sg.convolve(b, kernel, mode='same', method='auto')
    img_g = sg.convolve(g, kernel, mode='same', method='auto')
    img_r = sg.convolve(r, kernel, mode='same', method='auto')
    plt.imshow(Image.fromarray(img_g, 'L'))
    return img_r, img_g
