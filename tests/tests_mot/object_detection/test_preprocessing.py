import numpy as np

from mot.object_detection.preprocessing import (resize_to_min_dimension,
                                                preprocess_for_serving)


def test_resize_to_min_dimension():
    image = np.zeros((100, 50, 3))
    # If min_dimension is put to 800 => scale = 16 and 100 will be put to 1600 which is over max_dimension
    max_dimension = 1000
    min_dimension = 800
    image_out, scale = resize_to_min_dimension(image, min_dimension, max_dimension)
    image_exp = np.zeros((1000, 500, 3))
    np.testing.assert_array_equal(image_exp, image_out)
    assert scale == 10

    max_dimension = 1600
    min_dimension = 800
    image_out, scale = resize_to_min_dimension(image, min_dimension, max_dimension)
    image_exp = np.zeros((1600, 800, 3))
    np.testing.assert_array_equal(image_exp, image_out)
    assert scale == 16


def test_preprocess_for_serving():
    image = np.zeros((100, 100, 3))
    image[:, :, 1] = 1
    image[:, :, 2] = 2

    signature, ratio = preprocess_for_serving(image)
    signature['inputs'] = np.array(signature['inputs'])

    assert ratio == 8
    assert np.sum(signature['inputs'][:, :, 0]) == 0
    assert np.sum(signature['inputs'][:, :, 1]) == 800**2
    assert np.sum(signature['inputs'][:, :, 2]) == 2 * 800**2
