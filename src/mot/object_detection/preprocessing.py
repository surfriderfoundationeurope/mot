import numpy as np
import cv2


def resize_to_min_dimension(image, min_dimension: int, max_dimension: int):
    """Resize an image given to the min size maintaining the aspect ratio.

    If one of the image dimensions is bigger than the max_dimension after resizing, it will scale
    the image such that its biggest dimension is equal to the max_dimension.
    Otherwise, will keep the image size as is.

    Arguments :

    - *image*: A np.array of size [height, width, channels].
    - *min_dimension*: minimum image dimension.
    - *max_dimension*: If the resized largest size is over max_dimension. Will use to max_dimension
    to compute the resizing ratio.

    Returns:

    - *resized_image*: The input image resized with the aspect_ratio preserved in float32
    - *scale_ratio*: The ratio used to scale back the boxes to the good shape
    """
    image = image.astype(np.float32, copy=False)
    im_size_min = np.min(image.shape[0:2])
    im_size_max = np.max(image.shape[0:2])
    scale_ratio = float(min_dimension) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(scale_ratio * im_size_max) > max_dimension:
        scale_ratio = float(max_dimension) / float(im_size_max)

    resized_image = cv2.resize(image,
                               None,
                               None,
                               fx=scale_ratio,
                               fy=scale_ratio,
                               interpolation=cv2.INTER_LINEAR)
    return resized_image, scale_ratio


def preprocess_for_serving(image, min_dimension=800, max_dimension=1300):
    """Adapt the preprocessing to the tensorpack

    Arguments:

    - *image*: A np.array of shape [height, width, channels] in BGR
    - *min_dimension*: minimum image dimension.
    - *max_dimension*: If the resized largest size is over max_dimension. Will use to max_dimension
    to compute the resizing ratio.



    Returns:

    - *input_signature*: A dictionary which match the server signature
    - *scaling_ratio*: A float representing the scaling to resize the image.
    """
    resized_image, scale_ratio = resize_to_min_dimension(image, min_dimension, max_dimension)
    return {"inputs": resized_image.tolist()}, scale_ratio
