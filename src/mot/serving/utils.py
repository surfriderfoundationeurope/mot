import json
import os
from typing import Dict

import cv2
import numpy as np
from flask import flash, redirect, render_template, request
from mot.object_detection.query_server import \
    localizer_tensorflow_serving_inference
from werkzeug import FileStorage
from werkzeug.utils import secure_filename

SERVING_URL = "http://localhost:8501"  # the url where the tf-serving container exposes the model
UPLOAD_FOLDER = 'tmp'  # folder used to store images or videos when sending files


def handle_post_request(upload_folder=UPLOAD_FOLDER) -> Dict[str, np.array]:
    """This method is the first one to be called when a POST request is coming. It analyzes the incoming
        format (file or JSON) and then call the appropiate methods to do the prediction.

    Arguments:

    - *request*: the POST request coming. It might be an uploaded file or a JSON.
        If you want to make a prediction by sending the data as a JSON, it has to be in this format:

    ```python
        {"image":[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]]}
    ````

    or

    ```python
        {"video": TODO}
    ```

    Returns:

    - *Dict[str, np.array]*: The predictions of the TF serving module

    Raises:

    - *NotImplementedError*: If the format of data isn't handled yet
    """
    if "file" in request.files:
        return handle_file(request.files['file'], upload_folder)
    data = json.loads(request.data.decode("utf-8"))
    if "image" in data:
        image = np.array(data["image"])
        return localizer_tensorflow_serving_inference(image, SERVING_URL)
    elif "video" in data:
        raise NotImplementedError("video")


def handle_file(file: FileStorage, upload_folder=UPLOAD_FOLDER) -> Dict[str, np.array]:
    """Make the prediction if the data is coming from an uploaded file

    Arguments:

    - *file*: The file, can be either an image or a video
    - *upload_folder*: Where the files are temporarly stored

    Returns:

    - *Dict[str, np.array]*: The predictions of the TF serving module

    Raises:

    - *NotImplementedError*: If the format of data isn't handled yet
    """
    filename = secure_filename(file.filename)
    filename = os.path.join(upload_folder, filename)
    if not os.path.isdir(upload_folder):
        os.mkdir(upload_folder)
    if os.path.isfile(filename):
        os.remove(filename)
    file.save(filename)
    file_type = file.mimetype.split("/")[0] # mimetype is for example 'image/png' and we only want the image
    if file_type == "image":
        image = cv2.imread(filename) # cv2 opens in BGR
        image = image[:, :, ::-1] # convert to RGB
        os.remove(filename) # remove it as we don't need it anymore
        return localizer_tensorflow_serving_inference(image, SERVING_URL)
    elif file_type == "video":
        os.remove(filename) # remove it as we don't need it anymore
        raise NotImplementedError(file_type)
    os.remove(filename) # remove it as we don't need it anymore
    raise NotImplementedError(file_type)
