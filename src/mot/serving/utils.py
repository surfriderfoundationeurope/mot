import json
import os
from typing import Dict

import cv2
import numpy as np
from flask import flash, redirect, render_template, request
from mot.object_detection.query_server import \
    localizer_tensorflow_serving_inference

from mot.tracker.video_utils import split_video, read_folder
from mot.tracker.tracker import ObjectTracking
from mot.object_detection.config import config as cfg

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
    ```

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
        outputs = localizer_tensorflow_serving_inference(image, SERVING_URL)
        detected_trash = []
        for box,label,score in zip(outputs["output/boxes:0"],outputs["output/labels:0"],outputs["output/scores:0"]):
            trash_json = {"box":[x for x in box], "label":cfg.DATA.CLASS_NAMES[label], "score":score}
            detected_trash.append(trash_json)
        return {"detected_trash": detected_trash}

    elif "video" in data:
        raise NotImplementedError("video")


def handle_file(file: FileStorage, upload_folder=UPLOAD_FOLDER, fps=2) -> Dict[str, np.array]:
    """Make the prediction if the data is coming from an uploaded file

    Arguments:

    - *file*: The file, can be either an image or a video
    - *upload_folder*: Where the files are temporarly stored

    Returns:

    - for an image: a json of format {"image": filename, "detected_trash": detected_trash}
    - for a video: a json of format ```python
    {"video_length": 132, "fps": 2, "video_id": "GOPRO1234.mp4", "detected_trash": [{"label": "bottle", "id": 0, "frames": [23,24,25]}, {"label": "fragment", "id": 1, "frames": [32]}]}
    ```

    Raises:

    - *NotImplementedError*: If the format of data isn't handled yet
    """
    filename = secure_filename(file.filename)
    full_filepath = os.path.join(upload_folder, filename)
    if not os.path.isdir(upload_folder):
        os.mkdir(upload_folder)
    if os.path.isfile(full_filepath):
        os.remove(full_filepath)
    file.save(full_filepath)
    file_type = file.mimetype.split("/")[0] # mimetype is for example 'image/png' and we only want the image

    if file_type == "image":
        image = cv2.imread(full_filepath) # cv2 opens in BGR
        image = image[:, :, ::-1] # convert to RGB
        os.remove(full_filepath) # remove it as we don't need it anymore
        outputs = localizer_tensorflow_serving_inference(image, SERVING_URL)
        detected_trash = []
        for box,label,score in zip(outputs["output/boxes:0"],outputs["output/labels:0"],outputs["output/scores:0"]):
            trash_json = {"box":[x for x in box], "label":cfg.DATA.CLASS_NAMES[label], "score":score}
            detected_trash.append(trash_json)
        return {"image": filename, "detected_trash": detected_trash}

    elif file_type == "video":
        folder = os.path.join(upload_folder, "{}_split".format(filename))
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        split_video(full_filepath, folder, fps = fps)
        list_path_images = read_folder(folder)
        if len(list_path_images) == 0:
            raise ValueError("No output image")
        list_inference_output = []
        for image_path in list_path_images:
            image = cv2.imread(image_path)
            output = localizer_tensorflow_serving_inference(image, SERVING_URL)
            list_inference_output.append(output)
        object_tracker = ObjectTracking(filename, list_path_images, list_inference_output, fps = fps)
        object_tracker.track_objects()
        return object_tracker.json_result()

        shutil.rmtree(folder) # remove it as we don't need it anymore


    else:
        raise NotImplementedError(file_type)
