import json
import os
import shutil
from typing import Dict, List

import cv2
import numpy as np
from flask import request
from tensorpack.utils import logger
from tqdm import tqdm
from werkzeug import FileStorage
from werkzeug.utils import secure_filename

from mot.object_detection.query_server import \
    localizer_tensorflow_serving_inference
from mot.tracker.tracker import ObjectTracking
from mot.tracker.video_utils import read_folder, split_video

SERVING_URL = "http://localhost:8501"  # the url where the tf-serving container exposes the model
UPLOAD_FOLDER = 'tmp'  # folder used to store images or videos when sending files
FPS = 4
RESOLUTION = (1024, 768)
CLASS_NAMES = ["bottles", "others", "fragments"]
CLASS_TO_THRESHOLD = {"bottles": 0.4, "others": 0.3, "fragments": 0.3}


def handle_post_request(upload_folder=UPLOAD_FOLDER,
                        fps=FPS,
                        resolution=RESOLUTION) -> Dict[str, np.array]:
    """This method is the first one to be called when a POST request is coming. It analyzes the incoming
        format (file or JSON) and then call the appropiate methods to do the prediction.

    If you want to make a prediction by sending the data as a JSON, it has to be in this format:

    ```json
    {"image":[[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]]}
    ```

    or

    ```json
    {"video": TODO}
    ```
    Arguments:

    - *upload_folder*: Where the files are temporarly stored

    Returns:

    - *Dict[str, np.array]*: The predictions of the TF serving module

    Raises:

    - *NotImplementedError*: If the format of data isn't handled yet
    """
    if "file" in request.files:
        return handle_file(request.files['file'], upload_folder, fps, resolution)
    data = json.loads(request.data.decode("utf-8"))
    if "image" in data:
        image = np.array(data["image"])
        return {"detected_trash": predict_and_format_image(image)}
    elif "video" in data:
        raise NotImplementedError("video")


def handle_file(file: FileStorage,
                upload_folder=UPLOAD_FOLDER,
                fps=FPS,
                resolution=RESOLUTION) -> Dict[str, np.array]:
    """Make the prediction if the data is coming from an uploaded file.

    Arguments:

    - *file*: The file, can be either an image or a video
    - *upload_folder*: Where the files are temporarly stored

    Returns:

    - for an image: a json of format

    ```json
    {
        "image": filename,
        "detected_trash":
            [
                {
                    "box": [1, 1, 2, 20],
                    "label": "fragments",
                    "score": 0.92
                }, {
                    "box": [10, 10, 25, 20],
                    "label": "bottles",
                    "score": 0.75
                }
            ]
    }
    ```

    - for a video: a json of format

    ```json
    {
        "video_length": 132,
        "fps": 2,
        "video_id": "GOPRO1234.mp4",
        "detected_trash":
            [
                {
                    "label": "bottles",
                    "id": 0,
                    "frame_to_box": {
                        23: [0, 0, 1, 10],
                        24: [1, 1, 4, 13]
                    }
                }, {
                    "label": "fragments",
                    "id": 1,
                    "frame_to_box": {
                        12: [10, 8, 9, 15]
                    }
                }
            ]
    }
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
    file_type = file.mimetype.split("/")[
        0]  # mimetype is for example 'image/png' and we only want the image

    if file_type == "image":
        image = cv2.imread(full_filepath)  # cv2 opens in BGR
        os.remove(full_filepath)  # remove it as we don't need it anymore
        return {"image": filename, "detected_trash": predict_and_format_image(image)}

    elif file_type == "video":
        folder = os.path.join(upload_folder, "{}_split".format(filename))
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        logger.info("Splitting video {} to {}.".format(full_filepath, folder))
        split_video(full_filepath, folder, fps=fps, resolution=resolution)
        list_path_images = read_folder(folder)
        if len(list_path_images) == 0:
            raise ValueError("No output image")
        logger.info("{} images to analyze.".format(len(list_path_images)))
        list_inference_output = []
        for i, image_path in enumerate(tqdm(list_path_images)):
            image = cv2.imread(image_path)  # cv2 opens in BGR
            output = localizer_tensorflow_serving_inference(image, SERVING_URL)
            list_inference_output.append(output)
        logger.info("Finish analyzing video {}.".format(full_filepath))
        logger.info("Starting tracking.")
        object_tracker = ObjectTracking(filename, list_path_images, list_inference_output, fps=fps)
        logger.info("Tracking finished.")
        return object_tracker.json_result()
    else:
        raise NotImplementedError(file_type)


def predict_and_format_image(
    image: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    class_to_threshold: Dict[str, float] = CLASS_TO_THRESHOLD
) -> List[Dict[str, object]]:
    """Make prediction on an image and return them in a human readable format.

    Arguments:

    - *image*: An numpy array in BGR
    - *class_names*: The list of class names without background
    - *class_to_threshold*: A dict assigning class names to threshold. If a class name isn't in
        this dict, no threshold will be applied, which means that all predictions for this class
        will be kept.

    Returns:

    - *List[Dict[str, object]]*: List of dicts such as:

    ```python3
    {
        "box": [1, 1, 2, 20],
        "label": "fragments",
        "score": 0.92
    }
    ```
    """
    class_names = ["BG"] + class_names
    outputs = localizer_tensorflow_serving_inference(image, SERVING_URL)
    detected_trash = []
    for box, label, score in zip(
        outputs["output/boxes:0"], outputs["output/labels:0"], outputs["output/scores:0"]
    ):
        if class_names[label] not in class_to_threshold or score >= class_to_threshold[
            class_names[label]]:
            trash_json = {"box": [x for x in box], "label": class_names[label], "score": score}
            detected_trash.append(trash_json)
    return detected_trash
