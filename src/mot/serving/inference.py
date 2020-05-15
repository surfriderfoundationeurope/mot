import logging
import multiprocessing
import os
import shutil
import zipfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from werkzeug.datastructures import FileStorage

from mot.object_detection.query_server import \
    localizer_tensorflow_serving_inference
from mot.serving.constants import (CLASS_NAMES, CLASS_TO_THRESHOLD, CPU_COUNT,
                                   DEFAULT_THRESHOLD, FPS, RESOLUTION,
                                   SERVING_URL)
from mot.serving.utils import save_file
from mot.tracker.object_tracking import ObjectTracking
from mot.tracker.video_utils import read_folder, split_video

logger = logging.getLogger(__file__)

###    TRACKING (FROM VIDEO OR ZIPFILE)    ###


def detect_and_track_images(
    file: FileStorage,
    upload_folder: str,
    fps: int = FPS,
    resolution: Tuple[int, int] = RESOLUTION,
) -> Dict[str, np.array]:
    """Performs object detection and then tracking on a video or a zip containing images.

    Arguments:

    - *file*: The file, can be either a video or a zipped folder
    - *upload_folder*: Where the files are temporarly stored

    Returns:

    - *predictions*:

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
    """

    filename, full_filepath = save_file(file, upload_folder)

    def process_video():
        images_folder = os.path.join(upload_folder, "{}_split".format(filename))
        shutil.rmtree(images_folder, ignore_errors=True)
        os.mkdir(images_folder)
        logger.info("Splitting video {} to {}.".format(full_filepath, images_folder))
        split_video(full_filepath, images_folder, fps=fps, resolution=resolution)
        return images_folder

    def process_zip():
        images_folder = os.path.join(upload_folder, "{}_split".format(filename))
        with zipfile.ZipFile(full_filepath, 'r') as zip_obj:
            zip_obj.extractall(images_folder)

        def move_files_to_root(directory, root_directory):
            for x in os.listdir(directory):
                path = os.path.join(directory, x)
                if x.startswith("._") or x.startswith("__"):
                    # unwanted files such as __MACOSX
                    shutil.rmtree(path)
                else:
                    if os.path.isfile(path):
                        # we want to move this file to the root of the zip directory
                        if not os.path.isfile(os.path.join(root_directory, x)):
                            # unless it is aleady present at root
                            shutil.move(path, root_directory)
                    else:
                        # if there is a folder, we want to move back the files to root
                        move_files_to_root(path, root_directory)

        move_files_to_root(images_folder, images_folder)

        return images_folder

    if file.mimetype == "":
        # no type: we try to unzip, and if it fails we split as a video
        try:
            images_folder = process_zip()
        except zipfile.BadZipFile:
            images_folder = process_video()
    elif file.mimetype == "application/zip":
        # zip case
        images_folder = process_zip()
    else:
        # video case: splitting video and saving frames
        images_folder = process_video()

    image_paths = read_folder(images_folder)
    if len(image_paths) == 0:
        raise ValueError("No output image")

    # making inference on frames
    logger.info("{} images to analyze on {} CPUs.".format(len(image_paths), CPU_COUNT))
    try:
        with multiprocessing.Pool(CPU_COUNT) as p:
            inference_outputs = list(
                tqdm(
                    p.imap(_process_image, image_paths),
                    total=len(image_paths),
                )
            )
    except ValueError as e:
        return {"error": str(e)}
    logger.info("Object detection on video {} finished.".format(full_filepath))

    # tracking objects
    logger.info("Starting tracking for video {}.".format(full_filepath))
    object_tracker = ObjectTracking(filename, image_paths, inference_outputs, fps=fps)
    tracks = object_tracker.compute_tracks()
    logger.info("Tracking finished.")
    predictions = object_tracker.json_result(tracks)
    return predictions


def _process_image(image_path: str) -> Dict:
    """Function used to open and predict on an image. It is suposed to be used in multiprocessing.

    Arguments:

    - *image_path*

    Returns:

    - *predictions*: Object detection predictions for this image path. A dict such as:

    ```python
    predictions = {
        'output/boxes:0': [[0, 0, 1, 1], [0, 0, 10, 10], [10, 10, 15, 100]],
        'output/labels:0': [3, 1, 2], # the labels start at 1 since 0 is for background
        'output/scores:0': [
                [0.001, 0.001, 0.98],
                [0.87, 0.05, 0.03],
                [0.1, 0.76, 0.1],
            ]
            # the scores don't necessarly sum up to 1 because. The remainder is the background score.
    }
    ```
    """
    image = cv2.imread(image_path)  # cv2 opens in BGR
    predictions = localizer_tensorflow_serving_inference(image, SERVING_URL, return_all_scores=True)
    return predictions


###   OBJECT DETECTION (FROM IMAGE)    ###


def predict_image_file(file: FileStorage, upload_folder: str):
    """Save an image file on disk and then perform object detection on it.

    Arguments:

    - *file*: An image file sent to the app.
    - *upload_folder*: The folder where the picture will be temporarly stored.

    Returns:

    - *[type]*: [description]

    Raises:

    - *ValueError*: [description]
    """
    filename, full_filepath = save_file(file, upload_folder)

    image = cv2.imread(full_filepath)  # cv2 opens in BGR
    try:
        detected_trash = predict_and_format_image(image)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "full_filepath": full_filepath,
        "filename": filename,
        "detected_trash": detected_trash,
    }


def predict_and_format_image(
    image: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    class_to_threshold: Dict[str, float] = CLASS_TO_THRESHOLD
) -> List[Dict]:
    """Make prediction on an image as an array and return them in a human readable format.

    Arguments:

    - *image*: An numpy array in BGR
    - *class_names*: The list of class names without background
    - *class_to_threshold*: A dict assigning class names to threshold. If a class name isn't in
        this dict, DEFAUL_THRESHOLD will be applied.

    Returns:

    - *detected_trash*: List of dicts such as:

    ```python3
    {
        "box": [1, 1, 2, 20],
        "label": "fragments",
        "score": 0.92
    }
    ```
    """
    class_names = ["BG"] + class_names
    outputs = localizer_tensorflow_serving_inference(image, SERVING_URL, return_all_scores=False)
    detected_trash = []
    for box, label, score in zip(
        outputs["output/boxes:0"], outputs["output/labels:0"], outputs["output/scores:0"]
    ):
        if score >= class_to_threshold.get(class_names[label], DEFAULT_THRESHOLD):
            trash_json = {
                "box": [round(coord, 2) for coord in box],
                "label": class_names[label],
                "score": score,
            }
            detected_trash.append(trash_json)

    return detected_trash
