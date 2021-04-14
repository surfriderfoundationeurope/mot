import json
import os
from unittest import mock

import cv2
import ffmpeg
import numpy as np
import pytest
from werkzeug.datastructures import FileStorage

from mot.serving.inference import (_process_image, detect_and_track_images,
                                   predict_and_format_image)

HOME = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(HOME, ".mot/tests/test_video.mp4")
PATH_TO_TEST_ZIP = os.path.join(HOME, ".mot/tests/test_video_folder.zip")


def mock_post_tensorpack_localizer(*args, **kwargs):
    boxes = [[0, 0, 120, 40], [0, 0, 120, 80]]
    scores = [[0.7, 0.01, 0.01], [0.1, 0.1, 0.6]]
    classes = [1, 3]

    class Response(mock.Mock):
        json_text = {
            'outputs':
                {
                    'output/boxes:0': boxes,
                    'output/scores:0': scores,
                    'output/labels:0': classes,
                }
        }

        @property
        def text(self):
            return json.dumps(self.json_text)

        def json(self):
            return self.json_text

    response = Response()
    return response


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_detect_and_track_images_video(mock_server_result, tmpdir):
    split_frames_folder = os.path.join(
        tmpdir, "{}_split".format(os.path.basename(PATH_TO_TEST_VIDEO))
    )
    os.mkdir(split_frames_folder)  # this folder should be deleted by handle post request
    output = detect_and_track_images(
        FileStorage(open(PATH_TO_TEST_VIDEO, "rb")), upload_folder=str(tmpdir), fps=2
    )

    assert len(output["detected_trash"]) == 2
    assert "id" in output["detected_trash"][0]
    assert "frame_to_box" in output["detected_trash"][1]
    for frame, box in output["detected_trash"][1]["frame_to_box"].items():
        assert isinstance(frame, int)
        assert isinstance(box, list)
        assert len(box) == 4
    assert output["video_length"] == 6 or output["video_length"] == 7
    assert output["fps"] == 2
    assert "video_id" in output

@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_detect_and_track_images_zip(mock_server_result, tmpdir):
    output = detect_and_track_images(
        FileStorage(open(PATH_TO_TEST_ZIP, "rb")), upload_folder=str(tmpdir), fps=2
    )

    assert len(output["detected_trash"]) == 2
    assert "id" in output["detected_trash"][0]
    assert "frame_to_box" in output["detected_trash"][1]
    for frame, box in output["detected_trash"][1]["frame_to_box"].items():
        assert isinstance(frame, int)
        assert isinstance(box, list)
        assert len(box) == 4
    assert output["video_length"] == 6 or output["video_length"] == 7
    assert output["fps"] == 2
    assert "video_id" in output


def test_handle_post_request_file_other(tmpdir):
    filename = "test.pdf"
    filepath = os.path.join(tmpdir, filename)
    with open(filepath, "w") as f:
        f.write("mock data")
    upload_folder = os.path.join(tmpdir, "upload_folder")
    with pytest.raises(ffmpeg._run.Error):
        detect_and_track_images(FileStorage(open(filepath, "rb")), upload_folder=upload_folder)
    assert os.path.isdir(
        upload_folder
    )  # the upload_folder should be created by handle post request


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_process_image(mock_server_result, tmpdir):
    image = np.ones((300, 200, 3))
    image_path = os.path.join(tmpdir, "image.jpg")
    cv2.imwrite(image_path, image)
    output = _process_image(image_path)
    expected_output = {
        'output/boxes:0': [[0, 0, 0.1, 0.05], [0, 0, 0.1, 0.1]],
        'output/scores:0': [[0.7, 0.01, 0.01], [0.1, 0.1, 0.6]],
        'output/labels:0': [1, 3]
    }
    assert output == expected_output


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_predict_and_format_image(mock_server_result, tmpdir):
    image = np.ones((300, 200, 3))
    predictions = predict_and_format_image(image)
    assert predictions == [{"box": [0, 0, 0.1, 0.05], "label": "bottles", "score": 0.7}]

    # tesing with different class names
    class_names = ["others", "fragments", "chicken", "bottles"]
    output = predict_and_format_image(image, class_names)
    expected_output = [
        {
            "box": [0, 0, 0.1, 0.05],
            "label": "others",
            "score": 0.7
        }, {
            "box": [0, 0, 0.1, 0.1],
            "label": "chicken",
            "score": 0.6
        }
    ]
    assert output == expected_output

    # testing with different thresholds
    class_to_threshold = {"bottles": 0.8, "others": 0.3, "fragments": 0.3}
    output = predict_and_format_image(image, class_to_threshold=class_to_threshold)
    expected_output = [{"box": [0, 0, 0.1, 0.1], "label": "fragments", "score": 0.6}]
    assert output == expected_output

    # testing with different thresholds
    class_to_threshold = {"bottles": 0.8, "others": 0.3, "fragments": 0.8}
    output = predict_and_format_image(image, class_to_threshold=class_to_threshold)
    expected_output = []
    assert output == expected_output


def mock_post_tensorpack_localizer_error(*args, **kwargs):

    class Response(mock.Mock):
        json_text = {'error': "¯\(°_o)/¯"}

        @property
        def text(self):
            return json.dumps(self.json_text)

        def json(self):
            return self.json_text

    response = Response()
    return response


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer_error)
def test_handle_post_request_file_error(mock_server_result, tmpdir):
    # videos
    split_frames_folder = os.path.join(
        tmpdir, "{}_split".format(os.path.basename(PATH_TO_TEST_VIDEO))
    )
    os.mkdir(split_frames_folder)  # this folder should be deleted by handle post request
    outputs = detect_and_track_images(
        FileStorage(open(PATH_TO_TEST_VIDEO, "rb")), upload_folder=str(tmpdir)
    )
    assert "error" in outputs
    assert outputs["error"].endswith("¯\(°_o)/¯")

    # images
    data = np.ones((300, 200, 3))
    filename = "test.jpg"
    filepath = os.path.join(tmpdir, filename)
    cv2.imwrite(filepath, data)
    outputs = detect_and_track_images(FileStorage(open(filepath, "rb")), upload_folder=str(tmpdir))
    assert "error" in outputs
    assert outputs["error"].endswith("¯\(°_o)/¯")


def mock_post_tensorpack_localizer_unknown(*args, **kwargs):

    class Response(mock.Mock):
        json_text = {'unknown': "¯\_(ツ)_/¯"}

        @property
        def text(self):
            return json.dumps(self.json_text)

        def json(self):
            return self.json_text

    response = Response()
    return response


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer_unknown)
def test_handle_post_request_file_unknwon(mock_server_result, tmpdir):
    # videos
    split_frames_folder = os.path.join(
        tmpdir, "{}_split".format(os.path.basename(PATH_TO_TEST_VIDEO))
    )
    os.mkdir(split_frames_folder)  # this folder should be deleted by handle post request
    outputs = detect_and_track_images(
        FileStorage(open(PATH_TO_TEST_VIDEO, "rb")), upload_folder=str(tmpdir)
    )
    assert "error" in outputs
    assert outputs["error"].endswith("{'unknown': '¯\\\\_(ツ)_/¯'}")

    # images
    data = np.ones((300, 200, 3))
    filename = "test.jpg"
    filepath = os.path.join(tmpdir, filename)
    cv2.imwrite(filepath, data)
    outputs = detect_and_track_images(FileStorage(open(filepath, "rb")), upload_folder=str(tmpdir))
    assert "error" in outputs
    assert outputs["error"].endswith("{'unknown': '¯\\\\_(ツ)_/¯'}")
