import io
import json
import os
import shutil
from unittest import mock

import cv2
import numpy as np
import pytest
from flask import Flask, request
from werkzeug import FileStorage

from mot.object_detection.config import config as cfg
from mot.serving.utils import handle_post_request, predict_and_format_image

home = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(home, ".mot/tests/test_video.mp4")


def mock_post_tensorpack_localizer(*args, **kwargs):
    boxes = [[0, 0, 40, 40], [0, 0, 80, 80]]
    scores = [0.71, 0.71]
    classes = [1, 3]
    response = mock.Mock()
    response.text = json.dumps(
        {
            'outputs':
                {
                    'output/boxes:0': boxes,
                    'output/scores:0': scores,
                    'output/labels:0': classes,
                }
        }
    )
    return response


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_handle_post_request_image(mock_server_result):
    data = "{}".format(json.dumps({"image": [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]}))
    data = data.encode('utf-8')
    m = mock.MagicMock()  # here we mock flask.request
    m.data = data
    with mock.patch("mot.serving.utils.request", m):
        output = handle_post_request()
    expected_output = {
        "detected_trash":
            [
                {
                    "box": [0.0, 0.0, 0.1, 0.1],
                    "label": "bottles",
                    "score": 0.71
                }, {
                    "box": [0.0, 0.0, 0.2, 0.2],
                    "label": "fragments",
                    "score": 0.71
                }
            ]
    }
    assert output == expected_output


def test_handle_post_request_video():
    # TODO test good behavior when implemented
    data = "{}".format(json.dumps({"video": [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]}))
    data = data.encode('utf-8')
    m = mock.MagicMock()  # here we mock flask.request
    m.data = data
    with pytest.raises(NotImplementedError):
        with mock.patch("mot.serving.utils.request", m):
            output = handle_post_request()


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_handle_post_request_file_image(mock_server_result, tmpdir):
    data = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    filename = "test.jpg"
    filepath = os.path.join(tmpdir, filename)
    cv2.imwrite(filepath, data)
    m = mock.MagicMock()
    files = {"file": FileStorage(open(filepath, "rb"), content_type='image/jpg')}
    m.files = files
    with mock.patch("mot.serving.utils.request", m):
        output = handle_post_request(upload_folder=str(tmpdir))
    expected_output = {
        "image":
            output["image"],
        "detected_trash":
            [
                {
                    "box": [0.0, 0.0, 0.1, 0.1],
                    "label": "bottles",
                    "score": 0.71
                }, {
                    "box": [0.0, 0.0, 0.2, 0.2],
                    "label": "fragments",
                    "score": 0.71
                }
            ]
    }
    assert output == expected_output


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_handle_post_request_file_video(mock_server_result, tmpdir):
    m = mock.MagicMock()
    files = {"file": FileStorage(open(PATH_TO_TEST_VIDEO, "rb"), content_type='video/mkv')}
    m.files = files
    with mock.patch("mot.serving.utils.request", m):
        output = handle_post_request(upload_folder=str(tmpdir))
        assert len(output["detected_trash"]) == 2
        assert "id" in output["detected_trash"][0]
        assert "frames" in output["detected_trash"][1]
        assert output["video_length"] == 6 or output["video_length"] == 7
        assert output["fps"] == 2
        assert "video_id" in output


def test_handle_post_request_file_other(tmpdir):
    filename = "test.pdf"
    filepath = os.path.join(tmpdir, filename)
    with open(filepath, "w") as f:
        f.write("mock data")
    m = mock.MagicMock()
    files = {"file": FileStorage(open(filepath, "rb"), content_type='application/pdf')}
    m.files = files
    with pytest.raises(NotImplementedError):
        with mock.patch("mot.serving.utils.request", m):
            output = handle_post_request(upload_folder=str(tmpdir))


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_predict_and_format_image(mock_server_result, tmpdir):
    image = np.ones((5, 5, 3))
    predictions = predict_and_format_image(image)
    assert predictions == [
        {
            "box": [0.0, 0.0, 0.25, 0.25],
            "label": "bottles",
            "score": 0.71
        }, {
            "box": [0.0, 0.0, 0.5, 0.5],
            "label": "fragments",
            "score": 0.71
        }
    ]
    class_names = ["others", "fragments", "chicken", "bottles"]
    predictions = predict_and_format_image(image, class_names)
    assert predictions == [
        {
            "box": [0.0, 0.0, 0.25, 0.25],
            "label": "others",
            "score": 0.71
        }, {
            "box": [0.0, 0.0, 0.5, 0.5],
            "label": "chicken",
            "score": 0.71
        }
    ]
