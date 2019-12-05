import io
import json
import os
import shutil

import cv2
import mock
import numpy as np
import pytest
from flask import Flask, request
from mot.serving.utils import handle_post_request
from werkzeug import FileStorage


def mock_post_tensorpack_localizer(*args, **kwargs):
    boxes = [[0, 0, 4, 4], [0, 0, 8, 8]]
    scores = [0.7, 0.7]
    classes = [0, 2]
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
        'output/boxes:0': [[0, 0, 0.01, 0.01], [0, 0, 0.02, 0.02]],
        'output/scores:0': [0.7, 0.7],
        'output/labels:0': [0, 2],
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
        'output/boxes:0': [[0, 0, 0.01, 0.01], [0, 0, 0.02, 0.02]],
        'output/scores:0': [0.7, 0.7],
        'output/labels:0': [0, 2],
    }
    assert output == expected_output


def test_handle_post_request_file_video(tmpdir):
    # TODO test good behavior when implemented
    data = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    filename = "test.png"
    filepath = os.path.join(tmpdir, filename)
    cv2.imwrite(filepath, data)
    m = mock.MagicMock()
    files = {"file": FileStorage(open(filepath, "rb"), content_type='video/mkv')}
    m.files = files
    with pytest.raises(NotImplementedError):
        with mock.patch("mot.serving.utils.request", m):
            output = handle_post_request(upload_folder=str(tmpdir))


def test_handle_post_request_file_other(tmpdir):
    data = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    filename = "test.png"
    filepath = os.path.join(tmpdir, filename)
    cv2.imwrite(filepath, data)
    m = mock.MagicMock()
    files = {"file": FileStorage(open(filepath, "rb"), content_type='application/pdf')}
    m.files = files
    shutil.rmtree(tmpdir)
    with pytest.raises(NotImplementedError):
        with mock.patch("mot.serving.utils.request", m):
            output = handle_post_request(upload_folder=str(tmpdir))
