import json
import os
import shutil
from unittest import mock

import cv2
import numpy as np
import pytest

from mot.serving.app import app
from mot.serving.constants import TMP_IMAGE_NAME


def mock_post_tensorpack_localizer(*args, **kwargs):
    boxes = [[0, 0, 120, 40], [0, 0, 120, 80]]
    scores = [[0.7, 0.01, 0.01], [0.1, 0.1, 0.6]]
    classes = [1, 3]

    class Response(mock.Mock):
        json_text = {
            "outputs":
                {
                    "output/boxes:0": boxes,
                    "output/scores:0": scores,
                    "output/labels:0": classes,
                }
        }

        @property
        def text(self):
            return json.dumps(self.json_text)

        def json(self):
            return self.json_text

    response = Response()
    return response


@mock.patch("requests.post", side_effect=mock_post_tensorpack_localizer)
def test_app_post_tracking(mock_server_result, tmpdir):
    images_folder = os.path.join(tmpdir, "images")
    os.makedirs(images_folder)
    for i in range(5):
        image = 255 * np.random.rand(300, 200, 3)
        image_path = os.path.join(images_folder, "{}.jpg".format(str(i)))
        cv2.imwrite(image_path, image)

    zip_path = os.path.join(tmpdir, "toto")
    shutil.make_archive(zip_path, "zip", images_folder)
    zip_path += ".zip"

    app_folder = os.path.join(tmpdir, "app_folder")
    os.makedirs(app_folder)
    app.config["UPLOAD_FOLDER"] = app_folder
    with app.test_client() as c:
        response = c.post("/tracking", data={"file": (open(zip_path, "rb"), "toto.zip")})
        output = response.json
    assert response.status_code == 200
    expected_output = {
        "detected_trash":
            [
                {
                    "frame_to_box":
                        {
                            "0": [0.0, 0.0, 0.1, 0.05],
                            "1": [0.0, 0.0, 0.1, 0.05],
                            "2": [0.0, 0.0, 0.1, 0.05],
                            "3": [0.0, 0.0, 0.1, 0.05],
                            "4": [0.0, 0.0, 0.1, 0.05]
                        },
                    "id": 0,
                    "label": "bottles"
                }, {
                    "frame_to_box":
                        {
                            "0": [0.0, 0.0, 0.1, 0.1],
                            "1": [0.0, 0.0, 0.1, 0.1],
                            "2": [0.0, 0.0, 0.1, 0.1],
                            "3": [0.0, 0.0, 0.1, 0.1],
                            "4": [0.0, 0.0, 0.1, 0.1]
                        },
                    "id": 1,
                    "label": "fragments"
                }
            ],
        "fps": 4,
        "video_id": "toto.zip",
        "video_length": 5,
    }
    assert output == expected_output


@mock.patch("requests.post", side_effect=mock_post_tensorpack_localizer)
def test_app_post_demo(mock_server_result, tmpdir):
    image = 255 * np.random.rand(300, 200, 3)
    image_path = os.path.join(tmpdir, "toto.jpg")
    cv2.imwrite(image_path, image)
    app.config["UPLOAD_FOLDER"] = tmpdir
    with app.test_client() as c:
        response = c.post("/demo", data={"file": (open(image_path, "rb"), "toto.jpg")})
    assert response.status_code == 200


@mock.patch("requests.post", side_effect=mock_post_tensorpack_localizer)
def test_app_post_image(mock_server_result, tmpdir):
    image = np.ones((300, 200, 3))
    app.config["UPLOAD_FOLDER"] = tmpdir
    with app.test_client() as c:
        response = c.post("/image", json={"image": image.tolist()})
        output = response.get_json()
    assert response.status_code == 200
    expected_output = {
        "detected_trash": [{
            "box": [0.0, 0.0, 0.1, 0.05],
            "label": "bottles",
            "score": 0.7
        }]
    }
    assert output == expected_output


@pytest.mark.skip(
    reason="This test is failing in CI, but isn't critical for the behaviour of serving."
    " See https://travis-ci.com/surfriderfoundationeurope/mot/jobs/279342336 for more details."
)
def test_app_get():
    with app.test_client() as c:
        response = c.get("/lmnav")
        assert response.status_code == 404
        response = c.get("/")
        assert response.status_code == 200
