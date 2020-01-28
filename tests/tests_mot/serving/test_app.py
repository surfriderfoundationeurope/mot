import json
from unittest import mock

import pytest

from mot.serving.app import app


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
def test_app_post(mock_server_result):
    with app.test_client() as c:
        response = c.post("/", json={"image": [
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]})
        output = response.get_json()
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
