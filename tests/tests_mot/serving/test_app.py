import json

import mock
from flask import request
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
        rv = c.post('/', json={"image": [
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]})
        output = rv.get_json()
    expected_output = {"detected_trash": [
    {"box":[0.0,0.0,0.1,0.1], "label":"bottles", "score":0.71},
    {"box":[0.0,0.0,0.2,0.2], "label":"fragments", "score":0.71}
    ]}
    assert output == expected_output


def test_app_get():
    with app.test_request_context('/'):
        assert request.path == '/'
