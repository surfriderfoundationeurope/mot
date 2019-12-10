import json

import mock
from flask import request
from mot.serving.app import app


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
    expected_output = {
        'output/boxes:0': [[0, 0, 0.01, 0.01], [0, 0, 0.02, 0.02]],
        'output/scores:0': [0.7, 0.7],
        'output/labels:0': [0, 2],
    }
    assert output == expected_output


def test_app_get():
    with app.test_request_context('/'):
        assert request.path == '/'
