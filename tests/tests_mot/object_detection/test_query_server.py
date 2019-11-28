import json

import numpy as np
from unittest import mock
from mot.object_detection.query_server import localizer_tensorflow_serving_inference


def mock_post_tensorpack_localizer(*args, **kwargs):
    boxes = [[0, 0, 4, 4], [0, 0, 8, 8]]
    scores = [0.7, 0.7]
    classes = [0, 2]
    response = mock.Mock()
    response.text = json.dumps({
        'outputs': {
            'output/boxes:0': boxes,
            'output/scores:0': scores,
            'output/labels:0': classes,
        }
    })
    return response


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer)
def test_localizer_tensorflow_serving_inference(mock_server_result):
    image = np.zeros((200, 200, 3))

    expected_output = {
            'output/boxes:0': [[0, 0, 1, 1], [0, 0, 2, 2]],
            'output/scores:0': [0.7, 0.7],
            'output/labels:0': [0, 2],
    }

    output = localizer_tensorflow_serving_inference(image, 'http:localhost:8899')

    assert output == expected_output
