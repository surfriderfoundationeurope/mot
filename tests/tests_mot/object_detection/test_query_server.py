import json
from unittest import mock

import numpy as np
import pytest

from mot.object_detection.query_server import \
    localizer_tensorflow_serving_inference


def mock_post_tensorpack_localizer_prediction_score(*args, **kwargs):
    boxes = [[0, 0, 120, 40], [0, 0, 120, 80]]
    scores = [0.7, 0.7]
    classes = [0, 2]

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


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer_prediction_score)
def test_localizer_tensorflow_serving_inference_prediction_score(mock_server_result):
    image = np.zeros((300, 200, 3))

    expected_output = {
        'output/boxes:0': [[0, 0, 0.1, 0.05], [0, 0, 0.1, 0.1]],
        'output/scores:0': [0.7, 0.7],
        'output/labels:0': [0, 2],
    }

    output = localizer_tensorflow_serving_inference(
        image,
        'http:localhost:8899',
        return_all_scores=False,
    )

    assert output == expected_output

    with pytest.raises(ValueError):
        localizer_tensorflow_serving_inference(
            image,
            'http:localhost:8899',
            return_all_scores=True,
        )


def mock_post_tensorpack_localizer_all_scores(*args, **kwargs):
    boxes = [[0, 0, 120, 40], [0, 0, 120, 80]]
    scores = [
        [0.7, 0.1, 0.1],
        [0.2, 0.05, 0.7],
    ]
    classes = [0, 2]

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


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer_all_scores)
def test_localizer_tensorflow_serving_inference_all_scores(mock_server_result):
    image = np.zeros((300, 200, 3))

    expected_output = {
        'output/boxes:0': [[0, 0, 0.1, 0.05], [0, 0, 0.1, 0.1]],
        'output/scores:0': [
            [0.7, 0.1, 0.1],
            [0.2, 0.05, 0.7],
        ],
        'output/labels:0': [0, 2],
    }
    output = localizer_tensorflow_serving_inference(
        image,
        'http:localhost:8899',
        return_all_scores=True,
    )
    assert output == expected_output

    expected_output = {
        'output/boxes:0': [[0, 0, 0.1, 0.05], [0, 0, 0.1, 0.1]],
        'output/scores:0': [0.7, 0.7],
        'output/labels:0': [0, 2],
    }
    output = localizer_tensorflow_serving_inference(
        image,
        'http:localhost:8899',
        return_all_scores=False,
    )
    assert output == expected_output

def mock_post_tensorpack_localizer_no_pred(*args, **kwargs):
    boxes = []
    scores = []
    classes = []

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


@mock.patch('requests.post', side_effect=mock_post_tensorpack_localizer_no_pred)
def test_localizer_tensorflow_serving_inference_no_pred(mock_server_result):
    image = np.zeros((300, 200, 3))

    expected_output = {
        'output/boxes:0': [],
        'output/scores:0': [],
        'output/labels:0': [],
    }
    output = localizer_tensorflow_serving_inference(
        image,
        'http:localhost:8899',
        return_all_scores=True,
    )
    assert output == expected_output

    expected_output = {
        'output/boxes:0': [],
        'output/scores:0': [],
        'output/labels:0': [],
    }
    output = localizer_tensorflow_serving_inference(
        image,
        'http:localhost:8899',
        return_all_scores=False,
    )
    assert output == expected_output
