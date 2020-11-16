import json
import os
from typing import Dict

import numpy as np
import requests
from tensorpack import logger

from mot.object_detection.preprocessing import preprocess_for_serving


def query_tensorflow_server(signature: Dict, url: str) -> Dict:
    """Will send a REST query to the tensorflow server.

    Arguments:

    - *signature*: A dict with the signature required by your tensorflow server. If you
    have installed the tensorflow serving package you can inspect
    the signature as follow:

    ```bash
    saved_model_cli show --dir directory_containing_the_saved_model.pb --all
    ```

    - *url*: Where you can find the tensorflow server

    Returns:

    - *Dict*: A dict with the answer signature.
    """
    url_serving = os.path.join(url, "v1/models/serving:predict")
    headers = {"content-type": "application/json"}
    json_response = requests.post(url_serving, data=json.dumps(signature), headers=headers)
    response = json_response.json()
    if "outputs" in response:
        return response["outputs"]
    if "error" in response:
        message = "Tensorflow serving returned an error. It probably means that your SavedModel " \
        " doesn't have the correct signature_def, which must be 'serving_default'. You can inspect " \
        " that by doing `saved_model_cli show --dir /path/to/your/model --all`. If the signature_def" \
        " isn't 'serving_default' you should export your checkpoints following the instructions" \
        " in the main README, section `Export` Here is the original" \
        " error raised by tensorflow serving:\n " + str(response["error"])
        raise ValueError(message)
    raise ValueError("Unknwon response from tensorflow serving: {}".format(response))


def localizer_tensorflow_serving_inference(
    image: np.ndarray,
    url: str,
    return_all_scores: bool = False,
) -> Dict:
    """Preprocess and query the tensorflow serving for the localizer

    Arguments:

    - *image*: A numpy array loaded in BGR.
    - *url*: A string representing the url.
    - *return_all_scores*: Wheter to return scores for all classes.
        The SavedModel you're querying must return all scores.

    Return:

    - *Dict*: A dict with the predictions with the following format:

    ```python
    if return_all_scores:
        predictions = {
            'output/boxes:0': [[0, 0, 1, 1], [0, 0, 10, 10], [10, 10, 15, 100]], (y1, x1, y2, x2)
            'output/labels:0': [3, 1, 2],  # the labels start at 1 since 0 is for background
            'output/scores:0': [
                [0.001, 0.001, 0.98],
                [0.87, 0.05, 0.03],
                [0.1, 0.76, 0.1],
            ]  # sorted in descending order of the prediction
        }
    else:
        predictions = {
            'output/boxes:0': [[0, 0, 1, 1], [0, 0, 10, 10], [10, 10, 15, 100]],
            'output/labels:0': [3, 1, 2], # the labels start at 1 since 0 is for background
            'output/scores:0': [0.98, 0.87, 0.76] # sorted in descending order
        }
    ```
    """
    signature, ratio = preprocess_for_serving(image)
    predictions = query_tensorflow_server(signature, url)
    scores = np.array(predictions['output/scores:0'])
    if len(predictions["output/boxes:0"]) > 0:
        predictions['output/boxes:0'] = np.array(predictions['output/boxes:0'], np.int32) / ratio
        predictions["output/boxes:0"][:, 0] /= image.shape[0] # scaling coords to [0, 1]
        predictions["output/boxes:0"][:, 1] /= image.shape[1] # scaling coords to [0, 1]
        predictions["output/boxes:0"][:, 2] /= image.shape[0] # scaling coords to [0, 1]
        predictions["output/boxes:0"][:, 3] /= image.shape[1] # scaling coords to [0, 1]
        predictions['output/boxes:0'] = predictions['output/boxes:0'].tolist()
        if return_all_scores and len(scores.shape) == 1:
            raise ValueError(
                "return_all_scores is True but the model you're using only returns the score of "
                "the predicted entity. Try changing the model you're using."
            )
    if not return_all_scores and len(scores.shape) == 2:
        scores = np.max(scores, axis=1)
    predictions['output/scores:0'] = scores.tolist()
    return predictions
