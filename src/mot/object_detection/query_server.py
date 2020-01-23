import json
import os
from typing import Dict

import numpy as np
import requests
from tensorpack import logger

from mot.object_detection.preprocessing import preprocess_for_serving


def query_tensorflow_server(signature: Dict[str, object], url: str) -> Dict[str, object]:
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

    - *Dict[str, object]*: A dict with the answer signature.
    """
    url_serving = os.path.join(url, "v1/models/serving:predict")
    headers = {"content-type": "application/json"}
    json_response = requests.post(url_serving, data=json.dumps(signature), headers=headers)
    return json.loads(json_response.text)['outputs']


def localizer_tensorflow_serving_inference(image: np.ndarray, url: str) -> Dict[str, object]:
    """Preprocess and query the tensorflow serving for the localizer

    Arguments:

    - *image*: A numpy array loaded in BGR.
    - *url*: A string representing the url.

    Return:

    - *Dict[str, object]*: A dict with the predictions with the following format:

    ```python
    predictions = {
        'output/boxes:0': [[0, 0, 1, 1], [0, 0, 10, 10], [10, 10, 15, 100]],
        'output/labels:0': [3, 1, 2], # the labels start at 1 since 0 is for background
        'output/scores:0': [0.98, 0.87, 0.76] # sorted in descending order
    }
    ```
    """
    signature, ratio = preprocess_for_serving(image)
    predictions = query_tensorflow_server(signature, url)
    predictions['output/boxes:0'] = np.array(predictions['output/boxes:0'], np.int32) / ratio
    predictions['output/boxes:0'] = predictions['output/boxes:0'].tolist()
    return predictions
