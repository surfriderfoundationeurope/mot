import os
import requests
import json
import numpy as np

from mot.object_detection.preprocessing import preprocess_for_serving


def query_tensorflow_server(signature, url):
    """Will send a REST query to the tensorflow server.

    - *signature*: A dict with the signature required by your tensorflow server. If you
    have installed the tensorflow serving package you can inspect
    the signature as follow:

    ```bash
    saved_model_cli show --dir directory_containing_the_saved_model.pb --all
    ```

    - *url*: Where you can find the tensorflow server

    Return:

    A dict with the answer signature.
    """

    url_serving = os.path.join(url, "v1/models/serving:predict")
    headers = {"content-type": "application/json"}
    json_response = requests.post(url_serving, data=json.dumps(signature), headers=headers)
    return json.loads(json_response.text)['outputs']


def localizer_tensorflow_serving_inference(image, url):
    """Preprocess and query the tensorflow serving for the localizer

    Arguments:

    - *image*: A numpy array loaded in BGR.
    - *url*: A string representing the url.

    Return:

    A dict with the predictions with the following format:

    ```python
    predictions = {
        'output/boxes:0': [[0, 0, 1, 1]],
        'output/labels:0': [[0, 0, 1]],
        'output/scores:0': [[0.98]]
    }
    ```
    """
    
    signature, ratio = preprocess_for_serving(image)
    predictions = query_tensorflow_server(signature, url)
    predictions['output/boxes:0'] = np.array(predictions['output/boxes:0'], np.int32) / ratio
    predictions['output/boxes:0'] = predictions['output/boxes:0'].tolist()
    return predictions
