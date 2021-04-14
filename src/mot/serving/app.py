import json
import os

import cv2
import numpy as np
from flask import Flask, render_template, request

from mot.serving.constants import TMP_IMAGE_NAME, UPLOAD_FOLDER
from mot.serving.inference import (
    detect_and_track_images, predict_and_format_image, predict_image_file
)
from mot.serving.viz import draw_boxes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/tracking', methods=['GET', 'POST'])
def tracking():
    if request.method == "GET":
        # landing page on browser
        return render_template("upload.html")
    return detect_and_track_images(request.files['file'], app.config["UPLOAD_FOLDER"])


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Route to upload an image and visualize the prediction of a localizer."""
    if request.method == "GET":
        # landing page on browser
        return render_template("upload_image.html")

    analysis_results = predict_image_file(request.files["file"], app.config["UPLOAD_FOLDER"])
    draw_boxes(analysis_results["full_filepath"], analysis_results["detected_trash"])
    return render_template("image.html", filename=analysis_results["full_filepath"])


@app.route('/image', methods=['POST'])
def image():
    """Route to upload an image file or a JSON image in BGR and get the prediction of a localizer."""
    if "file" in request.files:
        return predict_image_file(request.files["file"], app.config["UPLOAD_FOLDER"])
    else:
        data = json.loads(request.data.decode("utf-8"))
        if "image" not in data:
            return {
                "error":
                    "Your JSON must have a field image with the image as an array in RGB"
            }
        image = np.array(data["image"])
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], TMP_IMAGE_NAME)
        cv2.imwrite(image_path, image)
        detected_trash = predict_and_format_image(image)
        return {"detected_trash": detected_trash}


if __name__ == "__main__":
    app.run(threaded=True, port=5000, debug=False, host="0.0.0.0")
