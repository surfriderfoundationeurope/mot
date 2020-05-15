import multiprocessing


# app configuration
UPLOAD_FOLDER = "static/tmp"
TMP_IMAGE_NAME = "tmp_image.jpg"
SERVING_URL = "http://localhost:8501"  # the url where the tf-serving container exposes the model
CPU_COUNT = min(int(multiprocessing.cpu_count() / 2), 32)


# video settings
FPS = 4
RESOLUTION = (1024, 768)
SUM_THRESHOLD = 0.6  # the sum of scores for all classes must be greater than this value

# object detection settings
CLASS_NAMES = ["bottles", "others", "fragments"]
# for the prediction to be kept
CLASS_TO_THRESHOLD = {"bottles": 0.7, "others": 0.7, "fragments": 0.7}
DEFAULT_THRESHOLD = 0.5 # default threshold applied when the class isn't in CLASS_TO_THRESHOLD


CLASS_NAME_TO_COLOR = {
    "bottles": (255, 0, 0),
    "others": (0, 255, 0),
    "fragments": (0, 0, 255),
}
