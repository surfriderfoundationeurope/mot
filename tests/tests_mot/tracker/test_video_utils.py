from mot.tracker import video_utils
import os
import numpy as np
import shutil

home = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(home, ".mot/tests/test_video.mp4")

PATH_TO_OUTPUT_SPLIT = "/tmp/test_outut_video/"
def test_split_video():

    if os.path.isdir(PATH_TO_OUTPUT_SPLIT):
        shutil.rmtree(PATH_TO_OUTPUT_SPLIT)
    os.mkdir(PATH_TO_OUTPUT_SPLIT)

    video_utils.split_video(PATH_TO_TEST_VIDEO, PATH_TO_OUTPUT_SPLIT, fps=2)
    # Different versions of FFMPEG yield different results when splitting the video
    assert len(os.listdir(PATH_TO_OUTPUT_SPLIT)) == 6 or len(os.listdir(PATH_TO_OUTPUT_SPLIT)) == 7
