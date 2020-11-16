import os
import shutil

import numpy as np

from mot.tracker import video_utils

home = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(home, ".mot/tests/test_video.mp4")


def test_split_video(tmpdir):
    output_folder = os.path.join(tmpdir, "output")
    video_utils.split_video(PATH_TO_TEST_VIDEO, output_folder, fps=2)

    assert os.path.isdir(output_folder)
    # Different versions of FFMPEG yield different results when splitting the video
    assert len(os.listdir(output_folder)) == 6 or len(os.listdir(output_folder)) == 7
