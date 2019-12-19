from mot.tracker import video_utils
import os
import numpy as np
import shutil

home = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(home, ".mot/tests/test_video.mp4")

def test_split_video(tmpdir):
    video_utils.split_video(PATH_TO_TEST_VIDEO, tmpdir, fps=2)
    # Different versions of FFMPEG yield different results when splitting the video
    assert len(os.listdir(tmpdir)) == 6 or len(os.listdir(tmpdir)) == 7
