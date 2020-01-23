from mot.tracker import video_utils
from mot.tracker import camera_flow
import os
import numpy as np
import shutil
import cv2

home = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(home, ".mot/tests/test_video.mp4")

PATH_TO_OUTPUT_SPLIT = "/tmp/test_outut_video/"

def test_camera_flow(tmpdir):
    video_utils.split_video(PATH_TO_TEST_VIDEO, tmpdir, fps=2)

    frames_array = [cv2.cvtColor(cv2.imread(im_path),cv2.COLOR_BGR2GRAY)
           for im_path in video_utils.read_folder(tmpdir)]

    camflow = camera_flow.CameraFlow()
    matrix = camflow.compute_transform_matrix(frames_array[0], frames_array[1])
    assert matrix.shape == (2, 3)

    # test transform points

    point = np.array([448., 448.])
    points = []
    points.append(point)
    matrices = camflow.compute_transform_matrices(frames_array)
    for m in matrices:
        points.append(camflow.warp_coords(points[-1], m))
    assert len(points) == 7 or len(points) == 6
