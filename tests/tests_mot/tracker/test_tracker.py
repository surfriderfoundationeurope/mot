from mot.tracker import video_utils
from mot.tracker import tracker
import os
import numpy as np

home = os.path.expanduser("~")
PATH_TO_TEST_VIDEO = os.path.join(home, ".mot/tests/test_video.mp4")

PATH_TO_OUTPUT_SPLIT = "/tmp/test_outut_video/"
def test_split_and_open():
    '''
    split video
    '''
    if not os.path.isdir(PATH_TO_OUTPUT_SPLIT):
        os.mkdir(PATH_TO_OUTPUT_SPLIT)
    print(PATH_TO_TEST_VIDEO)
    video_utils.split_video(PATH_TO_TEST_VIDEO, PATH_TO_OUTPUT_SPLIT)
    assert len(os.listdir(PATH_TO_OUTPUT_SPLIT)) == 4
    '''
    open path and read images
    '''
    frames_array = video_utils.open_images(video_utils.read_folder(PATH_TO_OUTPUT_SPLIT))
    assert len(frames_array) == 4
    assert frames_array[0].shape == (768, 1024)


def test_track():
    if not os.path.isdir(PATH_TO_OUTPUT_SPLIT) or len(os.listdir(PATH_TO_OUTPUT_SPLIT)) != 4:
        video_utils.split_video(PATH_TO_TEST_VIDEO, PATH_TO_OUTPUT_SPLIT)

    frames_array = video_utils.open_images(video_utils.read_folder(PATH_TO_OUTPUT_SPLIT))

    camflow = tracker.CameraFlow()
    matrix = camflow.compute_transform_matrix(frames_array[0], frames_array[1])
    assert matrix.shape == (2, 3)

    '''
    transform points
    '''
    point = np.array([448., 448.])
    points = []
    points.append(point)
    matrices = camflow.compute_transform_matrices(frames_array)
    for m in matrices:
        points.append(camflow.warp_coords(points[-1], m))
    assert len(points) == 4
    print(points)
