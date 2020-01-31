import os

import numpy as np

from mot.object_detection.config import config as cfg
from mot.tracker import object_tracking


def test_similarity():
    detection1 = [np.array([0., 0.,1.,0.]), np.array([0.56, 0.38, 0.6, 0.41]), 4]
    detection2 = [np.array([0., 0.,1.,0.]), np.array([0.56, 0.38, 0.6, 0.41]), 3]
    assert object_tracking.similarity(detection1, detection2) == 1.0
    detection1 = [np.array([0., 0.,1.,0.]), np.array([0.70, 0.38, 0.74, 0.41]), 4]
    detection2 = [np.array([0., 0.,1.,0.]), np.array([0.10, 0.38, 0.14, 0.41]), 3]
    assert object_tracking.similarity(detection1, detection2) < 0.75
    detection1 = [np.array([0., 0.,1.,0.]), np.array([0.56, 0.38, 0.6, 0.41]), 4]
    detection2 = [np.array([0., 0.,1.,0.]), np.array([0.56, 0.38, 0.6, 0.41]), 3]
    assert object_tracking.similarity(detection1, detection2) == 1.0
    detection1 = [np.array([0., 0.,1.,0.]), np.array([0.4, 0.3, 0.6, 0.4]), 4]
    detection2 = [np.array([0., 0.,1.,0.]), np.array([0.0, 0.34, 1.0, 0.36]), 3]
    assert abs(object_tracking.similarity(detection1, detection2) - 0.8) < 0.0001

def test_build_tracklets():
    object_tracker = object_tracking.ObjectTracking("test_video", [], [], fps=1)
    test_input_detections = [
        {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.19, 0.00], [0.0, 0.0, 1.0], [0.1, 0.89, 0.01]]
        }, {},
        {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.19, 0.00], [0.0, 0.0, 1.0], [0.1, 0.89, 0.01]]
        }, {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.19, 0.00], [0.0, 0.0, 1.0], [0.1, 0.89, 0.01]]
        }
    ]

    tracklets = object_tracker.build_tracklets(test_input_detections, time_window = 2, matching_threshold = 0.7)
    assert len(tracklets)==3
    assert len(tracklets[0].boxes) == 3

    test_input_detections = [
        {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.19, 0.00], [0.0, 0.0, 1.0], [0.1, 0.89, 0.01]]
        }, {}, {},
        {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.19, 0.00], [0.0, 0.0, 1.0], [0.1, 0.89, 0.01]]
        }, {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.19, 0.00], [0.0, 0.0, 1.0], [0.1, 0.89, 0.01]]
        }
    ]

    tracklets = object_tracker.build_tracklets(test_input_detections, time_window = 2, matching_threshold = 0.7)
    assert len(tracklets)==6
    assert len(tracklets[0].boxes) == 1


def test_track_objects():
    test_image_list = ["mock_frame_1", "mock_frame_2", "mock_frame_3"]
    test_inference_data = [
        {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.524, 0.186, 0.618, 0.210], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.68, 0.62], [0.81, 0.68, 0.62], [0.81, 0.68, 0.62]]
        }, {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.68, 0.62], [0.81, 0.68, 0.62]]
        }, {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41], [0.459, 0.357, 0.507, 0.381]],
            "output/scores:0": [[0.81, 0.68, 0.62], [0.81, 0.68, 0.62]]
        }
    ]

    object_tracker = object_tracking.ObjectTracking(
        "test_video", test_image_list, test_inference_data, fps=1
    )
    tracklets = object_tracker.compute_tracks()
    assert len(tracklets) == 2


def test_json_output():
    test_inference_data = [
        {
            "output/boxes:0": [[0.56, 0.38, 0.6, 0.41]],
            "output/scores:0": [[0.81]]
        }
    ]

    object_tracker = object_tracking.ObjectTracking(
        "test_video", ["mock_frame"], test_inference_data, fps=1
    )
    tracklet = object_tracking.Track(0, [0.0, 0.0, 0.81], [0.56, 0.38, 0.6, 0.41], 0)
    json_result = object_tracker.json_result([tracklet])

    # yapf: disable
    assert json_result == {
        "video_length": 1,
        "fps": 1,
        "video_id": "test_video",
        "detected_trash":
            [{
                "label": "fragments",
                "id": 0,
                "score": 0.81,
                "frame_to_box": {
                    0: [0.56, 0.38, 0.6, 0.41]
                }
            }]
    }
    # yapf: enable


def test_track_json_result():
    track = object_tracking.Track(id=1, class_scores=[0.,1.], box=[1, 3, 2, 10], frame=7)
    assert track.json_result(class_names=["requin", "poisson"]) == {
        "label": "poisson",
        "frame_to_box": {
            7: [1, 3, 2, 10]
        },
        "score": 1.0,
        "id": 1
    }
    track.add_matching_detection([0.,0.8],[1.00004, 4, 5, 10], 8)
    assert track.json_result(class_names=["requin", "poisson"]) == {
        "label": "poisson",
        "frame_to_box": {
            7: [1, 3, 2, 10],
            8: [1.0, 4, 5, 10]
        },
        "score": 0.9,
        "id": 1
    }
