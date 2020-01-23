import os

import numpy as np

from mot.object_detection.config import config as cfg
from mot.tracker import tracker


def test_find_best_match():
    test_trash = tracker.Trash(1, 1, [558.1, 382.1, 597.1, 415.1], 4)
    trash_list = [
        tracker.Trash(0, 1, [558.1, 382.1, 597.1, 415.1], 2),
        tracker.Trash(1, 1, [530, 360, 591, 414], 2),
        tracker.Trash(2, 2, [558.1, 382.1, 597.1, 415.1], 3),
    ]

    matching_id = test_trash.find_best_match_in_list(trash_list, 0.3)
    assert matching_id == 0


def test_potential_matching_trash_list():
    # 3 Trash, 2 on first frame, 1 on second frame
    test_trash_list = [
        tracker.Trash(0, 1, [558.1, 382.1, 597.1, 415.1], 0),
        tracker.Trash(1, 1, [530, 360, 591, 414], 0),
        tracker.Trash(2, 2, [558.1, 382.1, 597.1, 415.1], 1),
    ]

    test_objects_per_frame_list = [[0, 1], [2]]

    object_tracker = tracker.ObjectTracking("test_video", [], [], fps=1)

    # find objects anterior to second frame
    potential_list = object_tracker.potential_matching_trash_list(
        1, test_trash_list, test_objects_per_frame_list
    )
    assert len(potential_list) == 2


def test_track_objects():
    test_image_list = ["mock_frame_1", "mock_frame_2", "mock_frame_3"]
    test_inference_data = [
        {
            "output/boxes:0": [[558, 382, 597, 415], [524, 186, 618, 210], [459, 357, 507, 381]],
            "output/labels:0": [3, 1, 2],
            "output/scores:0": [0.81, 0.68, 0.62]
        }, {
            "output/boxes:0": [[558, 382, 597, 415], [524, 186, 618, 210], [459, 357, 507, 381]],
            "output/labels:0": [3, 1, 2],
            "output/scores:0": [0.81, 0.68, 0.62]
        }, {
            "output/boxes:0": [[558, 382, 597, 415], [524, 186, 618, 210], [459, 357, 507, 381]],
            "output/labels:0": [3, 1, 2],
            "output/scores:0": [0.81, 0.68, 0.62]
        }
    ]

    object_tracker = tracker.ObjectTracking(
        "test_video", test_image_list, test_inference_data, fps=1
    )
    trash_list = object_tracker.track_objects()
    assert len(trash_list) == 3
    found_labels_dict = {1: False, 2: False, 3: False}
    for trash in trash_list:
        found_labels_dict[trash.label] = True
    assert found_labels_dict == {1: True, 2: True, 3: True}


def test_json_output():
    test_inference_data = [
        {
            "output/boxes:0": [[558, 382, 597, 415]],
            "output/labels:0": [3],
            "output/scores:0": [0.81]
        }
    ]

    object_tracker = tracker.ObjectTracking(
        "test_video", ["mock_frame"], test_inference_data, fps=1
    )
    json_result = object_tracker.json_result()

    # yapf: disable
    assert json_result == {
        "video_length": 1,
        "fps": 1,
        "video_id": "test_video",
        "detected_trash":
            [{
                "label": "fragments",
                "id": 0,
                "frame_to_box": {
                    0: [558, 382, 597, 415]
                }
            }]
    }
    # yapf: enable


def test_trash_json_result():
    trash = tracker.Trash(id=1, label=2, box=[1, 3, 2, 10], frame=7)
    assert trash.json_result(class_names=["requin", "poisson"]) == {
        "label": "poisson",
        "frame_to_box": {
            7: [1, 3, 2, 10]
        },
        "id": 1
    }
    trash.add_matching_object([1.00004, 4, 5, 10], 8)
    assert trash.json_result(class_names=["requin", "poisson"]) == {
        "label": "poisson",
        "frame_to_box": {
            7: [1, 3, 2, 10],
            8: [1.0, 4, 5, 10]
        },
        "id": 1
    }
