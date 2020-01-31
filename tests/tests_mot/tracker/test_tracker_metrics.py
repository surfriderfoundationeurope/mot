import os

import numpy as np
import random

from mot.object_detection.config import config as cfg
from mot.tracker import tracker_metrics, object_tracking

def build_test_track(score, box, random_modifs=False):
    random.seed(0)
    track = object_tracking.Track(id=1, class_scores=score[:], box=box[:], frame=7)
    for i in range(10):
        # Skipping 20% of frames
        if not random_modifs or random.random()<0.8:
            new_score = score[:]
            new_box = box[:]
            if random_modifs and random.random()<0.8:
                score_offset = (random.random() -0.5) / 2.
                new_score[0] -= score_offset
                new_score[1] += score_offset
            if random_modifs and random.random()<0.8:
                box_offsets = [(random.random() -0.5) / 2. for i in range(4)]
                new_box = [min(1., max(0., new_box[i] + box_offsets[i])) for i in range(4)]
            track.add_matching_detection(new_score, new_box, 8+i)
    return track

def test_count_match():
    tracks_gt = [build_test_track([0., 1.], [0.1, 0.3, 0.2, 0.4]),
                 build_test_track([0.05, 0.95], [0.12, 0.32, 0.22, 0.42]),
                 build_test_track([1., 0.], [0.4, 0.5, 0.45, 0.6])]

    tracks = [build_test_track([0.45, 0.55], [0.1, 0.3, 0.2, 0.4]),
              build_test_track([0.75, 0.25], [0.12, 0.32, 0.22, 0.42]),
              build_test_track([1., 0.], [0.4, 0.5, 0.45, 0.6])]

    tracks_noisy = [build_test_track([0., 1.], [0.1, 0.3, 0.2, 0.4], True),
                    build_test_track([0.05, 0.95], [0.12, 0.32, 0.22, 0.42], True),
                    build_test_track([1., 0.], [0.4, 0.5, 0.45, 0.6], True)]

    count_diff = tracker_metrics.count_match(tracks_gt, tracks_gt)
    assert count_diff == {1:0, 2:0}

    count_diff = tracker_metrics.count_match(tracks, tracks_gt)
    assert count_diff == {1:-1, 2:1}

    count_diff = tracker_metrics.count_match(tracks_noisy, tracks_gt)
    assert count_diff == {1:0, 2:0}

def test_match():
    tracks_gt = [build_test_track([0., 1.], [0.1, 0.3, 0.2, 0.4]),
                 build_test_track([0.05, 0.95], [0.52, 0.32, 0.62, 0.42]),
                 build_test_track([1., 0.], [0.4, 0.5, 0.45, 0.6])]

    tracks = [build_test_track([0., 1.], [0.1, 0.3, 0.2, 0.4], True),
              build_test_track([0.05, 0.95], [0.52, 0.32, 0.62, 0.42], True),
              build_test_track([1., 0.], [0.4, 0.5, 0.45, 0.6], True)]

    score_MT, score_OT, score_UT = tracker_metrics.compute_scores(tracks_gt, tracks_gt)
    print("score_MT, score_OT, score_UT", score_MT, score_OT, score_UT)
    assert score_MT == 1.
    assert score_OT == 0.
    assert score_UT == 0.
