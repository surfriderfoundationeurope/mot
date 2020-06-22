import copy
from typing import List

import math
import numpy as np
from collections import Counter

from mot.object_detection.utils import np_box_ops
from mot.tracker import object_tracking

def compute_scores(tracks: List[object_tracking.Track], tracks_gt: List[object_tracking.Track]):
    ''' Computes the tracking matches between a list of tracklets and a list of tracklets ground truths

    ** Bo Wu and Ram Nevatia. Tracking of multiple, partially occluded humans based on static body part detection, CVPR 2006 **

    - Mostly Tracked (MT) trajectories: number of ground-truth trajectories that are correctly tracked in at least 50%
    of the frames.
    - Over Tracked (OT) trajectories: number of ground-truth trajectories that are tracked by several (>1) different tracks
    - Un-tracked (UT) trajectories: number of ground-truth trajectories that are not matched with a single trajectory
    - False trajectories (FT): predicted trajectories which do not correspond to a real object (i.e. to a ground truth
    trajectory).
    '''
    score_MT = 0.
    score_OT = 0.
    score_UT = 0.
    remaining_tracks = copy.deepcopy(tracks)
    for track_gt in tracks_gt:
        label_gt = track_gt.get_label()
        matched_tracks = []
        for track_idx, track in enumerate(remaining_tracks):
            if track_gt.contains_subtrack(track):
                matched_tracks.append(track_idx)
        for index in sorted(matched_tracks, reverse=True):
            del remaining_tracks[index]
        if matched_tracks == []:
            score_UT += 1.
        elif len(matched_tracks) == 1:
            score_MT += 1.
        else:
            score_OT += 1.
    score_UT /= len(tracks_gt)
    score_OT /= len(tracks_gt)
    score_MT /= len(tracks_gt)
    return score_MT, score_OT, score_UT


def count_match(tracks: List[object_tracking.Track], tracks_gt: List[object_tracking.Track]):
    ''' Computes the tracking matches between a list of tracklets and a list of tracklets ground truths
    Only based on the number of detected object in the section
    '''
    num_gt = Counter([track.get_label() for track in tracks_gt])
    num = Counter([track.get_label() for track in tracks])
    num_gt.subtract(num)
    return dict(num_gt)
