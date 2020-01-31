import copy
from typing import List

import math
import numpy as np
from cached_property import cached_property

from mot.object_detection.utils import np_box_ops
from mot.tracker.tracker_utils import ratio, area, center, center_dist


def similarity(new_detection, old_detection):
    new_scores, new_box, new_idx = new_detection
    old_scores, old_box, old_idx = old_detection
    scorediff = np.mean(np.abs(new_scores - old_scores))
    ratiodiff = min(1.0, abs(ratio(new_box) - ratio(old_box)))
    sizediff = min(1.0, abs(area(new_box) - area(old_box))/ max(area(new_box), area(old_box)))
    centerdiffx = min(1.0, abs(center(new_box)[0] - center(old_box)[0]))
    centerdiffy = min(1.0, abs(center(new_box)[1] - center(old_box)[1]))
    framediff = min(1.0, (new_idx-1-old_idx)/3)
    return 1.0 - 0.5 * scorediff - 0.2 * ratiodiff - 0.2 * sizediff - 0.5 * centerdiffx - 0.2 * centerdiffy - 0.2*framediff

class Track():
    '''Track (or trajectory) class mainly defined by a sequence of frames and the corresponding detections

    Multi-object tracking notations:
    - The output of the detector are called "detections"
    - The matching of successive boxes are called "tracklets" which are small tracks
    - The final trajectories are called "tracks"
    '''

    def __init__(self, id: int, class_scores: List[float], box: List[float], frame: int):
        self.id = id
        self.scores = [class_scores]
        self.boxes = [box]
        self.frames = [frame]
        self.speed = None
        self.track_score = max(class_scores)

        self.THR_SPEED = 0.0
        self.THR_BOX_CENTER = 0.05
        self.THR_BOX_RATIO = 0.1

    def add_matching_detection(self, scores, box, frame):
        self.scores.append(scores)
        self.boxes.append(box)
        self.frames.append(frame)
        if len(self.boxes) > 2:
            self.speed = self.compute_speed()
        self.track_score = np.max(np.mean(np.array(self.scores), axis=0))

    def get_latest_np_box(self):
        return np.array(self.boxes[-1])

    def get_latest_detection(self, apply_speed, new_frame_id):
        last_box = np.array(self.boxes[-1])
        frame_offset = new_frame_id - self.frames[-1]
        if apply_speed and self.speed is not None:
            last_box = np.clip(last_box + self.speed * frame_offset, 0.0, 1.0)
        return [np.array(self.scores[-1]), last_box, self.frames[-1]]

    def is_in_range(self, frame_idx, time_window):
        return (frame_idx - self.frames[-1]) <= time_window

    def is_valid(self, min_length):
        return len(self.frames) >= min_length

    def has_valid_speed(self, speed):
        speed_dot = np.dot(speed, self.speed[0:2])
        return np.sum(speed_dot) > self.THR_SPEED

    def get_center(self):
        x1, y1, x2, y2 = self.boxes[-1]
        return (x2 + x1) / 2, (y2 + y1) / 2

    def get_average_scores(self):
        return np.mean(np.array(self.scores), axis=0)

    def get_label(self):
        return np.argmax(np.array(self.scores).sum(axis=0)) + 1

    def compute_speed(self):
        boxes_array = np.array(self.boxes)
        time_differences = np.array([float(y-x) for x,y in zip(self.frames[1:], self.frames[0:-1])])
        speeds = (boxes_array[1:, :] - boxes_array[0:-1, :]).T / time_differences
        speed = np.mean(speeds, axis = -1)
        vx = (speed[0]+speed[2])/2
        vy = (speed[1]+speed[3])/2
        return np.array([vx, vy, vx, vy])

    def compatibility(self, track):
        """Computes a soft compatibility score between this track and a later one.

        Arguments:

        - track: the newer track to compare to this one.

        Returns:

        - A floating point value corresponding to the compatibility. -1 is not compatible
        """
        if self.frames[-1] >= track.frames[0]:
            return -1.0

        projected_box = self.get_latest_detection(apply_speed=True, new_frame_id=track.frames[0])
        old_detection = [self.get_average_scores(), projected_box, self.frames[-1]]
        new_detection = [track.get_average_scores(), track.boxes[0], track.frames[0]]
        return similarity(old_detection, new_detection)

    def contains_subtrack(self, track):
        """Compare two tracks and verify if the track is a subpart of this one

        Arguments:

        - track: the newer track to compare to this one.

        Returns:

        - True if it matches, false otherwise
        """
        if self.get_label() != track.get_label():
            return False
        common_frames = set(self.frames).intersection(set(track.frames))
        match_box = 0.
        for frame in common_frames:
            box1 = self.boxes[self.frames.index(frame)]
            box2 = track.boxes[track.frames.index(frame)]
            if center_dist(box1, box2) < self.THR_BOX_CENTER and \
               abs(ratio(box1) - ratio(box2)) < self.THR_BOX_RATIO:
               match_box += 1.
        match_ratio = match_box / len(track.frames)
        if match_ratio > 0.2:
            return True
        return False

    def append_track(self, track):
        self.scores.extend(track.scores)
        self.boxes.extend(track.boxes)
        self.frames.extend(track.frames)
        if len(self.boxes) > 2:
            self.speed = self.compute_speed()
        self.track_score = np.max(np.mean(np.array(self.scores), axis=0))


    def __repr__(self):
        return "(id:{}, label:{}, center:({:.1f},{:.1f}), frames:{})".format(
            self.id, self.get_label(),
            self.get_center()[0],
            self.get_center()[1], self.frames
        )

    def json_result(self, class_names=["bottles", "others", "fragments"]):
        class_names = ["BG"] + class_names
        rounded_boxes = [[round(coord, 2) for coord in box] for box in self.boxes]
        return {
            "label": class_names[self.get_label()],
            "score": self.track_score,
            "frame_to_box": {frame: box for frame, box in zip(self.frames, rounded_boxes)},
            "id": self.id
        }


class ObjectTracking():
    '''Wrapper class to tracking trash objects in video output frames
    '''

    def __init__(
        self, video_id, list_path_images, list_inference_output=None, fps=2, list_geoloc=None
    ):
        self.video_id = video_id

        self.list_path_images = list_path_images
        self.list_inference_output = list_inference_output

        self.num_images = len(list_path_images)
        self.list_geoloc = list_geoloc
        self.fps = fps

        self.tracking_done = False

        self.iou_threshold = 0.3
        self.rewind_window_match = 2

    def build_tracklet_similarity_matrix(self, tracklets:List):
        """Builds a compatibility matrix between tracklets

        Arguments:

        - new_scores: list of [classes scores] of length N
        - new_boxes: list of [4 coordinates] of length N
        - idx: integer frame idx
        - potential_matching_tracklets: list of tracklets of length M

        Returns:
        - a similarity matrix (numpy array) of shape (N, M)

        """
        nb_tracklets = len(tracklets)
        if nb_tracklets == 0:
            return None
        m = np.zeros((nb_tracklets, nb_tracklets))
        for i in range(nb_tracklets):
            for j in range(i, nb_tracklets):
                m[i,j] = tracklets[i].compatibility(tracklets[j])
        return m

    def build_similarity_matrix(self, new_scores: List, new_boxes: List, idx: int, potential_matching_tracklets:List):
        """Builds a similarity matrix between new scores and boxes and existing tracklets

        Arguments:

        - new_scores: list of [classes scores] of length N
        - new_boxes: list of [4 coordinates] of length N
        - idx: integer frame idx
        - potential_matching_tracklets: list of tracklets of length M

        Returns:
        - a similarity matrix (numpy array) of shape (N, M)

        """
        nb_new = len(new_boxes)
        nb_old = len(potential_matching_tracklets)
        if nb_old == 0 or nb_new == 0:
            return None
        m = np.zeros((nb_new, nb_old))
        for i in range(nb_new):
            new_detection = [np.array(new_scores[i]), np.array(new_boxes[i]), idx]
            for j in range(nb_old):
                m[i,j] = similarity(new_detection, potential_matching_tracklets[j].get_latest_detection(apply_speed=True, new_frame_id=idx))
        return m

    def average_move_speed(self, tracklets):
        """Computes the average displacement of tracklets
        """
        speeds = np.array([[tracklet.speed[0], tracklet.speed[1]] for tracklet in tracklets if tracklet.speed is not None])
        return np.mean(speeds, axis=0)

    def compute_tracks(self):
        """Main function which computes tracks from detection on successive frames
        """
        tracklets = self.build_tracklets(self.list_inference_output, 2, 0.5)
        print("build tracks length tracklets:", len(tracklets))
        average_speed = self.average_move_speed(tracklets)
        filtered_tracklets = tracklets
        # Filter tracklets with wrong speed
        if np.linalg.norm(average_speed) > 0.05:
            filtered_tracklets = list(filter(lambda t:t.has_valid_speed(average_speed), filtered_tracklets))
        # Match tracklets
        matched_tracks = self.match_tracklets(filtered_tracklets, average_speed)
        # Filter tracklets that are too small
        matched_tracks = list(filter(lambda t:t.is_valid(2), matched_tracks))
        return matched_tracks

    def build_tracklets(self, input_detections, time_window = 2, matching_threshold = 0.5):
        """Builds tracklets, i.e. confident matching between successive frames
        using a greedy algorithm (considers the best matching previous box)

        Arguments:

        - input_detections: list of successive frames and their corresponding boxes and classes
        - time_window: integer corresponding to the number of previous frames considered
        - matching_threshold: float threshold for accepting a match

        Returns:

        - a list of Track objects, mainly defined by [frame_ids, boxes and classes]

        """
        tracklets = []

        for frame_idx, json_object in enumerate(input_detections):
            new_scores = json_object.get("output/scores:0", [])
            new_boxes = json_object.get("output/boxes:0", [])
            # Expects boxes with Non maximum suppression ?

            # Build the list of previous trash that could be matched and similarity with the new
            potential_matching_tracklets = list(filter(lambda t:t.is_in_range(frame_idx, time_window), tracklets))
            sim_matrix = self.build_similarity_matrix(new_scores, new_boxes, frame_idx, potential_matching_tracklets)

            # greedily match the new boxes:
            new_tracklets_idxs = list(range(len(new_boxes)))
            if sim_matrix is not None:
                for i in range(len(new_boxes)):
                    max_values = np.max(sim_matrix, axis=1)
                    if np.max(max_values) < matching_threshold:
                        break
                    new_idx = np.argmax(max_values)
                    matching_idx = np.argmax(sim_matrix[new_idx])

                    # append to the corresponding tracklet
                    potential_matching_tracklets[matching_idx].add_matching_detection(new_scores[new_idx], new_boxes[new_idx], frame_idx)

                    # remove from similarity matrix
                    sim_matrix[new_idx,:] = -1.0
                    sim_matrix[:,matching_idx] = -1.0
                    new_tracklets_idxs.remove(new_idx)

            #remaining boxes become new tracklets
            for i in new_tracklets_idxs:
                tracklets.append(Track(len(tracklets), new_scores[i], new_boxes[i], frame_idx))

        return tracklets

    def match_tracklets(self, tracklets, average_speed, matching_threshold=0.5):
        """Match tracklets

        """
        # greedily match the tracklets
        # may be improved using :
        # scipy.optimize.linear_sum_assignment(matrix)
        sim_matrix = self.build_tracklet_similarity_matrix(tracklets)
        matches = []
        nb_tracklets = len(tracklets)
        list_outbound = list(range(nb_tracklets))
        list_inbound = list(range(nb_tracklets))
        if nb_tracklets == 0:
            return []

        if sim_matrix is not None:
            for i in range(nb_tracklets):
                max_values = np.max(sim_matrix, axis=1)
                if np.max(max_values) < matching_threshold:
                    break
                new_idx = np.argmax(max_values)
                matching_idx = np.argmax(sim_matrix[new_idx])

                # remove from similarity matrix
                sim_matrix[new_idx,:] = -1.0
                sim_matrix[:,matching_idx] = -1.0
                matches.append([new_idx, matching_idx])


        for match in matches[::-1]:
            tracklets[match[1]].append_tracklet(tracklets[match[0]])

        tracklet_idxes_to_remove = [m[0] for m in matches]
        return [tracklet for i,tracklet in enumerate(tracklets) if i not in tracklet_idxes_to_remove]


    def json_result(self, tracks, include_geo=False):
        '''Outputs a json result centered on tracked objects. Score not yet included

        Arguments:

        - include_geo: Boolean which specifies whether the return format includes
        the geolocalization data, or just the simple timestamp data

        Returns:

        - a json file of the following format:
        ```python
            {"video_length": 132,
            "fps": 2,
            "video_id": "GOPRO1234.mp4",
            "detected_trash": [
              {"label": "bottle", "id": 0, "frames": [23,24,25]},
              {"label": "fragment", "id": 1, "frames": [32]},
            ]}
        ```
        '''

        json_output = {}
        json_output["video_length"] = self.num_images
        json_output["fps"] = self.fps
        json_output["video_id"] = self.video_id
        json_output["detected_trash"] = [track.json_result() for track in tracks]
        return json_output
