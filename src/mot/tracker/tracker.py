import copy
from typing import List

import numpy as np
from cached_property import cached_property

from mot.object_detection.utils import np_box_ops


class Trash():
    '''Detected trash class
    '''

    def __init__(self, id: int, label: int, box: List[float], frame: int):
        self.id = id
        self.label = label
        self.boxes = [box]
        self.frames = [frame]

    def add_matching_object(self, box, frame):
        self.boxes.append(box)
        self.frames.append(frame)

    def get_np_box(self):
        return np.array(self.boxes[-1])

    def find_best_match_in_list(self, list_trash, iou_threshold):
        '''Finds the best matching trash index with regards to a list of trash
        matching is defined by same class and best iou between boxes

        Arguments:
        - list_trash: A list of reference trash
        - iou_threshold: A float IoU threshold

        Returns:
        - None, or the index of the matching trash
        '''
        matching_id = None
        trash_box = self.get_np_box()[np.newaxis, :]
        boxes_list = []
        idx_list = []
        for matching_trash in list_trash:
            if self.label == matching_trash.label:
                boxes_list.append(matching_trash.get_np_box())
                idx_list.append(matching_trash.id)

        if boxes_list:
            ious = np_box_ops.iou(trash_box, np.vstack(boxes_list))
            best_iou_idx = np.argmax(ious, axis=-1)[0]
            assert ious.shape == (1, len(boxes_list))
            if ious[0, best_iou_idx] >= iou_threshold:
                matching_id = idx_list[best_iou_idx]

        return matching_id

    def get_center(self):
        x1, y1, x2, y2 = self.boxes[-1]
        return (x2 + x1) / 2, (y2 + y1) / 2

    def __repr__(self):
        return "(id:{}, label:{}, center:({:.1f},{:.1f}), frames:{})".format(
            self.id, self.label,
            self.get_center()[0],
            self.get_center()[1], self.frames
        )

    def json_result(self, class_names=["bottles", "others", "fragments"]):
        class_names = ["BG"] + class_names
        rounded_boxes = [[round(coord, 2) for coord in box] for box in self.boxes]
        return {
            "label": class_names[self.label],
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

    def potential_matching_trash_list(self, frame_idx, detected_trashes, objects_per_frame_list):
        '''Creates a list of trash which appear in the last `self.rewind_window_match` frames

        Arguments:

        - frame_idx: the frame index as integer
        - objects_per_frame_list: the list of trash object for each past frame_idx

        Returns:

        - A list of trash objects that could potentially match
        '''
        potential_matching_ids = set()
        for idx_rewind in range(frame_idx - 1, frame_idx - self.rewind_window_match - 1, -1):
            if idx_rewind >= 0:
                potential_matching_ids = potential_matching_ids.union(
                    set(objects_per_frame_list[idx_rewind])
                )
        return [trash for trash in detected_trashes if trash.id in potential_matching_ids]

    @cached_property
    def detected_trash(self):
        return self.track_objects()

    def track_objects(self):
        '''Main function which tracks trash objects. Assumes

        Returns:

        - The detected trash list
        '''
        detected_trashes = []

        if not self.list_inference_output:
            raise ValueError("No inference was run, can't track objects")

        nb_detected_objects = 0
        objects_per_frame_list = [[] for f in self.list_inference_output]

        for frame_idx, json_object in enumerate(self.list_inference_output):
            found_classes = json_object.get("output/labels:0")
            found_boxes = json_object.get("output/boxes:0")

            # Build the list of previous trash that could be matched
            potential_matching_trash = self.potential_matching_trash_list(
                frame_idx, detected_trashes, objects_per_frame_list
            )
            for label, box in zip(found_classes, found_boxes):
                current_trash = Trash(nb_detected_objects, label, box, frame_idx)

                # Is this object already matching something?
                matching_id = current_trash.find_best_match_in_list(
                    potential_matching_trash, self.iou_threshold
                )
                if matching_id is not None:
                    # append the frame & box to the matching trash
                    detected_trashes[matching_id].add_matching_object(box, frame_idx)
                    objects_per_frame_list[frame_idx].append(matching_id)
                else:
                    # Otherwise, create new object
                    detected_trashes.append(copy.deepcopy(current_trash))
                    objects_per_frame_list[frame_idx].append(current_trash.id)
                    nb_detected_objects += 1

        return detected_trashes

    def json_result(self, include_geo=False):
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
        json_output["detected_trash"] = [trash.json_result() for trash in self.detected_trash]
        return json_output
