import numpy as np
from mot.object_detection.utils import np_box_ops
import copy

class Trash():
    '''Detected trash class
    '''
    def __init__(self, id, label, box, frame):
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
            mathcing is defined by same class and best iou between boxes
        '''
        matching_id = None
        trash_box = self.get_np_box()[np.newaxis,:]
        boxes_list = []
        idx_list = []
        for matching_trash in list_trash:
            if self.label == matching_trash.label:
                boxes_list.append(matching_trash.get_np_box())
                idx_list.append(matching_trash.id)

        if boxes_list:
            ious = np_box_ops.iou(trash_box, np.vstack(boxes_list))
            best_iou_idx = np.argmax(ious,axis=-1)[0]
            assert ious.shape == (1, len(boxes_list))
            if ious[0,best_iou_idx] >= iou_threshold:
                matching_id = idx_list[best_iou_idx]

        return matching_id

    def get_center(self):
        x1,y1,x2,y2 = self.boxes[-1]
        return (x2+x1)/2, (y2+y1)/2

    def __repr__(self):
        return "(id:{}, label:{}, center:({:.1f},{:.1f}), frames:{})".format(self.id, self.label, self.get_center()[0], self.get_center()[1], self.frames)


class ObjectTracking():
    '''Wrapper class to tracking trash objects in video output frames

    '''
    def __init__(self, video_id, list_path_images, list_inference_output = None, fps = 2, list_geoloc = None):
        self.video_id = video_id

        self.list_path_images = list_path_images
        self.list_inference_output = list_inference_output

        self.num_images = len(list_path_images)
        self.list_geoloc = list_geoloc
        self.fps = fps

        self.detected_trash = []
        self.tracking_done = False

        self.iou_threshold = 0.3
        self.rewind_window_match = 2

    def potential_matching_trash_list(self, frame_idx, objects_per_frame_list):
        '''Creates a list of trash which appear in the last `self.rewind_window_match` frames
        '''
        potential_matching_ids = set()
        for idx_rewind in range(frame_idx -1, frame_idx - self.rewind_window_match - 1, -1):
            if idx_rewind >= 0:
                potential_matching_ids = potential_matching_ids.union(set(objects_per_frame_list[idx_rewind]))
        return [trash for trash in self.detected_trash if trash.id in potential_matching_ids]

    def track_objects(self):
        '''Main function which tracks trash objects
        '''
        if not self.list_inference_output:
            raise ValueError("No inference was run, can't track objects")

        nb_detected_objects = 0
        objects_per_frame_list = [[] for f in self.list_inference_output]

        for frame_idx, json_object in enumerate(self.list_inference_output):
            found_classes = json_object.get("output/labels:0")
            found_boxes = json_object.get("output/boxes:0")

            # Build the list of previous trash that could be matched
            potential_matching_trash = self.potential_matching_trash_list(frame_idx, objects_per_frame_list)
            for label, box in zip(found_classes, found_boxes):
                current_trash = Trash(nb_detected_objects, label, box, frame_idx)

                # Is this object already matching something?
                matching_id = current_trash.find_best_match_in_list(potential_matching_trash, self.iou_threshold)
                if matching_id is not None:
                    # append the frame & box to the matching trash
                    self.detected_trash[matching_id].add_matching_object(box, frame_idx)
                    objects_per_frame_list[frame_idx].append(matching_id)
                else:
                    # Otherwise, create new object
                    self.detected_trash.append(copy.deepcopy(current_trash))
                    objects_per_frame_list[frame_idx].append(current_trash.id)
                    nb_detected_objects += 1

        self.tracking_done = True
        return self.detected_trash


    def json_result(self, include_geo = False):
        '''Outputs a json result centered on tracked objects

        Arguments:

        - include_geo: Boolean which specifies whether the return format includes
        the geolocalization data, or just the simple timestamp data

        Returns:

        - a json file of the following format:
        ```python
            {"video_length": 132,
            "video_id": "GOPRO1234.mp4",
            "detected_trash": [
              {"type": "bottle", "score": 0.97, "ts": [23,24,25]},
              {"type": "fragment", "score": 0.93, "ts": [32]},
            ],
            "model_version": "0.1"}
        ```
        '''
        if not self.tracking_done:
            self.track_objects()

        json_output = {}
        json_output["video_length"] = self.num_images / fps
        json_output["video_id"] = self.video_id
        return json_output
