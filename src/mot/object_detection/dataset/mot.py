import json
import os

import numpy as np

from tensorpack.utils import logger
from mot.object_detection.utils.np_box_ops import area as np_area
from mot.object_detection.config import finalize_configs
from mot.object_detection.dataset import DatasetRegistry, DatasetSplit

__all__ = ["register_mot"]


class MotDataset(DatasetSplit):

    def __init__(
            self,
            base_dir,
            split,
    ):
        assert split in ["train", "val"]
        self.split = split
        self.ratio = 0.99
        self.base_dir = base_dir
        self.json_dataset = os.path.join(base_dir, "new_dataset.json")
        if not os.path.isfile(self.json_dataset):
            raise FileNotFoundError("Missing dataset {}".format(self.json_dataset))
        with open(os.path.join(base_dir, 'classes.json')) as f:
            classes = json.load(f)
        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}

    def read_labels(self, labels):
        boxes = []
        classes = []
        for crop in labels:
            boxes.append([int(coord) for coord in crop["bbox"]])
            classes.append(self.class_to_idx[crop["label"]])
        return np.array(boxes), np.array(classes)

    def read_lines(self):
        with open(self.json_dataset, "r") as f:
            lines = [json.loads(line) for line in f]
        if self.split == "train":
            return lines[:int(len(lines) * self.ratio)]
        else:
            return lines[int(len(lines) * self.ratio):]

    def read_file_name(self, md5):
        return os.path.join(self.base_dir, "Images_md5", md5)

    def training_roidbs(self):
        lines = self.read_lines()
        roidbs = []
        for line in lines:
            file_name = self.read_file_name(line["md5"])
            if os.path.isfile(file_name):
                boxes, classes = self.read_labels(line["labels"])
                # Remove boxes with empty area
                if boxes.size:
                    non_zero_area = np_area(boxes) > 0
                    boxes = boxes[non_zero_area, :]
                    classes = classes[non_zero_area]
                boxes = np.float32(boxes)
                roidb = {
                    "file_name": file_name,
                    "boxes": boxes,
                    "class": classes,
                    "is_crowd": np.zeros((classes.shape[0]))
                }
                roidbs.append(roidb)
        return roidbs

    def inference_roidbs(self):
        lines = self.read_lines()
        roidbs = []
        for line in lines:
            roidb = {"file_name": self.read_file_name(line["md5"]), "image_id": line["md5"]}
            roidbs.append(roidb)
        return roidbs

    def eval_inference_results(self, results, output=None):
        id_to_entity_id = {v: k for k, v in self.class_to_idx.items()}
        for res in results:
            res["category_id"] = id_to_entity_id[res["category_id"]]

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        # TODO : implement metrics calculation to return them
        return results


def register_mot(basedir):
    basedir = os.path.expanduser(basedir)
    classes_path = os.path.join(basedir, "classes.json")
    class_names = ["BG"] + get_class_names(classes_path)
    for split in ["train", "val"]:
        name = "mot_" + split
        DatasetRegistry.register(name, lambda x=split: MotDataset(basedir, split=x))
        DatasetRegistry.register_metadata(name, "class_names", class_names)


def get_class_names(classes_path):
    """
    :param classes_path: The path to a file containing class names, for example ["bottles", "others", "fragments"]
    :return: A list: ["bottles", "others", "fragments"]
    """
    if not os.path.isfile(classes_path):
        raise FileNotFoundError("Missing file for classes at {}".format(classes_path))
    with open(classes_path, "r") as f:
        lines = f.readlines()
    class_names = json.loads(lines[0])
    return class_names
