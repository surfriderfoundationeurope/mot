import json
import os
from typing import Dict, List

import numpy as np

from mot.object_detection.dataset import DatasetRegistry, DatasetSplit
from mot.object_detection.utils.np_box_ops import area as np_area


class MotDataset(DatasetSplit):
    """For more details on each methods see the description of dataset.py

    Arguments:

    - *base_dir*: the directory where the dataset is located. There should be inside a JSON named {dataset_name}.
    - *split*: "train" or "val"
    - *dataset_file*: the name of the JSON to use as dataset in base_dir. A line of this JSON should look like:

    ```json
    {
        "md5": "159af544cd304c3ee50f63c281afa032",
        "labels":
            [
                {
                    "bbox": ["250", "255", "278", "289"],
                    "label": "fragments"
                }, {
                    "bbox": ["432", "231", "459", "280"],
                    "label": "fragments"
                }
            ],
    }
    ```

    - *classes_file*: the name of the JSON to use as classes in base_dir. A line of this JSON should look like:

    ```json
    ["bottles", "others", "fragments"]
    ```

    - *images_folder*: Where the images are stored, and named by their md5.
        - if it starts with "/", the absolute path will be used
        - if not, the path os.path.join(base_dir, images_folder) will be used

    Raises:

    - *FileNotFoundError*: if the dataset or classes are missing.
    - *NotADirectoryError*: if the images folder is missing.
    """

    def __init__(
        self,
        base_dir: str,
        split: str,
        dataset_file: str = "dataset.json",
        classes_file: str = "classes.json",
        images_folder: str = "Images_md5"
    ):
        assert split in ["train", "val"]
        self.split = split
        self.ratio = 0.99
        self.base_dir = base_dir
        self.dataset_path = os.path.join(base_dir, dataset_file)
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(
                "Missing file {}. Make sure your file is named correctly.".format(self.dataset_path)
            )
        self.classes_path = os.path.join(base_dir, classes_file)
        if not os.path.isfile(self.classes_path):
            raise FileNotFoundError(
                "Missing file {}. Make sure your file is named correctly.".format(self.classes_path)
            )
        with open(self.classes_path) as f:
            classes = json.load(f)
        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}
        # yapf: disable
        if images_folder[0] == "/": # absolute path if the images are not stored inside the dataset folder for example
            self.images_folder = images_folder
        else: # relative path, then it means that the images are stored inside the dataset folder
            self.images_folder = os.path.join(base_dir, images_folder)
        # yapf: enable
        if not os.path.isdir(self.images_folder):
            raise NotADirectoryError("Missing images folder {}.".format(self.images_folder))

    def read_labels(self, labels: List[Dict[str, object]]) -> (np.ndarray, np.ndarray):
        """[summary]

        Arguments:

        - *labels*: A list of dicts such as:

        ```python3
        labels = [
            {
                "bbox": ["250", "255", "278", "289"],
                "label": "fragments"
            }, {
                "bbox": ["432", "231", "459", "280"],
                "label": "fragments"
            }
        ]
        ```

        Returns:

        - *np.ndarray, np.ndarray*: The boxes and classes. Classes starts at 1 since 0 is for background.
        """
        boxes = []
        classes = []
        for crop in labels:
            boxes.append([int(coord) for coord in crop["bbox"]])
            classes.append(self.class_to_idx[crop["label"]])
        return np.array(boxes), np.array(classes)

    def read_lines(self) -> List[Dict[str, object]]:
        """Reads the int(len(lines) * self.ratio) first lines if training, and the last ones otherwise.

        Arguments:

        Returns:

        - *List[Dict[str, object]]*: Each dict should look like:

        ```python
        {
            "md5": "159af544cd304c3ee50f63c281afa032",
            "labels":
                [
                    {
                        "bbox": ["250", "255", "278", "289"],
                        "label": "fragments"
                    }, {
                        "bbox": ["432", "231", "459", "280"],
                        "label": "fragments"
                    }
                ],
        }
        ```
        """
        with open(self.dataset_path, "r") as f:
            lines = [json.loads(line) for line in f]
        if self.split == "train":
            return lines[:int(len(lines) * self.ratio)]
        else:
            return lines[int(len(lines) * self.ratio):]

    def read_file_name(self, md5: str) -> str:
        return os.path.join(self.images_folder, md5)

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

    def eval_inference_results(self, results: Dict[str, object], output: str = None):
        id_to_entity_id = {v: k for k, v in self.class_to_idx.items()}
        for res in results:
            res["category_id"] = id_to_entity_id[res["category_id"]]

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        # TODO : implement metrics calculation to return them
        return {}


def register_mot(
    base_dir: str,
    dataset_file: str = "dataset.json",
    classes_file: str = "classes.json",
    images_folder: str = "Images_md5"
):
    """Register a dataset to the registry. See `MotDataset` for more details on arguments.
    """
    base_dir = os.path.expanduser(base_dir)
    classes_path = os.path.join(base_dir, "classes.json")
    class_names = ["BG"] + get_class_names(classes_path)
    for split in ["train", "val"]:
        name = "mot_" + split
        DatasetRegistry.register(name, lambda x=split: MotDataset(base_dir, split=x))
        DatasetRegistry.register_metadata(name, "class_names", class_names)


def get_class_names(classes_path: str) -> List[str]:
    """
    Arguments:

    - *classes_path*: The path to a file containing class names, for example ["bottles", "others", "fragments"]

    Returns:

    - *List[str]*: A list: ["bottles", "others", "fragments"]
    """
    if not os.path.isfile(classes_path):
        raise FileNotFoundError("Missing file for classes at {}".format(classes_path))
    with open(classes_path, "r") as f:
        lines = f.readlines()
    class_names = json.loads(lines[0])
    return class_names
