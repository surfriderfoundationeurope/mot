import json
import os
import pathlib
from collections import OrderedDict

import cv2
import numpy as np

from mot.object_detection.dataset.mot import MotDataset


def prepare_dataset_folder(tmpdir):
    images_folder = os.path.join(tmpdir, "Images_md5")
    pathlib.Path(images_folder).mkdir(parents=True, exist_ok=True)
    image_md5 = "abcdef.png"
    image_path = os.path.join(images_folder, image_md5)
    image_array = np.ones((600, 600, 3))
    cv2.imwrite(image_path, image_array)
    # We write two annotations, because one will be used for the train split and one for the val split
    annotations = [
        {
            'md5':
                image_md5,
            'size':
                OrderedDict([('width', '600'), ('height', '600'), ('depth', '3')]),
            'labels':
                [
                    {
                        'bbox': ['250', '255', '278', '289'],
                        'label': 'fragments'
                    }, {
                        'bbox': ['432', '231', '459', '280'],
                        'label': 'bottles'
                    }
                ],
        }, {
            'md5':
                image_md5,
            'size':
                OrderedDict([('width', '600'), ('height', '600'), ('depth', '3')]),
            'labels':
                [
                    {
                        'bbox': ['250', '255', '278', '289'],
                        'label': 'fragments'
                    }, {
                        'bbox': ['432', '231', '459', '280'],
                        'label': 'bottles'
                    }
                ],
        }
    ]
    with open(os.path.join(tmpdir, "new_dataset.json"), "w+") as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + "\n")

    with open(os.path.join(tmpdir, "classes.json"), "w+") as f:
        f.write("[\"bottles\", \"others\", \"fragments\"]")


def test_mot(tmpdir):
    prepare_dataset_folder(tmpdir)
    for split in "train", "val":
        dataset = MotDataset(base_dir=tmpdir, split=split)
        roidbs = dataset.training_roidbs()
        assert len(roidbs) == 1
        assert os.path.isfile(roidbs[0]["file_name"])
        assert np.array_equal(
            roidbs[0]["boxes"], np.array([[250., 255., 278., 289.], [432., 231., 459., 280.]])
        )
        assert np.array_equal(roidbs[0]["class"], np.array([3, 1], np.float32))
