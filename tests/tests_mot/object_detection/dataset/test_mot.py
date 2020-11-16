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
        } for _ in range(2)
    ]
    with open(os.path.join(tmpdir, "dataset.json"), "w+") as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + "\n")

    with open(os.path.join(tmpdir, "classes.json"), "w+") as f:
        f.write("[\"bottles\", \"others\", \"fragments\"]")


def test_training_roidbs(tmpdir):
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


def test_inference_roidbs(tmpdir):
    prepare_dataset_folder(tmpdir)
    for split in "train", "val":
        dataset = MotDataset(base_dir=tmpdir, split=split)
        roidbs = dataset.inference_roidbs()
        assert len(roidbs) == 1
        assert os.path.isfile(roidbs[0]["file_name"])
        assert "boxes" not in roidbs[0]
        assert "class" not in roidbs[0]


def test_eval_inference_results(tmpdir):
    prepare_dataset_folder(tmpdir)
    for split in "train", "val":
        dataset = MotDataset(base_dir=tmpdir, split=split)
        results = [
            {
                'image_id': "abcdef.png",
                'category_id': 3,
                'bbox': [255, 265, 270, 300],
                'score': 0.9,
            }, {
                'image_id': "abcdef.png",
                'category_id': 1,
                'bbox': [255, 265, 270, 300],
                'score': 0.9,
            }
        ]
        output_file = os.path.join(tmpdir, "output.json")
        dataset.eval_inference_results(results, output_file)
        assert os.path.isfile(output_file)
        assert len(results) == 2
        with open(output_file, "r") as f:
            assert json.load(f) == [
                {
                    'image_id': "abcdef.png",
                    'category_id': "fragments",
                    'bbox': [255, 265, 270, 300],
                    'score': 0.9,
                }, {
                    'image_id': "abcdef.png",
                    'category_id': "bottles",
                    'bbox': [255, 265, 270, 300],
                    'score': 0.9,
                }
            ]
