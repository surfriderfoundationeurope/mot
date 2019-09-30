import json

from mot.object_detection.dataset.coco import COCODetection


def test_coco(tmpdir):
    annotation_file = tmpdir.mkdir("annotations").join("instances_train.json")
    annotation_content = {"info": "test"}
    annotation_file.write(json.dumps(annotation_content))
    tmpdir.mkdir("train")
    dataset = COCODetection(tmpdir, "train")
