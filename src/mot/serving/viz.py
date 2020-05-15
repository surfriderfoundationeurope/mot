from typing import Dict, List

import cv2

from mot.serving.constants import CLASS_NAME_TO_COLOR


def draw_boxes(image_path: str, trashes: List[Dict]):
    img = cv2.imread(image_path)  # img.shape = [height, width, channels]

    for trash in trashes:
        box = trash["box"]
        box[0] = int(box[0] * img.shape[0])  # y1
        box[1] = int(box[1] * img.shape[1])  # x1
        box[2] = int(box[2] * img.shape[0])  # y2
        box[3] = int(box[3] * img.shape[1])  # x2
        img = cv2.rectangle(
            img,
            (box[0], box[1]),
            (box[2], box[3]),
            CLASS_NAME_TO_COLOR[trash["label"]],
            3,
        )
        img = cv2.putText(
            img,
            trash["label"] + " " + str(round(trash["score"], 2)),
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            CLASS_NAME_TO_COLOR[trash["label"]],
            cv2.LINE_4,
        )

    cv2.imwrite(image_path, img)
