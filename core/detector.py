import cv2
import numpy as np


class PersonDetector:
    """
    Person detector using OpenCV HOG + SVM.
    Works best when full human bodies are visible.
    """

    def __init__(self, config):
        self.config = config

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.win_stride = tuple(config["detector"]["win_stride"])
        self.padding = tuple(config["detector"]["padding"])
        self.scale = config["detector"]["scale"]
        self.nms_threshold = config["detector"]["nms_threshold"]

    def detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
        )

        if len(boxes) == 0:
            return []

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        final_boxes = self.non_max_suppression(boxes, self.nms_threshold)

        final_boxes_xywh = []

        for (x1, y1, x2, y2) in final_boxes:
            final_boxes_xywh.append((x1, y1, x2 - x1, y2 - y1))

        return final_boxes_xywh

    def non_max_suppression(self, boxes, overlap_thresh):
        if len(boxes) == 0:
            return []

        boxes = boxes.astype("float")

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        indexes = np.argsort(y2)

        picked = []

        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]
            picked.append(i)

            xx1 = np.maximum(x1[i], x1[indexes[:last]])
            yy1 = np.maximum(y1[i], y1[indexes[:last]])
            xx2 = np.minimum(x2[i], x2[indexes[:last]])
            yy2 = np.minimum(y2[i], y2[indexes[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[indexes[:last]]

            indexes = np.delete(
                indexes,
                np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
            )

        return boxes[picked].astype("int")