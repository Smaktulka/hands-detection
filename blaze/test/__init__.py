import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch.utils.data

import blaze.train
from blaze.model import BlazeModel
from blaze.model.detector import BlazeDetector

from tqdm import tqdm

from blaze.test.dataset import TestDataset, RgbImageShape
from blaze.test.predictor import Predictor, BlazePredictor


@dataclass
class ConfusionMatrix:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def update_with(self, confusion_matrix):
        self.true_positive += confusion_matrix.true_positive
        self.true_negative += confusion_matrix.true_negative
        self.false_positive += confusion_matrix.false_positive
        self.false_negative += confusion_matrix.false_negative


@dataclass
class TestMetrics:
    accuracy: float
    precisions_for_classes: List[float]
    recalls_for_classes: List[float]
    f1_for_classes: List[float]


class ModelTester:
    def __init__(self, predictor: Predictor, classes_num: int):
        self.confidence_thresh = None
        self.iou_threshold = None
        self.predictor = predictor
        self.confusion_matrices = [ConfusionMatrix(0, 0, 0, 0)] * classes_num
        self.classes_num = classes_num
        self.class_ids = list(range(classes_num))
        self.ious_amount = 0

    def test(self, test_dataset: TestDataset, iou_threshold):
        self.iou_threshold = iou_threshold

        for image, target in tqdm(test_dataset):
            detections = self.predictor.predict_on_image(image)
            self.compare_and_update_matrices(detections, target)

        print(self.confusion_matrices)
        return self.compute_metrics()

    def compare_and_update_matrices(self, model_detections, target):
        if len(model_detections) == 0 and len(target) == 0:
            confusion_matrix = ConfusionMatrix(0, 0, 0, 0)
            confusion_matrix.true_negative += 1
            [matrix.update_with(confusion_matrix) for matrix in self.confusion_matrices]
            return

        true_class_ids = target[:, -1]
        true_boxes = target[:, :4]
        matched_true_boxes = set()

        for detection in model_detections:
            pred_box = detection[1:5]
            pred_class_id = int(detection[-1])
            match = False

            for true_box, true_class_id in zip(true_boxes, true_class_ids):
                iou = self.calculate_iou(pred_box, true_box)
                if iou > self.iou_threshold:
                    self.ious_amount += 1
                    if true_class_id == pred_class_id:
                        match = True
                        self.confusion_matrices[pred_class_id].true_positive += 1
                        matched_true_boxes.add(true_box)
                    else:
                        print(f"true class_id {true_class_id} - pred_class_id {pred_class_id}")
                        self.confusion_matrices[pred_class_id].false_positive += 1

            if not match:
                self.confusion_matrices[pred_class_id].false_positive += 1

    @staticmethod
    def calculate_iou(pred_box, true_box):
        x1 = max(pred_box[0], true_box[0])
        y1 = max(pred_box[1], true_box[1])
        x2 = min(pred_box[2], true_box[2])
        y2 = min(pred_box[3], true_box[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        area_true = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])

        union = area_pred + area_true - intersection
        return intersection / union if union > 0 else 0

    @staticmethod
    def calculate_accuracy(true_detections, total_detections):
        return true_detections / total_detections if total_detections > 0 else 0

    @staticmethod
    def calculate_precision(true_positive, false_positive):
        positives = true_positive + false_positive
        return true_positive / positives if positives > 0 else 0

    @staticmethod
    def calculate_recall(true_positive, false_negative):
        tf_pn = true_positive + false_negative
        return true_positive / tf_pn if tf_pn > 0 else 0

    @staticmethod
    def calculate_f1(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def calculate_mean_ap(average_precisions):
        return np.mean(average_precisions)

    def compute_metrics(self):
        total_tp = sum(cm.true_positive for cm in self.confusion_matrices)
        total_tn = sum(cm.true_negative for cm in self.confusion_matrices)
        total_fp = sum(cm.false_positive for cm in self.confusion_matrices)
        total_fn = sum(cm.false_negative for cm in self.confusion_matrices)
        total_detections = total_tp + total_tn + total_fp + total_fn

        accuracy = self.calculate_accuracy(total_tp + total_tn, total_detections)

        precisions_for_classes = list(range(self.classes_num))
        recalls_for_classes = list(range(self.classes_num))
        f1_for_classes = list(range(self.classes_num))
        for class_id in self.class_ids:
            cm = self.confusion_matrices[class_id]
            precisions_for_classes[class_id] = self.calculate_precision(cm.true_positive, cm.false_positive)
            recalls_for_classes[class_id] = self.calculate_recall(cm.true_positive, cm.false_negative)
            f1_for_classes[class_id] = self.calculate_f1(precisions_for_classes[class_id],
                                                         recalls_for_classes[class_id])

        return TestMetrics(accuracy, precisions_for_classes, recalls_for_classes, f1_for_classes)


if __name__ == '__main__':
    start_path = "d:\\4 course\\DIPLOMA\\like\\hagrid_yolo_format\\"
    images_path = start_path + "test"
    labels_path = start_path + "test_labels"

    resize_shape = RgbImageShape(width=128, height=128)
    test_dataset = TestDataset(images_path, labels_path, resize_shape)

    anchors_path = blaze.ANCHORS_PATH
    anchors = torch.tensor(np.load(anchors_path), dtype=torch.float32, device=blaze.DEVICE)

    model = BlazeModel()
    model_file_name = "blazeface3 (from pc13).pt"
    model_path = os.path.join(blaze.WEIGHTS_PATH, model_file_name)
    model.load_state_dict(torch.load(model_path, map_location=blaze.DEVICE))

    detector = BlazeDetector(anchors)
    predictor = BlazePredictor(model, detector)
    model_tester = ModelTester(predictor, len(test_dataset.class_ids))

    metrics = model_tester.test(test_dataset, 0.0)

    print(metrics)
    print(model_tester.ious_amount)
