from abc import ABC, abstractmethod

import torch
from blaze.model.detector import SsdDetector


class Predictor(ABC):
    @abstractmethod
    def predict_on_image(self, image):
        pass


class BlazePredictor(Predictor):
    def __init__(self, model: torch.nn.Module, detector: SsdDetector):
        self.model = model
        self.detector = detector

    def predict_on_image(self, image):
        image_tensor = image.unsqueeze(0)

        with torch.no_grad():
            raw_preds = self.model(image_tensor)

        detections = self.detector(raw_preds)
        condition = detections[0, :, 0] > self.detector.confidence_threshold
        return detections[0, condition, :]
