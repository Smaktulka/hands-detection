import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image


@dataclass
class ImageProportions:
    width_ratio: float
    height_ratio: float
    horizontal_offset: float
    vertical_offset: float

    def ratios(self):
        return self.width_ratio, self.height_ratio

    def offsets(self):
        return self.horizontal_offset, self.vertical_offset


class Normalized:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check()

    def check(self):
        for val in self.__dict__.values():
            if not (0.0 <= val <= 1.0):
                raise ValueError("Coordinates are not normalized between 0 and 1")


@dataclass
class SimpleBoundingBox(Normalized):
    x_max: float
    y_max: float
    x_min: float
    y_min: float

    def to_array(self) -> List[float]:
        return [self.x_max, self.y_max, self.x_min, self.y_min]


@dataclass
class ProportionalBoundingBox(SimpleBoundingBox):
    @staticmethod
    def create(simple_bbox: SimpleBoundingBox, proportions: ImageProportions):
        width_ratio, height_ratio = proportions.ratios()
        horizontal_offset, vertical_offset = proportions.offsets()
        return ProportionalBoundingBox(
            x_max=(simple_bbox.x_max * width_ratio + horizontal_offset),
            y_max=(simple_bbox.y_max * height_ratio + vertical_offset),
            x_min=(simple_bbox.x_min * width_ratio + horizontal_offset),
            y_min=(simple_bbox.y_min * height_ratio + vertical_offset)
        )


@dataclass
class YoloBoundingBox(Normalized):
    x_center: float
    y_center: float
    width: float
    height: float


class BoundingBoxConverter:
    @staticmethod
    def simple_to_yolo(simple_bbox: SimpleBoundingBox):
        x_center = (simple_bbox.x_max + simple_bbox.x_min) / 2
        y_center = (simple_bbox.y_max + simple_bbox.y_min) / 2

        width = simple_bbox.x_max - simple_bbox.x_min
        height = simple_bbox.y_max - simple_bbox.y_min

        return YoloBoundingBox(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height
        )

    @staticmethod
    def yolo_to_simple(yolo_bbox: YoloBoundingBox) -> SimpleBoundingBox:
        return SimpleBoundingBox(
            x_max=(yolo_bbox.x_center - yolo_bbox.width / 2),
            y_max=(yolo_bbox.y_center - yolo_bbox.height / 2),
            x_min=(yolo_bbox.x_center + yolo_bbox.width / 2),
            y_min=(yolo_bbox.y_center + yolo_bbox.height / 2),
        )


@dataclass
class YoloAnnotations:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @staticmethod
    def parse(label_str: str):
        parts = label_str.strip().split()
        return YoloAnnotations(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4])
        )

    def bbox(self):
        return YoloBoundingBox(
            self.x_center,
            self.y_center,
            self.width,
            self.height
        )


@dataclass
class RgbImageShape:
    width: int
    height: int

    def __getitem__(self, ind):
        if ind == 0:
            return self.width
        elif ind == 1:
            return self.height

    def __iter__(self):
        return iter((self.width, self.height))

    def to_seq(self):
        return [self.width, self.height]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, resize_shape: RgbImageShape):
        self.images_path = images_path
        self.labels_path = labels_path
        self.resize_shape = resize_shape
        self.label_file_names = []
        self.class_names = []
        self.class_ids = []

        for class_id, class_name in enumerate(os.listdir(labels_path)):
            self.class_names.append(class_name)
            self.class_ids.append(class_id)
            class_dir = os.path.join(labels_path, class_name)
            for label_file_name in os.listdir(class_dir):
                self.label_file_names.append(os.path.join(class_name, label_file_name))

    def __len__(self):
        return len(self.label_file_names)

    def __getitem__(self, idx: int):
        label_file_name = self.label_file_names[idx]
        image = self.load_image_by_label(label_file_name)
        image, proportions = self.prepare_image(image)

        labels = self.prepare_labels(self.label_file_names[idx])
        simple_bboxes = [BoundingBoxConverter.yolo_to_simple(ann.bbox()) for ann in labels]
        proportional_bboxes = [ProportionalBoundingBox.create(bbox, proportions) for bbox in simple_bboxes]

        bboxes = [bbox.to_array() for bbox in proportional_bboxes]
        class_ids = [ann.class_id for ann in labels]

        try:
            target = self.to_target(bboxes, class_ids)
        except RuntimeError as e:
            raise RuntimeError(f"Cannot create target for label {label_file_name}. Error: {e}")

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Resize(self.resize_shape.to_seq())
        ])

        return transform(image), target

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

    def load_image_by_label(self, label_file_name) -> np.ndarray:
        image_file_name = label_file_name[:-3] + 'jpg'
        image_path = os.path.join(self.images_path, image_file_name)
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def prepare_image(self, image: np.ndarray):
        proportional_resize_shape = self.calculate_proportional_resize_shape(image.shape)
        return self.pad(image, proportional_resize_shape)

    def prepare_labels(self, label_file_name) -> list[YoloAnnotations]:
        label_file_path = os.path.join(self.labels_path, label_file_name)
        with open(label_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        labels = self.parse_label_file_content(content)
        return labels

    @staticmethod
    def parse_label_file_content(label_file_content):
        annotations = []
        for label_str in label_file_content.strip().splitlines():
            annotations.append(YoloAnnotations.parse(label_str))

        return annotations

    def calculate_proportional_resize_shape(self, image_size):
        height, width = image_size[:-1]
        if height > width:
            x_size = int(self.resize_shape[0] * width / height)
            y_size = self.resize_shape.height
        else:
            x_size = self.resize_shape.width
            y_size = int(self.resize_shape[1] * height / width)

        return RgbImageShape(width=x_size, height=y_size)

    def pad(self, image, new_shape: RgbImageShape):
        width, height = new_shape
        top = max(0, width - height) // 2
        bottom = self.resize_shape[0] - height - top
        left = max(0, height - width) // 2
        right = self.resize_shape[1] - width - left

        image = cv2.resize(image, new_shape.to_seq())
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

        proportions = ImageProportions(
            width_ratio=(width / self.resize_shape[0]),
            height_ratio=(height / self.resize_shape[1]),
            horizontal_offset=(left / self.resize_shape[0]),
            vertical_offset=(top / self.resize_shape[1])
        )

        return image, proportions

    def to_target(self, bboxes, class_ids):
        bboxes = torch.tensor(bboxes).clip(0.0, 1.0)
        class_ids = torch.tensor(class_ids).unsqueeze(dim=0).reshape(-1, 1)
        valid_classes = torch.all(torch.isin(class_ids, torch.tensor(self.class_ids)))
        if not valid_classes:
            raise RuntimeError("Invalid class id in label")

        target = torch.concatenate((bboxes, class_ids), dim=1)

        return target
