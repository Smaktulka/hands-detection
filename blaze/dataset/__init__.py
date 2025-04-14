import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torchvision import transforms


class BlazeDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path: str, image_size: int, augment: A.Compose = None):
        self.labels = list(sorted(glob(f'{labels_path}/*')))
        self.labels = [x for x in self.labels if os.stat(x).st_size != 0]
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((image_size, image_size))
        ])
        self.image_size = image_size

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.labels[idx].replace('labels', 'images')[:-3] + 'jpg'
        img = self.load_image(img_path)
        rescale_output = self.resize_and_pad(img, self.image_size)
        img = rescale_output['image']
        target = self.read_and_convert_labels(self.labels[idx], rescale_output)
        if self.augment is not None:
            augmented = self.augment(image=img, bboxes=target)
            img = augmented['image']
            target = np.array(augmented['bboxes'])

        img = self.transform(img)
        return self.transform(img.copy()), np.clip(target, 0, 1)

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    @staticmethod
    def read_and_convert_labels(labels_idx: str, rescale_output: dict) -> np.ndarray:
        annotations = pd.read_csv(labels_idx, header=None, sep=' ')
        labels = annotations.values[:, 0]
        x1, x2, y1, y2 = BlazeDataset.convert_yolo_to_simple(annotations, rescale_output)

        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 1)
        y1 = np.expand_dims(y1, 1)
        y2 = np.expand_dims(y2, 1)

        coords = np.concatenate((x1, y1, x2, y2), axis=1).clip(0., 1.)
        target = np.concatenate((coords, labels.reshape(-1, 1)), axis=1)
        return target

    @staticmethod
    def convert_yolo_to_simple(annotations, rescale_output):
        yolo_bboxes = annotations.values[:, 1:]
        cx = yolo_bboxes[:, 0]
        cy = yolo_bboxes[:, 1]
        w = yolo_bboxes[:, 2]
        h = yolo_bboxes[:, 3]

        x1 = (cx - w / 2) * rescale_output['x_ratio'] + rescale_output['x_offset']
        x2 = (cx + w / 2) * rescale_output['x_ratio'] + rescale_output['x_offset']
        y1 = (cy - h / 2) * rescale_output['y_ratio'] + rescale_output['y_offset']
        y2 = (cy + h / 2) * rescale_output['y_ratio'] + rescale_output['y_offset']

        return x1, x2, y1, y2

    @staticmethod
    def resize_and_pad(img: np.ndarray, target_size: int = 128) -> dict:
        if img.shape[0] > img.shape[1]:
            new_y = target_size
            new_x = int(target_size * img.shape[1] / img.shape[0])
        else:
            new_y = int(target_size * img.shape[0] / img.shape[1])
            new_x = target_size
        output_img = cv2.resize(img, (new_x, new_y))
        top = max(0, new_x - new_y) // 2
        bottom = target_size - new_y - top
        left = max(0, new_y - new_x) // 2
        right = target_size - new_x - left
        output_img = cv2.copyMakeBorder(
            output_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )

        x_ratio = new_x / target_size
        y_ratio = new_y / target_size
        x_offset = left / target_size
        y_offset = top / target_size

        return {'image': output_img, 'x_ratio': x_ratio, 'x_offset': x_offset, 'y_ratio': y_ratio, 'y_offset': y_offset}
