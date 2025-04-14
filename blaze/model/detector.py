import dataclasses
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

import numpy as np
import torchvision

import blaze


@dataclasses.dataclass
class SsdDetector(ABC):
    anchors: np.ndarray
    confidence_threshold: float
    top_k: int
    nms_threshold: float

    @abstractmethod
    def __call__(self, raw_predictions):
        pass


class BlazeDetector(SsdDetector):
    def __call__(self, raw_predictions):
        return self.forward(raw_predictions, self.anchors)

    def __init__(self, anchors, conf_thresh: float = 0.8, top_k: int = 200, nms_thresh: float = 0.6, ) -> None:
        super().__init__(anchors, conf_thresh, top_k, nms_thresh)
        self.softmax = nn.Softmax(dim=-1)
        self.anchors = anchors
        self.confidence_threshold = conf_thresh
        self.top_k = top_k
        self.nms_threshold = nms_thresh
        self.device = blaze.DEVICE

    @staticmethod
    def decode(loc: torch.Tensor, dbox_list: torch.Tensor) -> torch.Tensor:
        boxes = torch.cat((
            dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, :2],
            dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)

        # convert boxes to (xmin, ymin, xmax, ymax)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    def forward(self, raw_preds, anchors) -> torch.Tensor:
        loc_data = raw_preds[:, :, :4]
        conf_data = raw_preds[:, :, 4:]
        dbox_list = anchors
        num_batch = loc_data.shape[0]
        num_classes = conf_data.shape[2]
        conf_data = self.softmax(conf_data)

        # [batch, topk, 6]: 6 for [confidence, x_min, y_min, x_max, y_max, class]
        output = torch.zeros(num_batch, self.top_k, 6)

        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            decoded_boxes = self.decode(loc_data[i], dbox_list.to(blaze.DEVICE))
            conf_scores = conf_preds[i].clone()
            total_dets = 0
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.confidence_threshold)

                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids = torchvision.ops.nms(boxes, scores, self.nms_threshold)
                class_ids_tensor = torch.tensor((cl - 1) * np.ones((len(ids), 1))).to(blaze.DEVICE)
                if total_dets + len(ids) < self.top_k:
                    output[i, total_dets:total_dets + len(ids)] = torch.cat(
                        (scores[ids].unsqueeze(1), boxes[ids], class_ids_tensor), 1)
                    total_dets += len(ids)

        return output


# class BlazeDetector(SsdDetector):
#     def __call__(self, raw_predictions):
#         return self.forward(raw_predictions, self.anchors)
#
#     def __init__(self, anchors, conf_thresh: float = 0.5, top_k: int = 200, nms_thresh: float = 0.6, ) -> None:
#         """Initialize Detect.
#
#         Args:
#             conf_thresh (float, optional): Confidence threshold. Defaults to 0.01.
#             top_k (int, optional): Top K results to keep. Defaults to 200.
#             nms_thresh (float, optional): Non-maximum suppression threshold. Defaults to 0.45.
#         """
#         # super(SsdDetector, self).__init__(anchors, conf_thresh, top_k, nms_thresh)
#         self.softmax = nn.Softmax(dim=-1)
#         self.anchors = anchors
#         self.confidence_threshold = conf_thresh
#         self.top_k = top_k
#         self.nms_threshold = nms_thresh
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     @staticmethod
#     def decode(loc: torch.Tensor, dbox_list: torch.Tensor) -> torch.Tensor:
#         """Decode bounding boxes from predictions using default boxes.
#
#         Args:
#             loc (torch.Tensor): Location predictions.
#             dbox_list (torch.Tensor): Default box list.
#
#         Returns:
#             torch.Tensor: Decoded bounding boxes.
#         """
#         boxes = torch.cat((
#             dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, :2],
#             dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
#
#         # convert boxes to (xmin, ymin, xmax, ymax)
#         boxes[:, :2] -= boxes[:, 2:] / 2
#         boxes[:, 2:] += boxes[:, :2]
#
#         return boxes
#
#     def forward(self, raw_preds, anchors) -> int | torch.Tensor:
#         """Forward pass for Detect.
#
#         Args:
#             args (tuple): Tuple containing location data, confidence data, and default box list.
#
#         Returns:
#             torch.Tensor: Detection results.
#         """
#         # (loc_data, conf_data, dbox_list) = args
#         loc_data = raw_preds[:, :, :4]
#         conf_data = raw_preds[:, :, 4:]
#         dbox_list = anchors
#         num_batch = loc_data.shape[0]
#         num_classes = conf_data.shape[2]
#         # conf_data = torch.from_numpy(conf_data)
#         # loc_data = torch.from_numpy(loc_data)
#         conf_data = self.softmax(conf_data)
#
#         # [batch, topk, 6]: 6 for [confidence, x_min, y_min, x_max, y_max, class]
#         output = torch.zeros(num_batch, self.top_k, 6)
#
#         conf_preds = conf_data.transpose(2, 1)
#
#         for i in range(num_batch):
#             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#             decoded_boxes = self.decode(loc_data[i], dbox_list.to(device))
#             #
#             conf_scores = conf_preds[i].clone()
#             total_dets = 0
#             class_for_ret = num_classes - 1
#             for cl in range(1, num_classes):
#                 # cla = conf_scores[cl]
#                 c_mask = conf_scores[cl].gt(self.confidence_threshold)
#
#                 scores = conf_scores[cl][c_mask]
#                 # if not (conf_scores[cl] > 0.5).any():
#                     # return cl - 1
#                     # continue
#                 if scores.nelement() == 0:
#                     # return cl - 1
#                     continue
#                 # return cl - 1
#                 # return cl - 1
#                 l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
#                 #
#                 boxes = decoded_boxes[l_mask].view(-1, 4)
#                 #
#                 ids = torchvision.ops.nms(boxes, scores, self.nms_threshold)
#                 if total_dets + len(ids) < self.top_k:
#                     output[i, total_dets:total_dets + len(ids)] = torch.cat((scores[ids].unsqueeze(1), boxes[ids],
#                                                                              torch.tensor(
#                                                                                  (cl - 1) * np.ones((len(ids), 1))).to(
#                                                                                  device)), 1)
#                     total_dets += len(ids)
#
#         return output
