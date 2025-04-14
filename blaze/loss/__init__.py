
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh: float = 0.5, neg_pos: int = 3,
                 device: torch.device = torch.device('cpu'), dbox_list: torch.Tensor = None) -> None:
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device
        self.dbox_list = dbox_list

    def forward(self, predictions: torch.Tensor, targets: list) -> tuple:
        loc_data = predictions[:, :, :4]
        conf_data = predictions[:, :, 4:]

        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_t = torch.zeros((num_batch, num_dbox), dtype=torch.int64, device=self.device)
        loc_t = torch.zeros((num_batch, num_dbox, 4), dtype=torch.float32, device=self.device)

        for idx in range(num_batch):
            if len(targets[idx]) > 0:
                truths = targets[idx][:, :-1].to(self.device)  # BBox
                labels = targets[idx][:, -1].to(self.device)
                dbox = self.dbox_list.to(self.device)
                variance = [0.1, 0.2]
                loc_t[idx], conf_t[idx] = self.match_dbox_to_truth(self.jaccard_thresh, truths, dbox,
                                                                   variance, labels)

        pos_mask = conf_t > 0
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        batch_conf = conf_data.view(-1, num_classes)

        loss_c = F.cross_entropy(
            batch_conf, torch.clamp(conf_t.view(-1), 0, num_classes - 1), reduction='none')

        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)

        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)

        conf_t_label_hnm = conf_t[(pos_mask + neg_mask).gt(0)]
        loss_c = F.cross_entropy(conf_hnm, torch.clamp(conf_t_label_hnm, 0, num_classes - 1), reduction='sum')

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

    @staticmethod
    def match_dbox_to_truth(threshold: float, truths: torch.Tensor, dbox: torch.Tensor, variances: list,
                            labels: torch.Tensor) -> tuple[Tensor, Tensor]:
        overlaps = MultiBoxLoss.jaccard(
            truths,
            MultiBoxLoss.point_form(dbox)
        )
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # [1, objects_num]
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # [1, priors_num]

        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)

        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)

        best_truth_overlap.index_fill_(0, best_prior_idx, 2)
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        matches = truths[best_truth_idx]  # Shape: [num_priors, 4]
        conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
        conf[best_truth_overlap < threshold] = 0  # label as background
        loc = MultiBoxLoss.encode(matches, dbox, variances)
        return loc, conf

    @staticmethod
    def encode(matched: torch.Tensor, priors: torch.Tensor, variances: list) -> torch.Tensor:
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        g_cxcy /= (variances[0] * priors[:, 2:])
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        return torch.cat([g_cxcy, g_wh], 1)  # [priors_num, 4]

    @staticmethod
    def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        inter = MultiBoxLoss.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    @staticmethod
    def point_form(boxes: torch.Tensor) -> torch.Tensor:
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                          boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax

    @staticmethod
    def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

