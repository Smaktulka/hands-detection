import argparse
import dataclasses

import albumentations as A
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import blaze
from blaze.dataset import BlazeDataset
from blaze.loss import MultiBoxLoss
from blaze.model import BlazeModel
from blaze.model.detector import BlazeDetector


def od_collate_fn(batch: list) -> tuple:
    imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


@dataclasses.dataclass
class HyperParameters:
    learning_rate: float = 0.0001
    batch_size: int = 16
    epochs: int = 200
    detection_threshold: float = 0.5
    confidence_threshold: float = 0.8
    lr_scheduler_patience: int = 10


@dataclasses.dataclass
class Dataloaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader


@dataclasses.dataclass
class EpochLosses:
    epoch: int
    train_loss: float
    train_loc_loss: float
    train_class_loss: float
    val_loss: float

    def __str__(self):
        return (f'[{self.epoch + 1}] train loss: {self.train_loss:.3f} | val loss: {self.val_loss:.3f}\n'
                f'train loc loss: {self.train_loc_loss:.3f} | train class loss: {self.train_class_loss:.3f}')


class BlazeTrainer:
    def __init__(self,
                 dataloaders: Dataloaders,
                 hyper_params: HyperParameters):
        self.model = BlazeModel()
        self.model.load_anchors(blaze.ANCHORS_PATH)

        self.criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=blaze.DEVICE,
                                      dbox_list=self.model.dbox_list)
        self.optimizer = optim.Adam(self.model.parameters(), lr=hyper_params.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                           factor=0.5, patience=hyper_params.lr_scheduler_patience)

        self.hyper_params = hyper_params

        self.detector = BlazeDetector(self.model.dbox_list)

        self.dataloaders = dataloaders

    def train_model(self):
        self.model = self.model.to(blaze.DEVICE)

        for epoch in range(hyper_params.epochs):
            running_loss, running_loc_loss, running_class_loss = self.train_loop()

            self.model.eval()

            with torch.no_grad():
                val_loss, val_loc_loss, val_class_loss = self.val_loop()

                if epoch == hyper_params.epochs - 1:
                    val_grid_images = self.test_training()
                    plt.imsave('val_grid.jpg', val_grid_images.permute(1, 2, 0).cpu().numpy())

            epoch_losses: EpochLosses = self.calculate_epoch_loss(epoch, running_loss, running_loc_loss,
                                                                  running_class_loss, val_loss)

            print(epoch_losses)

            self.scheduler.step(val_loss)

            torch.save(self.model.state_dict(), blaze.MODEL_PATH)

    def train_loop(self) -> tuple[float, float, float]:
        running_loss = 0.
        running_loc_loss = 0.
        running_class_loss = 0.

        for images, targets in tqdm(self.dataloaders.train):
            images = images.to(blaze.DEVICE)
            targets = [ann.to(blaze.DEVICE) for ann in targets]

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss_l, loss_c = self.criterion(outputs, targets)
            loss = loss_l + loss_c

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_loc_loss += loss_l.item()
            running_class_loss += loss_c.item()

        return running_loss, running_loc_loss, running_class_loss

    def val_loop(self) -> tuple[float, float, float]:
        val_loss = 0.
        val_loc_loss = 0.
        val_class_loss = 0.
        for images, targets in self.dataloaders.val:
            images = images.to(blaze.DEVICE)
            targets = [ann.to(blaze.DEVICE) for ann in targets]
            outputs = self.model(images)
            loss_l, loss_c = self.criterion(outputs, targets)
            loss = loss_l + loss_c
            val_loss += loss.item()
            val_loc_loss += loss_l.item()
            val_class_loss += loss_c.item()

        return val_loss, val_loc_loss, val_class_loss

    def calculate_epoch_loss(self, epoch, running_loss, running_loc_loss, running_class_loss, val_loss) -> EpochLosses:
        train_loss = running_loss / len(self.dataloaders.train)
        train_loc_loss = running_loc_loss / len(self.dataloaders.train)
        train_class_loss = running_class_loss / len(self.dataloaders.train)
        val_loss = val_loss / len(self.dataloaders.val)

        return EpochLosses(epoch, train_loss, train_loc_loss, train_class_loss, val_loss)

    def test_training(self):
        images, targets = next(iter(self.dataloaders.val))

        outputs = self.model(images)

        detections = self.detector(outputs)

        classes = blaze.CLASSES

        imgs_with_boxes = []
        print(detections)
        for i in range(min(len(images), 32)):
            if len(detections[i, :, 0] > hyper_params.detection_threshold) > 0:
                filtered_dets = detections[i, detections[i, :, 0] > hyper_params.detection_threshold, :]
                print(filtered_dets)

                classes = [classes[int(pred_class)] for pred_class in filtered_dets[:, -1]]

                img_with_boxes = torchvision.utils.draw_bounding_boxes(
                    ((images[i] * 0.5 + 0.5) * 255).to(torch.uint8),
                    filtered_dets[:, 1:-1] * images.shape[-1],
                    classes,
                    "red"
                )

                classes = [classes[int(gt_class)] for gt_class in targets[i][:, -1]]
                img_with_boxes = torchvision.utils.draw_bounding_boxes(
                    img_with_boxes,
                    targets[i][:, :4] * images.shape[-1],
                    classes,
                    "green"
                )
                imgs_with_boxes.append(img_with_boxes.unsqueeze(0).cpu())
            else:
                imgs_with_boxes.append(((images[i] * 0.5 + 0.5) * 255).to(torch.uint8).unsqueeze(0).cpu())

        imgs_with_boxes = torch.cat(imgs_with_boxes)
        grid_images = torchvision.utils.make_grid(imgs_with_boxes, nrow=4)

        return grid_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train blaze model')
    parser.add_argument('--dataset', help='the dataset path', type=str, default='./dataset_path/')
    parser.add_argument('--batch_size', help='the batch size', type=int, default=16)
    parser.add_argument('--epochs', help='the number of epochs', type=int, default=100)
    parser.add_argument('--lr', help='the initial learning rate', type=float, default=0.003)
    parser.add_argument('--det_threshold', help='the detection threshold', type=float, default=0.5)
    parser.add_argument('--conf_threshold', help='the confidence threshold', type=float, default=0.8)
    parser.add_argument('--lr_sched_patience', help='learning rate scheduler patience', type=int, default=10)
    parser.add_argument('--workers', help='num of workers for dataloader', type=int, default=2)
    args = parser.parse_args()

    hyper_params = HyperParameters(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=200,
        detection_threshold=args.det_threshold,
        confidence_threshold=args.conf_threshold,
        lr_scheduler_patience=args.lr_sched_patience
    )

    augment = A.Compose([
        A.Affine(
            scale=(0.8, 1.7),
            translate_percent=(0.1, 0.1),
            rotate=15,
            shear=10,
            keep_ratio=True,
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomCropFromBorders(
            crop_left=0.05,
            crop_right=0.05,
            crop_top=0.05,
            crop_bottom=0.05,
            p=0.9,
        ),
        A.ColorJitter(brightness=0.2,
                      contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ToGray(p=0.1),
        A.RandomFog(p=0.5),
    ],
        bbox_params=A.BboxParams(
            format='albumentations')
    )

    train_dataset = BlazeDataset(args.dataset + '/labels/', image_size=blaze.IMAGE_SIZE, augment=augment)

    valid_dataset = BlazeDataset(args.dataset + '/labels/val/', image_size=blaze.IMAGE_SIZE)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyper_params.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=od_collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=hyper_params.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=od_collate_fn
    )

    dataloaders = Dataloaders(train_dataloader, val_dataloader)

    trainer = BlazeTrainer(dataloaders, hyper_params)

    trainer.train_model()
