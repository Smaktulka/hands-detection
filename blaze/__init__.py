import torch
import os

from conf import RESOURCES_PATH

DEFAULT_CLASSES_NUM = 2 + 1  # two classes are informative (dislike, like), one - is by default treated as background
IMAGE_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


WEIGHTS_PATH = os.path.join(RESOURCES_PATH, "weights")
MODEL_PATH = os.path.join(WEIGHTS_PATH, "blaze_model/blaze_model.pt")
ANCHORS_PATH = os.path.join(RESOURCES_PATH, "anchors.npy")

CLASSES = ['dislike', 'like', 'no_gesture']


