import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F

from dataset import HandsDataset
from model import BlazePalm

classes_ids = {
    0: 'like',
    1: 'dislike',
    2: 'timeout',
    3: 'stop'
}

data_dir = "./train"
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = HandsDataset("./train", transform)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, labels in dataloader:
    break

print(labels)
classes_num = 4
model = BlazePalm(classes_num=4, batch_size=32)
out = model(images)
print(out)
criterion = torch.nn.CrossEntropyLoss()
# out = out[:, :, :]
# clamped = torch.clamp(labels.view(-1), 0, 3)
# batch_out = out.view(-1, 4)
loss = criterion(out, labels)
print(out.shape)
print(loss)
# print(max_.item())
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(images.shape)
# print(labels)

# num_epoch = 5
# train_losses, val_losses = [], []
#
# for epoch in range(num_epoch):
#     model.train()
#     running_loss = 0.0
#     for images, labels in dataloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * labels.size(0)
#
#     train_loss = running_loss / len(dataloader.dataset)
#     train_losses.append(train_loss)
#
#     model.eval()
#     running_loss = 0.0
#     # with torch.no_grad():
#         # for images, labels in dataloader
#
#

