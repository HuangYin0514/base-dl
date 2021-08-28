import os
import glob
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import datasets

from models import *
from util import util, Logger

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Person ReID Frame")
# base
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="hymenoptera")
parser.add_argument("--phase", type=str, default="train")
# env setting
parser.add_argument("--random_seed", type=int, default="1")
# data
parser.add_argument("--img_height", type=int, default=12)
parser.add_argument("--img_width", type=int, default=12)
parser.add_argument(
    "--train_dir", type=str, default="./datasets/hymenoptera_data/train"
)
parser.add_argument("--test_dir", type=str, default="./datasets/hymenoptera_data/val")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# Optimizer
parser.add_argument("--lr", type=float, default=0.1)
# train
parser.add_argument("--start_epoch", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=1)
# parse
opt = parser.parse_args()

util.print_options(opt)

# env setting ==============================================================================
# Fix random seed
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)
# speed up compution
torch.backends.cudnn.benchmark = True
# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# data ============================================================================================================
# data Augumentation
train_transforms = T.Compose(
    [
        T.Resize(((opt.img_height, opt.img_width)), interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transforms = T.Compose(
    [
        T.Resize(((opt.img_height, opt.img_width)), interpolation=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# data loader'
train_dataset = datasets.ImageFolder(root=opt.train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=opt.test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
)
class_names = train_dataset.classes

# model ============================================================================================================
model = Resnet18Custom()
model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)

# criterion ============================================================================================================
criterion = F.cross_entropy

# optimizer ============================================================================================================
optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)

# scheduler ============================================================================================================
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# save dir path ============================================================================================================
save_dir_path = os.path.join(opt.checkpoints_dir, opt.name)

# Training and test ============================================================================================================
def train():
    start_time = time.time()

    # Logger instance
    logger = Logger.Logger(save_dir_path)
    # logger.info('-' * 10)
    # logger.info(vars(opt))
    # logger.info(model)
    logger.info("train starting...")

    for epoch in range(opt.num_epochs):
        model.train()
        # Training
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # net ---------------------
            optimizer.zero_grad()
            output = model(inputs)

            _, preds = torch.max(output, 1)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            # --------------------------

            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)

        if epoch % 1 == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            time_remaining = (
                (opt.num_epochs - epoch) * (time.time() - start_time) / (epoch + 1)
            )
            logger.info(
                "Epoch:{}/{} \tLoss:{:.4f} \tETA:{:.0f}h{:.0f}m".format(
                    epoch + 1,
                    opt.num_epochs,
                    epoch_loss,
                    time_remaining // 3600,
                    time_remaining / 60 % 60,
                )
            )

    print("training is done !")


if __name__ == "__main__":
    train()
