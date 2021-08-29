import os
import time
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import datasets

from models import *
from utils import util, logger, draw_curve

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="hymenoptera")
# data
parser.add_argument("--img_height", type=int, default=4)
parser.add_argument("--img_width", type=int, default=4)
parser.add_argument(
    "--train_dir", type=str, default="./datasets/hymenoptera_data/train"
)
parser.add_argument("--test_dir", type=str, default="./datasets/hymenoptera_data/val")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)

# parse
opt = parser.parse_args()
util.print_options(opt)

# env setting ==============================================================================
# Fix random seed
random_seed = 2021
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# speed up compution
torch.backends.cudnn.benchmark = True
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
# save dir path
save_dir_path = os.path.join(opt.checkpoints_dir, opt.name)
# Logger instance
logger = logger.Logger(save_dir_path)
# draw curve instance
curve = draw_curve.Draw_Curve(save_dir_path)

# data ============================================================================================================
# data Augumentation
train_transforms = T.Compose(
    [
        T.Resize(((opt.img_height, opt.img_width)), interpolation=3),
        T.RandomResizedCrop((opt.img_height, opt.img_width)),
        # T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transforms = T.Compose(
    [
        # T.Resize(((opt.img_height, opt.img_width)), interpolation=3),
        # T.Resize(256),
        T.CenterCrop(224),
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

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.test_batch_size, num_workers=opt.num_workers
)
# model ============================================================================================================
model = Resnet18Custom()
model = model.to(device)
if device == "cuda":
    model = torch.nn.DataParallel(model)

# criterion ============================================================================================================
criterion = F.cross_entropy

# optimizer ============================================================================================================
# optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# scheduler ============================================================================================================
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Training and test ============================================================================================================
def train():
    start_time = time.time()

    # logger.info('-' * 10)
    # logger.info(vars(opt))
    # logger.info(model)
    # logger.info("train starting...")

    for epoch in range(opt.num_epochs):
        model.train()

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
            running_corrects += torch.sum(preds == labels.data)

        # scheduler
        scheduler.step()

        # print train infomation
        if epoch % 1 == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            time_remaining = (
                (opt.num_epochs - epoch) * (time.time() - start_time) / (epoch + 1)
            )

            logger.info(
                "Epoch:{}/{} \tTrain Loss:{:.4f} \tAcc:{:.4f} \tETA:{:.0f}h{:.0f}m".format(
                    epoch + 1,
                    opt.num_epochs,
                    epoch_loss,
                    epoch_acc,
                    time_remaining // 3600,
                    time_remaining / 60 % 60,
                )
            )

            # plot curve
            curve.x_train_epoch_loss.append(epoch + 1)
            curve.y_train_loss.append(epoch_loss)
            curve.x_train_epoch_acc.append(epoch + 1)
            curve.y_train_acc.append(epoch_acc)

        # test
        if epoch % 1 == 0:
            test(epoch)

    curve.save_curve()
    print("training is done !")


def test(epoch):
    model.eval()

    test_loss = 0.0
    test_corrects = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # net ---------------------
        output = model(inputs)

        _, preds = torch.max(output, 1)

        loss = criterion(output, labels)
        # --------------------------

        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)

    # print test infomation
    if epoch % 1 == 0:
        epoch_loss = test_loss / len(test_loader.dataset)
        epoch_acc = test_corrects.double() / len(test_loader.dataset)

        logger.info(
            "Epoch:{}/{} \tTest Loss:{:.4f} \tAcc:{:.4f}".format(
                epoch + 1,
                opt.num_epochs,
                epoch_loss,
                epoch_acc,
            )
        )

        curve.x_test_epoch_acc.append(epoch + 1)
        curve.y_test_acc.append(epoch_acc)

    # print("test is done !")


if __name__ == "__main__":
    train()
