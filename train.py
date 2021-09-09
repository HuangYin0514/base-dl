import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from loss.crossEntropyLabelSmoothLoss import CrossEntropyLabelSmoothLoss
from dataloader.collate_batch import train_collate_fn, val_collate_fn
from dataloader.market1501 import Market1501
from models import *
from utils import draw_curve, load_network, logger, util

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15_reduce"
)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)
# other
parser.add_argument("--img_height", type=int, default=4)
parser.add_argument("--img_width", type=int, default=2)

# parser.add_argument("--Resize", type=int, default=2)
# parser.add_argument("--CenterCrop", type=int, default=2)

# RandomResizedCrop=224
# Resize=256
# CenterCrop=224

# parse
opt = parser.parse_args()
util.print_options(opt)

# env setting ==============================================================================
# Fix random seed
random_seed = 2021
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)  # Numpy module.
random.seed(random_seed)  # Python random module.
torch.backends.cudnn.deterministic = True
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
        T.Resize((opt.img_height, opt.img_width), interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = transform = T.Compose(
    [
        T.Resize((opt.img_height, opt.img_width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# data loader
train_dataset = Market1501(
    root=opt.data_dir,
    data_folder="bounding_box_train",
    transform=train_transforms,
    relabel=True,
)

num_classes = train_dataset.num_pids

query_dataset = Market1501(
    root=opt.data_dir, data_folder="query", transform=train_transforms, relabel=False
)
gallery_dataset = Market1501(
    root=opt.data_dir,
    data_folder="bounding_box_test",
    transform=train_transforms,
    relabel=False,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    collate_fn=train_collate_fn,
)

query_loader = torch.utils.data.DataLoader(
    query_dataset,
    batch_size=opt.test_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    collate_fn=val_collate_fn,
)
gallery_loader = torch.utils.data.DataLoader(
    gallery_dataset,
    batch_size=opt.test_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    collate_fn=val_collate_fn,
)

# model ============================================================================================================
model = Resnet_pcb(num_classes)
model = model.to(device)

# criterion ============================================================================================================
criterion = F.cross_entropy
ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes)

# optimizer ============================================================================================================
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr=0.1
base_param_ids = set(map(id, model.backbone.parameters()))
new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
param_groups = [{'params': model.backbone.parameters(), 'lr': lr/10},
                {'params': new_params, 'lr': lr}]
optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True)

# # scheduler ============================================================================================================
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Training and test ============================================================================================================
def train():
    start_time = time.time()

    for epoch in range(opt.num_epochs):
        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # net ---------------------
            optimizer.zero_grad()

            parts_scores = model(inputs)

            # parts loss-------------------------------------------------
            part_loss = 0
            for logits in parts_scores:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss += stripe_loss

            loss = part_loss

            loss.backward()
            optimizer.step()
            # --------------------------

            running_loss += loss.item() * inputs.size(0)
            

        # scheduler
        scheduler.step()

        # print train infomation
        if epoch % 1 == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            time_remaining = (
                (opt.num_epochs - epoch) * (time.time() - start_time) / (epoch + 1)
            )

            logger.info(
                "Epoch:{}/{} \tTrain Loss:{:.4f} \tETA:{:.0f}h{:.0f}m".format(
                    epoch + 1,
                    opt.num_epochs,
                    epoch_loss,
                    time_remaining // 3600,
                    time_remaining / 60 % 60,
                )
            )

            # plot curve
            curve.x_epoch_loss.append(epoch + 1)
            curve.y_train_loss.append(epoch_loss)
           

        # test
        # if epoch % 1 == 0:
        #     test(epoch)

    # Save the loss curve
    curve.save_curve()
    # Save final model weights
    load_network.save_network(model, save_dir_path, "final")

    print("training is done !")


# def test(epoch):
#     model.eval()

#     test_loss = 0.0
#     test_corrects = 0

#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         # net ---------------------
#         output = model(inputs)

#         _, preds = torch.max(output, 1)

#         loss = criterion(output, labels)
#         # --------------------------

#         test_loss += loss.item() * inputs.size(0)
#         test_corrects += torch.sum(preds == labels.data)

#     # print test infomation
#     if epoch % 1 == 0:
#         epoch_loss = test_loss / len(test_loader.dataset)
#         epoch_acc = test_corrects.double() / len(test_loader.dataset)

#         logger.info(
#             "Epoch:{}/{} \tTest Loss:{:.4f} \tAcc:{:.4f}".format(
#                 epoch + 1,
#                 opt.num_epochs,
#                 epoch_loss,
#                 epoch_acc,
#             )
#         )

#         curve.x_test_epoch_acc.append(epoch + 1)
#         curve.y_test_acc.append(epoch_acc)

#     # print("test is done !")


if __name__ == "__main__":
    train()
