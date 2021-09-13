import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
from dataloader.market1501 import Market1501
import torchvision.transforms as T
from dataloader.collate_batch import val_collate_fn

matplotlib.use("agg")
import matplotlib.pyplot as plt
from utils import reid_util

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15"
)
# parser.add_argument(
#     "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15"
# )
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)
# other
parser.add_argument("--img_height", type=int, default=128)
parser.add_argument("--img_width", type=int, default=64)

parser.add_argument(
    "--model_path", type=str, default="./checkpoints/person_reid/net_final-2.pth"
)

opt = parser.parse_args()

######################################################################
result = scipy.io.loadmat("checkpoints/person_reid/pytorch_result.mat")

query_feature = torch.FloatTensor(result["query_f"])
query_cam = result["query_cam"][0]
query_label = result["query_label"][0]
gallery_feature = torch.FloatTensor(result["gallery_f"])
gallery_cam = result["gallery_cam"][0]
gallery_label = result["gallery_label"][0]

qf = np.array(query_feature.cpu())
gf = np.array(gallery_feature.cpu())
dist = reid_util.cosine_dist(qf, gf)
rank_results = np.argsort(dist)[:, ::-1]

# Computing CMC and mAP------------------------------------------------------------------------
print("Computing CMC and mAP ...")
APs, CMC = [], []
for _, data in enumerate(zip(rank_results, query_cam, query_label)):
    a_rank, query_camid, query_pid = data
    ap, cmc = reid_util.compute_AP(
        a_rank, query_camid, query_pid, gallery_cam, gallery_label
    )
    APs.append(ap), CMC.append(cmc)
MAP = np.array(APs).mean()
min_len = min([len(cmc) for cmc in CMC])
CMC = [cmc[:min_len] for cmc in CMC]
CMC = np.mean(np.array(CMC), axis=0)
print(
    "Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f" % (CMC[0], CMC[4], CMC[9], MAP)
)


def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


i = 1
index = sort_img(
    query_feature[i],
    query_label[i],
    query_cam[i],
    gallery_feature,
    gallery_label,
    gallery_cam,
)


test_transforms = transform = T.Compose(
    [
        T.Resize((opt.img_height, opt.img_width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


query_dataset = Market1501(
    root=opt.data_dir, data_folder="query", transform=test_transforms, relabel=False
)
gallery_dataset = Market1501(
    root=opt.data_dir,
    data_folder="bounding_box_test",
    transform=test_transforms,
    relabel=False,
)

########################################################################
# Visualize the rank result
query_path, _, _ = query_dataset.dataset[i]
query_label1 = query_label[i]
# print(query_path)
print("Top 10 images are as follow:")
try:  # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis("off")
    imshow(query_path, "query")
    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis("off")

        img_path, _, _ = gallery_dataset.dataset[index[i]]
        label = gallery_label[index[i]]

        imshow(img_path)
        if label == query_label:
            ax.set_title("%d" % (i + 1), color="green")
        else:
            ax.set_title("%d" % (i + 1), color="red")
        print(img_path)
except RuntimeError:
    # for i in range(10):
    #     img_path = image_datasets.imgs[index[i]]
    #     print(img_path[0])
    print("This is error RuntimeError")
    print(
        "If you want to see the visualization of the ranking result, graphical user interface is needed."
    )

fig.savefig("show.png")
