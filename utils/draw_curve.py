import os

import matplotlib
import matplotlib.pyplot as plt

# Suppress pop-up windows during debugging
matplotlib.use("agg")
# self.fig.tight_layout() # 防止图像重叠
class Draw_Curve:
    def __init__(self, dir_path):

        self.dir_path = dir_path

        # Draw curve
        self.fig = plt.figure(clear=True)
        self.ax0 = self.fig.add_subplot(121, title="Training loss")
        self.ax1 = self.fig.add_subplot(122, title="Testing CMC/mAP")
        self.x_epoch_loss = []
        self.x_epoch_test = []
        self.y_train_loss = []
        self.y_test = {}
        self.y_test["top1"] = []
        self.y_test["mAP"] = []

    def save_curve(self):

        self.ax0.plot(
            self.x_epoch_loss, self.y_train_loss, "bs-", markersize="2", label="test"
        )
        self.ax0.set_ylabel("Training")
        self.ax0.set_xlabel("Epoch")
        self.ax0.legend()

        self.ax1.plot(
            self.x_epoch_test, self.y_test["top1"], "rs-", markersize="2", label="top1"
        )
        self.ax1.plot(
            self.x_epoch_test, self.y_test["mAP"], "bs-", markersize="2", label="mAP"
        )
        self.ax1.set_ylabel("%")
        self.ax1.set_xlabel("Epoch")
        self.ax1.legend()

        save_path = os.path.join(self.dir_path, "train_log.jpg")
        self.fig.savefig(save_path)
