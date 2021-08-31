import os

import matplotlib
import matplotlib.pyplot as plt

# Suppress pop-up windows during debugging
matplotlib.use('TkAgg')

class Draw_Curve:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        # Draw curve

        self.fig = plt.figure(clear=True)
       
        self.ax0 = self.fig.add_subplot(121, title="loss")
        self.ax1 = self.fig.add_subplot(122, title="acc")

        self.x_train_epoch_loss = []
        self.y_train_loss = []

        self.x_train_epoch_acc = []
        self.y_train_acc = []
        self.x_test_epoch_acc = []
        self.y_test_acc = []

    def save_curve(self):
       
        self.ax0.plot(
            self.x_train_epoch_loss, self.y_train_loss, "rs-", markersize="2", label="train"
        )
        self.ax0.set_ylabel("Training")
        self.ax0.set_xlabel("Epoch")
        self.ax0.legend()

        self.ax1.plot(
            self.x_train_epoch_acc, self.y_train_acc, "rs-", markersize="2", label="train"
        )
        self.ax1.plot(
            self.x_test_epoch_acc, self.y_test_acc, "bs-", markersize="2", label="test"
        )
        self.ax1.set_ylabel("Training")
        self.ax1.set_xlabel("Epoch")
        self.ax1.legend()


        self.fig.tight_layout() # 防止图像重叠
        save_path = os.path.join(self.dir_path, "train_log.jpg")
        self.fig.savefig(save_path)
