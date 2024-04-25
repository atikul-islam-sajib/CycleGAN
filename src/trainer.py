import sys
import os
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import device_init
from helper import helper


class Trainer:

    def __init__(
        self,
        in_channels=3,
        epochs=1000,
        lr=0.0002,
        device="mps",
        adam=True,
        SGD=False,
        lr_scheduler=False,
        is_display=True,
    ):
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.lr_scheduler = lr_scheduler
        self.is_display = is_display

        self.device = device_init(device=self.device)

        init = helper(
            in_channels=self.in_channels,
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
            lr_scheduler=self.lr_scheduler,
        )

        self.netG_XtoY = init["netG_XtoY"].to(self.device)
        self.netG_YtoX = init["netG_YtoX"].to(self.device)

        self.netD_X = init["netD_X"].to(self.device)
        self.netD_Y = init["netD_Y"].to(self.device)

        self.optimizer_G = init["optimizer_G"]
        self.optimizer_D_X = init["optimizer_D_X"]
        self.optimizer_D_Y = init["optimizer_D_Y"]

        self.schedulerG = init["schedulerG"]
        self.scheduler_D_X = init["scheduler_D_X"]
        self.scheduler_D_Y = init["scheduler_D_Y"]

        self.train_dataloader = init["train_dataloader"]
        self.test_dataloader = init["test_dataloader"]

        self.adversarial_loss = init["adversarial_loss"]
        self.cycle_consistency_loss = init["cycle_consistency_loss"]

    def l1(self, model):
        pass

    def l2(self, model):
        pass

    def elastic_net(self, model):
        pass

    def saved_checkpoints(self, **kwargs):
        pass

    def show_progress(self, **kwargs):
        pass

    def saved_trained_images(self, **kwargs):
        pass

    def train(self):
        pass

    @staticmethod
    def plot_history():
        pass
