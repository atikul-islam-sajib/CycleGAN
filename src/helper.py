import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from utils import params, dump, load, weight_init

from generator import Generator
from discriminator import Discriminator
from loss.gan_loss import GANLoss
from loss.cycle_loss import CycleConsistencyLoss


def load_dataloader():
    if os.path.exists(params()["path"]["processed_path"]):
        train_dataloader = load(
            filename=os.path.join(
                params()["path"]["processed_path"], "train_dataloader.pkl"
            )
        )
        test_dataloader = load(
            filename=os.path.join(
                params()["path"]["processed_path"], "test_dataloader.pkl"
            )
        )

        return {
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
        }


def helper(**kwargs):

    in_channels = kwargs["in_channels"]
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    lr_scheduler = kwargs["lr_scheduler"]

    netG_XtoY = Generator(in_channels=in_channels)
    netG_YtoX = Generator(in_channels=in_channels)

    netD_X = Discriminator(in_channels=in_channels)
    netD_Y = Discriminator(in_channels=in_channels)

    netG_XtoY.apply(weight_init)
    netG_YtoX.apply(weight_init)

    netD_X.apply(weight_init)
    netD_Y.apply(weight_init)

    if adam:
        optimizer_G = optim.Adam(
            params=list(netG_XtoY.parameters()) + list(netG_YtoX.parameters()),
            lr=lr,
            betas=(params()["model"]["beta1"], params()["model"]["beta2"]),
        )
        optimizer_D_X = optim.Adam(
            params=netD_X.parameters(),
            lr=lr,
            betas=(params()["model"]["beta1"], params()["model"]["beta2"]),
        )
        optimizer_D_Y = optim.Adam(
            params=netD_Y.parameters(),
            lr=lr,
            betas=(params()["model"]["beta1"], params()["model"]["beta2"]),
        )

    if SGD:
        optimizer_G = optim.SGD(
            params=list(netG_XtoY.parameters()) + list(netG_YtoX.parameters()),
            lr=lr,
            momentum=params()["model"]["momentum"],
        )
        optimizer_D_X = optim.SGD(
            params=netD_X.parameters(), lr=lr, momentum=params()["model"]["momentum"]
        )
        optimizer_D_Y = optim.SGD(
            params=netD_Y.parameters(), lr=lr, momentum=params()["model"]["momentum"]
        )

    if lr_scheduler:
        schedulerG = StepLR(
            optimizer=optimizer_G,
            step_size=params()["model"]["step_size"],
            gamma=params()["model"]["gamma"],
        )
        scheduler_D_X = StepLR(
            optimizer=optimizer_D_X,
            step_size=params()["model"]["step_size"],
            gamma=params()["model"]["gamma"],
        )
        scheduler_D_Y = StepLR(
            optimizer=optimizer_D_Y,
            step_size=params()["model"]["step_size"],
            gamma=params()["model"]["gamma"],
        )

    try:
        dataloader = load_dataloader()

    except Exception as e:
        print("The exception is: %s" % e)

    adversarial_loss = GANLoss(reduction="mean")
    cycle_consistency_loss = CycleConsistencyLoss(reduction="mean")

    return {
        "netG_XtoY": netG_XtoY,
        "netG_YtoX": netG_YtoX,
        "netD_X": netD_X,
        "netD_Y": netD_Y,
        "optimizer_G": optimizer_G,
        "optimizer_D_X": optimizer_D_X,
        "optimizer_D_Y": optimizer_D_Y,
        "schedulerG": schedulerG,
        "scheduler_D_X": scheduler_D_X,
        "scheduler_D_Y": scheduler_D_Y,
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
        "adversarial_loss": adversarial_loss,
        "cycle_consistency_loss": cycle_consistency_loss,
    }


if __name__ == "__main__":
    helper()
