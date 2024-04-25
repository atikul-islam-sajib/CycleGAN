import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from utils import params, dump, load, weight_init

from generator import Generator
from discriminator import Discriminator
from loss.gan_loss import GANLoss
from loss.cycle_loss import CycleConsistencyLoss


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

    if adam:
        pass


if __name__ == "__main__":
    helper()
