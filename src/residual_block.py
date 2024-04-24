import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=256):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.kernel = 3
        self.stride = 1
        self.padding = 1

        self.layers = OrderedDict()

        self.residual = self.residual_block()

    def residual_block(self):

        for idx in range(2):
            self.layers["conv_{}".format(idx)] = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
            )
            self.layers["instance_norm_{}".format(idx)] = nn.InstanceNorm2d(
                num_features=self.out_channels
            )

            if idx % 2 == 0:
                self.layers["ReLU_{}".format(idx)] = nn.ReLU(inplace=True)

        return nn.Sequential(self.layers)

    def forward(self, x):
        if x is not None:
            return x + self.residual(x)
        else:
            raise Exception("Input to the model cannot be empty".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Residual Block for netG".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=256,
        help="Number of channels in the input image".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels

    model = nn.Sequential(*[ResidualBlock(in_channels=in_channels) for _ in range(9)])
