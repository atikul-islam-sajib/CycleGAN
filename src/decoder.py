import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels=64):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2

        self.kernel = 3
        self.stride = 2
        self.padding = 1

        self.layers = OrderedDict()

        self.decoder = self.down_block()

    def down_block(self):

        self.layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )

        self.layers["instance_norm"] = nn.InstanceNorm2d(num_features=self.out_channels)

        self.layers["ReLU"] = nn.ReLU(inplace=True)

        return nn.Sequential(self.layers)

    def forward(self, x):
        if x is not None:
            return self.decoder(x)
        else:
            raise Exception("Input to the model cannot be empty".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Down Block for netG".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=64,
        help="Number of channels in the input image".capitalize(),
    )
    args = parser.parse_args()

    in_channels = args.in_channels

    layers = []
    for _ in range(2):
        layers.append(DownBlock(in_channels=in_channels))
        in_channels *= 2
