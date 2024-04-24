import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(InputBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = 7
        self.stride = 1
        self.padding = 3

        self.layers = OrderedDict()

        self.input = self.block()

    def block(self):

        self.layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            padding_mode="reflect",
        )

        self.layers["instance_norm"] = nn.InstanceNorm2d(num_features=self.out_channels)

        self.layers["ReLU"] = nn.ReLU(inplace=True)

        return nn.Sequential(self.layers)

    def forward(self, x):
        if x is not None:
            return self.input(x)
        else:
            raise Exception("Input to the model cannot be empty".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Input Block for netG".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of channels in the input image".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Number of channels in the output image".capitalize(),
    )

    args = parser.parse_args()

    input = InputBlock(in_channels=args.in_channels, out_channels=args.out_channels)

    assert input(torch.randn(1, 3, 256, 256)).shape == (1, 64, 256, 256)
