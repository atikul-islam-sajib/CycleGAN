import argparse
from collections import OrderedDict
import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        kernel=4,
        stride=2,
        padding=1,
        is_instance_norm=False,
        is_lr=True,
    ):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.use_norm = is_instance_norm
        self.use_lr = is_lr

        self.model = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )

        if self.use_norm:
            layers["instance_norm"] = nn.InstanceNorm2d(num_features=self.out_channels)

        if self.use_lr:
            layers["LeakyReLU"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if x is not None:
            return self.model(x)
        else:
            raise Exception("Input to the model cannot be empty".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Discriminator Block for netD".title()
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of channels in the input".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=64,
        help="Number of channels in the output".capitalize(),
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=4,
        help="Kernel size of the convolution".capitalize(),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride of the convolution".capitalize(),
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=1,
        help="Padding of the convolution".capitalize(),
    )
    parser.add_argument(
        "--is_instance_norm",
        type=bool,
        default=False,
        help="Whether to use instance normalization".capitalize(),
    )
    parser.add_argument(
        "--is_lr",
        type=bool,
        default=True,
        help="Whether to use LeakyReLU".capitalize(),
    )

    args = parser.parse_args()

    netD = DiscriminatorBlock(
        in_channels=args.in_channels, out_channels=args.out_channels
    )
