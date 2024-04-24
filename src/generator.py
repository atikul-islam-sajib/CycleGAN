import sys
import os
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from input_block import InputBlock
from decoder import DownBlock
from encoder import UpsampleBlock
from residual_block import ResidualBlock

from utils import params


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 64
        self.kernel = 7
        self.stride = 1
        self.padding = 3
        self.num_repetitive = 9

        self.layers = []

        self.layers.append(
            InputBlock(in_channels=self.in_channels, out_channels=self.out_channels)
        )

        for _ in range(2):
            self.layers.append(DownBlock(in_channels=self.out_channels))
            self.out_channels *= 2

        for _ in range(self.num_repetitive):
            self.layers.append(ResidualBlock(in_channels=self.out_channels))

        for _ in range(2):
            self.layers.append(UpsampleBlock(in_channels=self.out_channels))
            self.out_channels //= 2

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=3,
                    kernel_size=self.kernel,
                    stride=self.stride,
                    padding=self.padding,
                    padding_mode="reflect",
                ),
                nn.Tanh(),
            )
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if x is not None:
            return self.model(x)
        else:
            raise Exception("Input to the model cannot be empty".capitalize())

    @staticmethod
    def total_params(model=None):
        if model is not None:
            return sum(params.numel() for params in model.parameters())

        else:
            raise Exception("Input to the model cannot be empty".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator model for CycleGAN".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of channels in the input".capitalize(),
    )

    args = parser.parse_args()

    if args.in_channels:
        netG = Generator(in_channels=args.in_channels)

        assert netG(torch.randn(1, args.in_channels, 256, 256)).size() == (
            1,
            3,
            256,
            256,
        )  # Model output verify

        print(summary(netG, input_size=(args.in_channels, 256, 256)))

        if os.path.exists(params()["path"]["files_path"]):
            draw_graph(
                model=netG, input_data=torch.randn(1, args.in_channels, 256, 256)
            ).visual_graph.render(
                filename=os.path.join(params()["path"]["files_path"], "netG"),
                format="png",
            )
        else:
            raise Exception("Model architecture cannot be saved".capitalize())

        print("Total params of the Generator: {}".format(Generator.total_params(netG)))

    else:
        raise Exception("Input to the model cannot be empty".capitalize())
