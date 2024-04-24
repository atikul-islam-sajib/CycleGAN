import sys
import os
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from discriminator_block import DiscriminatorBlock

from utils import params


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.layers = []
        self.in_channels = in_channels
        self.out_channels = 64
        self.kernel = 4
        self.stride = 2
        self.padding = 1

        for idx in range(3):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel=self.kernel,
                    stride=self.stride,
                    padding=self.padding,
                    is_instance_norm=False if idx == 0 else True,
                )
            )

            self.in_channels = self.out_channels
            self.out_channels *= 2

        for idx in range(2):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel=self.kernel,
                    stride=self.stride // 2,
                    padding=self.padding,
                    is_instance_norm=True if idx == 0 else False,
                    is_lr=True if idx == 0 else False,
                )
            )
            self.in_channels = self.out_channels
            self.out_channels //= self.out_channels

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
            raise Exception(
                "Please provide a model to calculate the total number of parameters".capitalize()
            )


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

    args = parser.parse_args()

    netD = Discriminator(in_channels=args.in_channels)

    assert netD(torch.randn(1, 3, 256, 256)).size() == (1, 1, 30, 30)

    if os.path.exists(params()["path"]["files_path"]):

        print(summary(model=netD, input_size=(3, 256, 256), batch_size=1))

        draw_graph(
            model=netD, input_data=torch.randn(1, 3, 256, 256)
        ).visual_graph.render(
            filename=os.path.join(params()["path"]["files_path"], "netD"), format="png"
        )
    else:
        raise Exception(
            "Please create the files directory in the root of the project".capitalize()
        )

    print(
        "Total params of the netD is {}".format(Discriminator.total_params(model=netD))
    )
