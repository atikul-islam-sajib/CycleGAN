import argparse
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(GANLoss, self).__init__()

        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, predicted, actual):
        if type(predicted) == torch.Tensor and type(actual) == torch.Tensor:
            return self.loss(predicted, actual)
        else:
            raise ValueError("Both predicted and actual must be provided".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Loss for CycleGAN".title())

    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="Reduction method for loss calculation".capitalize(),
    )

    args = parser.parse_args()

    loss = GANLoss(reduction=args.reduction)

    predicted = torch.tensor([1.0, 1.0, 0.0, 0.0])
    actual = torch.tensor([1.0, 0.0, 0.0, 0.0])

    print("Total loss {}".format(loss(predicted, actual)))
