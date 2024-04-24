import argparse
import torch
import torch.nn as nn


class CycleConsistencyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CycleConsistencyLoss, self).__init__()

        self.reduction = reduction
        self.loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, reconstructed, real):

        if type(reconstructed) == torch.Tensor and type(real) == torch.Tensor:
            return self.loss(reconstructed, real)

        else:
            raise ValueError(
                "Both reconstructed and real must be provided".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cycle Consistency Loss for CycleGAN".title()
    )

    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="Reduction method for loss calculation".capitalize(),
    )

    args = parser.parse_args()

    loss = CycleConsistencyLoss(reduction=args.reduction)

    reconstructed = torch.tensor([1.0, 1.0, 1.0, 0.0])
    real = torch.tensor([1.0, 1.0, 1.0, 0.0])

    print("Total loss {}".format(loss(reconstructed, real)))
