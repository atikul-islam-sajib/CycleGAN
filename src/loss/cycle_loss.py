import argparse
import torch
import torch.nn as nn


class CycleConsistencyLoss(nn.Module):
    """
    Implements the Cycle Consistency Loss using L1 Loss, typically used in models
    like CycleGAN where the goal is to minimize the difference between the original
    and reconstructed images in a cyclic manner.

    Attributes:
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
        loss (torch.nn.modules.loss): The L1 Loss instance with the specified reduction.

    Methods:
        forward(reconstructed, real): Computes the L1 loss between reconstructed and real images.
    """

    def __init__(self, reduction="mean"):
        """
        Initializes the CycleConsistencyLoss module with the specified reduction method.

        Args:
            reduction (str): The method used to reduce the L1 loss over the batch. Defaults to "mean".
        """
        super(CycleConsistencyLoss, self).__init__()

        self.reduction = reduction
        self.loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, reconstructed, real):
        """
        Forward pass of the loss module to compute the L1 loss.

        Args:
            reconstructed (torch.Tensor): The reconstructed images from the generator.
            real (torch.Tensor): The original real images.

        Returns:
            torch.Tensor: The computed L1 loss as per the specified reduction method.

        Raises:
            ValueError: If inputs are not torch tensors.
        """
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
