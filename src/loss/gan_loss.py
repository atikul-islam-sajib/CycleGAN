import argparse
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Custom GAN Loss module that calculates the Mean Squared Error loss between
    the predicted and actual tensor values.

    Attributes:
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean' or 'sum'.
        loss (torch.nn.modules.loss): The MSE Loss instance with specified reduction.

    Methods:
        forward(predicted, actual): Computes the MSE loss between predicted and actual values.
    """

    def __init__(self, reduction="mean"):
        """
        Initializes the GANLoss module with the specified reduction method.

        Args:
            reduction (str): The method used to reduce the MSE loss over the batch. Defaults to "mean".
        """
        super(GANLoss, self).__init__()

        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, predicted, actual):
        """
        Forward pass of the loss module to compute the MSE loss.

        Args:
            predicted (torch.Tensor): The predicted outputs from the model.
            actual (torch.Tensor): The actual ground truth values.

        Returns:
            torch.Tensor: The computed MSE loss as per the specified reduction method.

        Raises:
            ValueError: If inputs are not torch tensors.
        """
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
