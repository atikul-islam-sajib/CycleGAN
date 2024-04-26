import sys
import os
import argparse

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from test import TestModel


def cli():

    parser = argparse.ArgumentParser(description="CLI for CycleGAN".title())
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the zip file containing the dataset".capitalize(),
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of the image".capitalize(),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of the batch".capitalize(),
    )

    parser.add_argument(
        "--split_size",
        type=float,
        default=0.25,
        help="Size of the batch".capitalize(),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of epochs to train the model".capitalize(),
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="in_channels of the input image".capitalize(),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate to train the model".capitalize(),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to train the model".capitalize(),
    )

    parser.add_argument(
        "--is_display",
        type=bool,
        default=True,
        help="Display the training progress".title(),
    )

    parser.add_argument(
        "--adam", type=bool, default=True, help="Adam optimizer".capitalize()
    )

    parser.add_argument(
        "--SGD", type=bool, default=True, help="SGD optimizer".capitalize()
    )
    parser.add_argument(
        "--test_result",
        type=str,
        default="test_result",
        help="Define the path to save the test result".capitalize(),
    )
    parser.add_argument(
        "--train",
        type=str,
        default="train_result",
        help="Define the path to save the train result".capitalize(),
    )
    args = parser.parse_args()

    if args.train:
        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        loader.unzip_folder()
        loader.create_dataloader()

        Loader.plot_images()
        Loader.dataset_details()

        trainer = Trainer(
            in_channels=args.in_channels,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            is_display=args.is_display,
            adam=args.adam,
        )

        trainer.train()

        trainer.plot_history()

    elif args.test_result:

        test_model = TestModel(in_channels=args.in_channels, device=args.device)

        test_model.test()


if __name__ == "__main__":
    cli()
