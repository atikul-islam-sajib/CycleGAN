import os
import sys
import argparse
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import load, params, device_init
from generator import Generator


class TestModel:
    def __init__(self, in_channels=3, device="mps", create_gif=False):
        self.in_channels = in_channels
        self.device = device_init(device=device)
        self.config = params()
        self.is_gif = create_gif

        self.netG_XtoY = Generator(in_channels=self.in_channels)
        self.netGYtoX = Generator(in_channels=self.in_channels)

        self.netG_XtoY.to(self.device)
        self.netGYtoX.to(self.device)

    def load_dataloader(self):
        if os.path.exists(self.config["path"]["processed_path"]):
            test_dataloader = load(
                filename=os.path.join(
                    self.config["path"]["processed_path"], "dataloader.pkl"
                )
            )

            return test_dataloader
        else:
            raise FileNotFoundError(
                f"{self.config['path']['processed_path']} does not exist"
            )

    def select_best_model(self):
        if os.path.exists(self.config["path"]["best_model"]):
            return torch.load(
                os.path.join(self.config["path"]["best_model"], "best_model.pth")
            )

        else:
            raise FileNotFoundError(
                f"{self.config['path']['best_model']} does not exist"
            )

    def image_normalized(self, **kwargs):
        return (kwargs["image"] - kwargs["image"].min()) / (
            kwargs["image"].max() - kwargs["image"].min()
        )

    def create_gif(self):
        if os.path.exists(self.config["path"]["train_results"]):
            self.images = []

            if len(os.listdir(self.config["path"]["train_results"])) == 0:

                print("No images to create gif".capitalize())

            else:
                for image in os.listdir(self.config["path"]["train_results"]):

                    if image == ".DS_Store":
                        pass

                    image_path = os.path.join(
                        self.config["path"]["train_results"], image
                    )
                    self.images.append(imageio.imread(image_path))

                imageio.mimsave(
                    os.path.join(self.config["path"]["train_gif"], "train_results.gif"),
                    self.images,
                    "GIF",
                )
        else:
            raise FileNotFoundError(
                f"{self.config['path']['train_results']} does not exist"
            )

    def plot(self, **kwargs):
        plt.figure(figsize=(10, 10))

        X, y = next(iter(kwargs["dataloader"]))

        predicted_y = self.netG_XtoY(X.to(self.device))
        reconstructed_X = self.netGYtoX(predicted_y.to(self.device))

        for index, image in enumerate(predicted_y):
            fake_y = image.permute(1, 2, 0).cpu().detach().numpy()
            constructed_X = (
                reconstructed_X[index].permute(1, 2, 0).cpu().detach().numpy()
            )

            real_X = X[index].permute(1, 2, 0).cpu().detach().numpy()
            real_y = y[index].permute(1, 2, 0).cpu().detach().numpy()

            fake_y = self.image_normalized(image=fake_y)
            constructed_X = self.image_normalized(image=constructed_X)
            real_X = self.image_normalized(image=real_X)
            real_y = self.image_normalized(image=real_y)

            plt.subplot(2 * 4, 2 * 4, 4 * index + 1)
            plt.imshow(real_X)
            plt.title("X")
            plt.axis("off")

            plt.subplot(2 * 4, 2 * 4, 4 * index + 2)
            plt.imshow(fake_y)
            plt.title("fake_y")
            plt.axis("off")

            plt.subplot(2 * 4, 2 * 4, 4 * index + 3)
            plt.imshow(real_y)
            plt.title("y")
            plt.axis("off")

            plt.subplot(2 * 4, 2 * 4, 4 * index + 4)
            plt.imshow(constructed_X)
            plt.title("revert_X")
            plt.axis("off")

        plt.tight_layout()

        if os.path.exists(self.config["path"]["test_result"]):
            plt.savefig(
                os.path.join(self.config["path"]["test_result"], "test_result.png")
            )

            print(
                "The result is saved in {}".format(self.config["path"]["test_result"])
            )

        else:
            os.makedirs(self.config["path"]["test_result"])
            plt.savefig(
                os.path.join(self.config["path"]["test_result"], "test_result.png")
            )

            print(
                "The result is saved in {}".format(self.config["path"]["test_result"])
            )

        plt.show()

    def test(self):
        dataloader = self.load_dataloader()

        self.netG_XtoY.load_state_dict(self.select_best_model()["netG_XtoY"])
        self.netGYtoX.load_state_dict(self.select_best_model()["netG_YtoX"])

        try:
            self.plot(dataloader=dataloader)

            if self.is_gif:
                self.create_gif()

        except Exception as e:
            print("The exception is {}".format(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model for CycleGAN".title())

    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Number of channels in the input".capitalize(),
    )

    parser.add_argument(
        "--device", type=str, default="mps", help="Define the device".capitalize()
    )

    parser.add_argument(
        "--test_result",
        type=str,
        default="test_result",
        help="Define the path to save the test result".capitalize(),
    )
    parser.add_argument(
        "--gif",
        type=bool,
        default=False,
        help="Create a gif from the test result".capitalize(),
    )

    args = parser.parse_args()

    if args.test_result:

        test_model = TestModel(
            in_channels=args.in_channels, device=args.device, create_gif=args.gif
        )
        test_model.test()

    else:
        raise ValueError(
            f"{args.test_result} is not a valid path to save the test result"
        )
