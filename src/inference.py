import sys
import os
import argparse
import zipfile
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.append("src/")

from utils import device_init, params
from generator import Generator


class Inference:
    def __init__(
        self, image_path=None, image=None, XtoY=None, YtoX=None, device="mps", **kwargs
    ):
        self.image_path = image_path
        self.image = image
        self.XtoY = XtoY
        self.YtoX = YtoX
        self.device = device
        self.kwargs = kwargs

        self.device = device_init(device=self.device)
        self.config = params()

        self.netG_XtoY = Generator(in_channels=self.kwargs["in_channels"])
        self.netG_YtoX = Generator(in_channels=self.kwargs["in_channels"])

        self.netG_XtoY.to(self.device)
        self.netG_YtoX.to(self.device)

        self.batch_images = []

    def select_best_model(self):
        if (self.XtoY is None) and (self.YtoX is None):
            if os.path.exists(self.config["path"]["best_model"]):
                load_state_dict = torch.load(
                    self.config["path"]["best_model"], "best_model.pth"
                )

                self.netG_XtoY.load_state_dict(load_state_dict["netG_XtoY"])
                self.netG_YtoX.load_state_dict(load_state_dict["netG_YtoX"])

            else:
                raise FileNotFoundError(
                    f"{self.config['path']['best_model']} does not exist"
                )
        else:
            self.netG_XtoY.load_state_dict(self.XtoY)
            self.netG_YtoX.load_state_dict(self.YtoX)

    def image_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (self.kwargs["image_size"], self.kwargs["image_size"]),
                    Image.BICUBIC,
                ),
                transforms.RandomCrop(
                    (self.kwargs["image_size"], self.kwargs["image_size"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def image_normalized(self, **kwargs):
        return (kwargs["image"] - kwargs["image"].min()) / (
            kwargs["image"].max() - kwargs["image"].min()
        )

    def extract_images(self):
        if os.path.exists(self.config["path"]["batch_results"]):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(path=self.config["path"]["batch_results"])

        else:
            raise FileNotFoundError(
                f"{self.config['path']['batch_results']} does not exist"
            )

        self.directory = os.path.join(self.config["path"]["batch_results"], "dataset")

        for image in os.listdir(self.directory):
            image_path = os.path.join(self.directory, image)
            image = Image.fromarray(cv2.imread(image_path))
            image = self.image_transform()(image)
            self.batch_images.append(image)

        dataloader = DataLoader(dataset=self.batch_images, batch_size=4, shuffle=True)

        return {"dataloader": dataloader}

    def batch_image(self):
        self.dataloader = self.extract_images()["dataloader"]
        plt.figure(figsize=(10, 5))

        for idx, data in enumerate(self.dataloader):
            fake_y = self.netG_XtoY(data.to(self.device))
            reconstructed_X = self.netG_YtoX(fake_y.to(self.device))

            for index, image in enumerate(fake_y):
                image_y = self.image_normalized(image=image)
                revert_X = self.image_normalized(image=reconstructed_X[index])
                real_data = self.image_normalized(image=data[index])

                plt.subplot(3 * 2, 3 * 2, 3 * index + 1)
                plt.imshow(real_data.permute(1, 2, 0).cpu().detach().numpy())
                plt.title("X")
                plt.axis("off")

                plt.subplot(3 * 2, 3 * 2, 3 * index + 2)
                plt.imshow(image_y.permute(1, 2, 0).cpu().detach().numpy())
                plt.title("fake_y")
                plt.axis("off")

                plt.subplot(3 * 2, 3 * 2, 3 * index + 3)
                plt.imshow(revert_X.permute(1, 2, 0).cpu().detach().numpy())
                plt.title("revert_X")
                plt.axis("off")

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    self.config["path"]["batch_results"],
                    "batch_result_{}.png".format(idx),
                )
            )
            plt.show()

    def single_image(self):
        image = Image.fromarray(cv2.imread(self.image))
        image = self.image_transform()(image)

        fake_y = self.netG_XtoY(image.to(self.device))
        reconstructed_X = self.netG_YtoX(fake_y.to(self.device))

        image = self.image_normalized(image=image)
        fake_y = self.image_normalized(image=fake_y)
        reconstructed_X = self.image_normalized(image=reconstructed_X)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image.permute(1, 2, 0).cpu().detach().numpy())
        plt.title("X")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(fake_y.permute(1, 2, 0).cpu().detach().numpy())
        plt.title("Y")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_X.permute(1, 2, 0).cpu().detach().numpy())
        plt.title("revert_X")
        plt.axis("off")

        if os.path.exists(self.config["path"]["single_result"]):

            plt.savefig(
                os.path.join(self.config["path"]["single_result"], "result_image.png")
            )
        else:
            raise Exception("Unable to save the images".capitalize())

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference class for CycleGAN".title())

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the zip file containing the dataset".capitalize(),
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to the image".capitalize()
    )
    parser.add_argument(
        "--XtoY",
        type=str,
        default=None,
        help="Path to the XtoY model".capitalize(),
    )
    parser.add_argument(
        "--YtoX", type=str, default=None, help="Path to the YtoX model".capitalize()
    )

    parser.add_argument("--single_image", action="store_true", default=False)
    parser.add_argument(
        "--device", type=str, default="mps", help="Define the device".capitalize()
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Define the in_channels".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Define the image_size".capitalize()
    )
    parser.add_argument("--batch_image", action="store_true", default=False)

    args = parser.parse_args()

    if args.single_image:
        infer = Inference(
            image=args.image,
            XtoY=args.XtoY,
            YtoX=args.YtoX,
            device=args.device,
            in_channels=args.in_channels,
            image_size=args.image_size,
        )

        infer.single_image()

    elif args.batch_image:
        infer = Inference(
            image_path=args.image_path,
            XtoY=args.XtoY,
            YtoX=args.YtoX,
            device=args.device,
            in_channels=args.in_channels,
            image_size=args.image_size,
        )

        infer.batch_image()
