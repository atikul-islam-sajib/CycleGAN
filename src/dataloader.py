import sys
import os
import argparse
import cv2
from PIL import Image
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("src/")

from utils import dump, load, params


class Loader:
    def __init__(self, image_path=None, image_size=256, batch_size=1, split_size=0.25):
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.X = []
        self.y = []

        self.config = params()

    def image_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop((self.image_size, self.image_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def image_splits(self, **kwargs):
        return train_test_split(
            kwargs["X"], kwargs["y"], test_size=self.split_size, random_state=42
        )

    def unzip_folder(self):
        if os.path.exists(self.config["path"]["raw_path"]):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(path=self.config["path"]["raw_path"])
        else:
            raise Exception("Unable to find the zip file".capitalize())

    def extract_features(self):
        if os.path.exists(self.config["path"]["raw_path"]):
            self.directory = os.path.join(self.config["path"]["raw_path"], "dataset")
            self.images = os.path.join(self.directory, os.listdir(self.directory)[0])
            self.masks = os.path.join(self.directory, os.listdir(self.directory)[1])

            for image in os.listdir(self.images):
                for mask in os.listdir(self.masks):
                    image_base_name = image.split(".")[0]
                    masks_base_name = mask.split(".")[0]

                    if image_base_name == masks_base_name:
                        self.X.append(
                            self.image_transforms()(
                                Image.fromarray(
                                    cv2.imread(os.path.join(self.images, image))
                                )
                            )
                        )
                        self.y.append(
                            self.image_transforms()(
                                Image.fromarray(
                                    cv2.imread(os.path.join(self.masks, mask))
                                )
                            )
                        )

            X_train, X_test, y_train, y_test = self.image_splits(X=self.X, y=self.y)

            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "X": self.X,
                "y": self.y,
            }

        else:
            raise Exception("Unable to find the zip file".capitalize())

    def create_dataloader(self):
        dataset = self.extract_features()

        if os.path.exists(self.config["path"]["processed_path"]):
            dataloader = DataLoader(
                dataset=list(zip(dataset["X"], dataset["y"])),
                batch_size=self.batch_size * 16,
                shuffle=True,
            )

            train_dataloader = DataLoader(
                dataset=list(zip(dataset["X_train"], dataset["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            test_dataloader = DataLoader(
                dataset=list(zip(dataset["X_test"], dataset["y_test"])),
                batch_size=self.batch_size * 4,
                shuffle=True,
            )

            dump(
                value=dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "dataloader.pkl"
                ),
            )
            dump(
                value=train_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "train_dataloader.pkl"
                ),
            )
            dump(
                value=test_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "test_dataloader.pkl"
                ),
            )

        else:
            raise Exception("Unable to create the dataloader file".capitalize())

    @staticmethod
    def plot_images():
        config = params()

        if os.path.exists(config["path"]["processed_path"]):
            test_dataloader = load(
                os.path.join(config["path"]["processed_path"], "dataloader.pkl")
            )

            data, label = next(iter(test_dataloader))

            plt.figure(figsize=(25, 15))

            for index, image in enumerate(data):
                X = image.permute(1, 2, 0).numpy()
                y = label[index].permute(1, 2, 0).numpy()

                X = (X - X.min()) / (X.max() - X.min())
                y = (y - y.min()) / (y.max() - y.min())

                plt.subplot(2 * 4, 2 * 4, 2 * index + 1)
                plt.imshow(X)
                plt.title("X")
                plt.axis("off")

                plt.subplot(2 * 4, 2 * 4, 2 * index + 2)
                plt.imshow(y, cmap="gray")
                plt.title("y")
                plt.axis("off")

            plt.savefig(os.path.join(config["path"]["files_path"], "images.png"))

            plt.tight_layout()
            plt.show()

        else:
            raise Exception("Unable to open the dataloader file".capitalize())

    @staticmethod
    def dataset_details():
        config = params()

        if os.path.exists(config["path"]["processed_path"]):
            dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "dataloader.pkl"
                )
            )
            train_dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "train_dataloader.pkl"
                )
            )
            test_dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "test_dataloader.pkl"
                )
            )

            train_image, _ = next(iter(train_dataloader))
            test_image, _ = next(iter(test_dataloader))

            pd.DataFrame(
                {
                    "total_images": str(sum(image.size(0) for image, _ in dataloader)),
                    "train_data": str(
                        sum(image.size(0) for image, _ in train_dataloader)
                    ),
                    "test_data": str(
                        sum(image.size(0) for image, _ in test_dataloader)
                    ),
                    "train_data_shape": str(train_image.size()),
                    "test_data_shape": str(test_image.size()),
                },
                index=["Quantity"],
            ).T.to_csv(
                os.path.join(config["path"]["files_path"], "dataset_details.csv")
            )

        else:
            raise Exception("Unable to find the dataloader file".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataloader class for CycleGAN".title()
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the zip file containing the dataset".capitalize(),
    ),
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of the image".capitalize(),
    ),
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Size of the batch".capitalize(),
    ),
    parser.add_argument(
        "--split_size",
        type=float,
        default=0.25,
        help="Size of the batch".capitalize(),
    ),

    args = parser.parse_args()

    if args.image_path:
        try:
            loader = Loader(
                image_path=args.image_path,
                image_size=args.image_size,
                batch_size=args.batch_size,
                split_size=args.split_size,
            )
        except AttributeError as e:
            print("The exception is: {}".format(e))

        else:
            loader.unzip_folder()
            loader.create_dataloader()

        finally:
            Loader.plot_images()
            Loader.dataset_details()

    else:
        raise Exception("Unable to find the zip file".capitalize())
