import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.utils import save_image
import numpy as np

sys.path.append("src/")

from utils import device_init, params, dump, load
from helper import helper


class Trainer:

    def __init__(
        self,
        in_channels=3,
        epochs=1000,
        lr=0.0002,
        device="mps",
        adam=True,
        SGD=False,
        lr_scheduler=False,
        is_display=True,
    ):
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.lr_scheduler = lr_scheduler
        self.is_display = is_display

        self.total_D_X_loss = []
        self.total_D_y_loss = []
        self.total_G_losses = []

        self.device = device_init(device=self.device)
        self.config = params()
        self.loss = float("inf")

        init = helper(
            in_channels=self.in_channels,
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
            lr_scheduler=self.lr_scheduler,
        )

        self.netG_XtoY = init["netG_XtoY"].to(self.device)
        self.netG_YtoX = init["netG_YtoX"].to(self.device)

        self.netD_X = init["netD_X"].to(self.device)
        self.netD_Y = init["netD_Y"].to(self.device)

        self.optimizer_G = init["optimizer_G"]
        self.optimizer_D_X = init["optimizer_D_X"]
        self.optimizer_D_Y = init["optimizer_D_Y"]

        self.train_dataloader = init["train_dataloader"]
        self.test_dataloader = init["test_dataloader"]

        self.adversarial_loss = init["adversarial_loss"]
        self.cycle_consistency_loss = init["cycle_consistency_loss"]

    def l1(self, model):
        if model is not None:
            return self.config["model"]["reg_lambda"] * sum(
                torch.norm(params, 1) for params in model.parameters()
            )
        else:
            raise ValueError("Model is not defined")

    def l2(self, model):
        if model is not None:
            return self.config["model"]["reg_lambda"] * sum(
                torch.norm(params, 2) for params in model.parameters()
            )

    def elastic_net(self, model):
        if model is not None:
            l1 = self.l1(model=model)
            l2 = self.l2(model=model)

            return self.config["model"]["reg_lambda"] * (l1 + l2)

    def saved_checkpoints_netG_XtoY(self, **kwargs):
        if os.path.exists(self.config["path"]["train_netG_XtoY"]):

            torch.save(
                self.netG_XtoY.state_dict(),
                os.path.join(
                    self.config["path"]["train_netG_XtoY"],
                    "netG_XtoY_{}.pth".format(kwargs["epoch"]),
                ),
            )
        else:
            raise FileNotFoundError(
                "Path is not found to save checkpoints(netG_XtoY)".capitalize()
            )

    def saved_checkpoints_netG_YtoX(self, **kwargs):
        if os.path.exists(self.config["path"]["train_netG_YtoX"]):

            torch.save(
                self.netG_YtoX.state_dict(),
                os.path.join(
                    self.config["path"]["train_netG_YtoX"],
                    "netG_YtoX_{}.pth".format(kwargs["epoch"]),
                ),
            )
        else:
            raise FileNotFoundError(
                "Path is not found to save checkpoints(netG_YtoX)".capitalize()
            )

    def update_generator(self, **kwargs):
        self.optimizer_G.zero_grad()

        predicted_fake_y = self.netD_Y(kwargs["fake_y"])
        loss_fake_y = self.adversarial_loss(
            predicted_fake_y, torch.ones_like(predicted_fake_y)
        )
        reconstructed_x = self.netG_YtoX(kwargs["fake_y"])
        loss_cycle_x = self.cycle_consistency_loss(reconstructed_x, kwargs["real_x"])

        predicted_fake_x = self.netD_X(kwargs["fake_x"])
        loss_fake_x = self.adversarial_loss(
            predicted_fake_x, torch.ones_like(predicted_fake_x)
        )
        reconstructed_y = self.netG_XtoY(kwargs["fake_x"])
        loss_cycle_y = self.cycle_consistency_loss(reconstructed_y, kwargs["real_y"])

        total_G_loss = loss_fake_y + loss_fake_x + 10 * loss_cycle_x + 10 * loss_cycle_y

        total_G_loss.backward(retain_graph=True)
        self.optimizer_G.step()

        return total_G_loss.item()

    def update_netD_X(self, **kwargs):
        self.optimizer_D_X.zero_grad()

        predicted_real_x = self.netD_X(kwargs["real_x"])
        predicted_fake_x = self.netD_X(kwargs["fake_x"])

        loss_real_x = self.adversarial_loss(
            predicted_real_x, torch.ones_like(predicted_real_x)
        )
        loss_fake_x = self.adversarial_loss(
            predicted_fake_x, torch.zeros_like(predicted_fake_x)
        )

        total_D_X_loss = (loss_real_x + loss_fake_x) * 0.5

        total_D_X_loss.backward(retain_graph=True)
        self.optimizer_D_X.step()

        return total_D_X_loss.item()

    def update_netD_Y(self, **kwargs):
        self.optimizer_D_Y.zero_grad()

        predicted_real_y = self.netD_Y(kwargs["real_y"])
        predicted_fake_y = self.netD_Y(kwargs["fake_y"])

        loss_real_y = self.adversarial_loss(
            predicted_real_y, torch.ones_like(predicted_real_y)
        )
        loss_fake_y = self.adversarial_loss(
            predicted_fake_y, torch.zeros_like(predicted_fake_y)
        )

        total_D_Y_loss = (loss_real_y + loss_fake_y) * 0.5

        total_D_Y_loss.backward(retain_graph=True)
        self.optimizer_D_Y.step()

        return total_D_Y_loss.item()

    def show_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs - [{}/{}] - G_loss: {:.4f} - D_X_loss: {:.4f} - D_Y_loss: {:.4f}".format(
                    kwargs["epoch"],
                    np.mean(kwargs["epochs"]),
                    np.mean(kwargs["G_losses"]),
                    np.mean(kwargs["D_X_loss"]),
                    np.mean(kwargs["D_y_loss"]),
                )
            )
        else:
            print(
                "Epochs - [{}/{}] is completed".format(
                    kwargs["epoch"], kwargs["epochs"]
                )
            )

    def saved_trained_best_model(self, **kwargs):
        if os.path.exists(self.config["path"]["best_model"]):
            if self.loss > kwargs["loss"]:
                self.loss = kwargs["loss"]
                torch.save(
                    {
                        "netG_XtoY": self.netG_XtoY.state_dict(),
                        "netG_YtoX": self.netG_YtoX.state_dict(),
                        "loss": kwargs["loss"],
                        "epoch": kwargs["epoch"],
                    },
                    os.path.join(self.config["path"]["best_model"], "best_model.pth"),
                )
        else:
            raise FileNotFoundError(
                "Path is not found to save checkpoints(best_model)".capitalize()
            )

    def saved_trained_images(self, **kwargs):
        data, label = next(iter(self.test_dataloader))

        XtoY = self.netG_XtoY(data.to(self.device))
        YtoX = self.netG_YtoX(label.to(self.device))

        save_image(
            XtoY,
            os.path.join(
                self.config["path"]["train_results"],
                "XtoY_{}.png".format(kwargs["epoch"] + 1),
            ),
            nrow=2,
            normalize=True,
        )

        save_image(
            YtoX,
            os.path.join(
                self.config["path"]["train_results"],
                "YtoX_{}.png".format(kwargs["epoch"] + 1),
            ),
            nrow=2,
            normalize=True,
        )

    def saved_model_history(self, **kwargs):
        if os.path.exists(self.config["path"]["model_history"]):

            pd.DataFrame(
                {
                    "G_loss": kwargs["G_losses"],
                    "D_X_loss": kwargs["D_X_loss"],
                    "D_Y_loss": kwargs["D_Y_loss"],
                }
            ).to_csv(
                os.path.join(self.config["path"]["model_history"], "model_history.csv"),
                index=True,
            )

    def train(self):
        warnings.filterwarnings("ignore")

        for epoch in tqdm(range(self.epochs)):
            D_X_loss = []
            D_y_loss = []
            G_losses = []

            for _, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                fake_y = self.netG_XtoY(X)
                fake_x = self.netG_YtoX(y)

                D_y_loss.append(self.update_netD_Y(fake_y=fake_y, real_y=y))
                D_X_loss.append(self.update_netD_X(fake_x=fake_x, real_x=X))

                G_losses.append(
                    self.update_generator(
                        fake_x=fake_x, fake_y=fake_y, real_x=X, real_y=y
                    )
                )

            try:
                self.show_progress(
                    epoch=epoch + 1,
                    epochs=self.epochs,
                    D_X_loss=D_X_loss,
                    D_y_loss=D_y_loss,
                    G_losses=G_losses,
                )

            except Exception as e:
                print("The exception is {}".format(e))

            else:
                self.saved_checkpoints_netG_XtoY(epoch=epoch + 1)
                self.saved_checkpoints_netG_YtoX(epoch=epoch + 1)
                self.saved_trained_best_model(epoch=epoch + 1, loss=np.mean(G_losses))

                if epoch % 100:

                    self.saved_trained_images(epoch=epoch + 1)

                self.total_G_losses.append(np.mean(G_losses))
                self.total_D_X_loss.append(np.mean(D_X_loss))
                self.total_D_y_loss.append(np.mean(D_y_loss))

            finally:
                pass

        if os.path.exists(self.config["path"]["train_metrics"]):

            dump(
                value=self.total_G_losses,
                filename=os.path.join(
                    self.config["path"]["train_metrics"], "G_losses.pkl"
                ),
            )
            dump(
                value=self.total_D_X_loss,
                filename=os.path.join(
                    self.config["path"]["train_metrics"], "D_X_loss.pkl"
                ),
            )
            dump(
                value=self.total_D_y_loss,
                filename=os.path.join(
                    self.config["path"]["train_metrics"], "D_y_loss.pkl"
                ),
            )

            try:
                self.saved_model_history(
                    G_losses=self.total_G_losses,
                    D_X_loss=self.total_D_X_loss,
                    D_Y_loss=self.total_D_y_loss,
                )

            except Exception as e:
                print("The exception is {}".format(e))
        else:
            os.mkdir(self.config["path"]["train_metrics"])

    @staticmethod
    def plot_history():
        config = params()
        G_losses = load(
            filename=os.path.join(config["path"]["train_metrics"], "G_losses.pkl")
        )
        D_X_loss = load(
            filename=os.path.join(config["path"]["train_metrics"], "D_X_loss.pkl")
        )
        D_y_loss = load(
            filename=os.path.join(config["path"]["train_metrics"], "D_y_loss.pkl")
        )

        if os.path.exists(config["path"]["train_metrics"]):
            plt.figure(figsize=(10, 5))

            for index, loss in enumerate([G_losses, D_X_loss, D_y_loss]):
                plt.subplot(1, 3, index + 1)
                plt.plot(
                    loss,
                    label=(
                        "G_loss"
                        if index == 0
                        else "D_X_loss" if index == 1 else "D_y_loss"
                    ),
                )
                plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel("Loss")

            plt.tight_layout()

            if os.path.exists(config["path"]["files_path"]):
                plt.savefig(
                    os.path.join(config["path"]["files_path"], "train_metrics.png")
                )
            else:
                raise FileNotFoundError("Files path is not found".capitalize())

            plt.show()

        else:
            raise FileNotFoundError(
                "Path is not found to save checkpoints(train_metrics)".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the training for CycleGAN".title()
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
        "--train", action="store_true", help="Train the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
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

    else:
        raise ValueError("Provide the appropriate argument".capitalize())
