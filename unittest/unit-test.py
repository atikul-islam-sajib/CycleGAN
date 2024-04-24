import sys
import os
import unittest
import torch

sys.path.append("src/")

from utils import load, params
from generator import Generator
from discriminator import Discriminator


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.dataloader = load(
            os.path.join(params()["path"]["processed_path"], "dataloader.pkl")
        )
        self.train_dataloader = load(
            os.path.join(params()["path"]["processed_path"], "train_dataloader.pkl")
        )
        self.test_dataloader = load(
            os.path.join(params()["path"]["processed_path"], "test_dataloader.pkl")
        )
        self.netG = Generator(in_channels=3)
        self.netD = Discriminator(in_channels=3)

    def test_data_quantity(self):
        self.assertEqual(sum(image.size(0) for image, _ in self.dataloader), 18)

    def test_train_dataset(self):
        data, _ = next(iter(self.train_dataloader))
        self.assertEqual(data.size(), torch.Size([1, 3, 256, 256]))

    def test_test_dataloader(self):
        data, _ = next(iter(self.test_dataloader))
        self.assertEqual(data.size(), torch.Size([4, 3, 256, 256]))

    def test_netG_output(self):
        self.assertEqual(
            self.netG(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 3, 256, 256])
        )

    def test_netD_output(self):
        self.assertEqual(
            self.netD(torch.randn(1, 3, 256, 256)).size(), torch.Size([1, 1, 30, 30])
        )


if __name__ == "__main__":
    unittest.main()
