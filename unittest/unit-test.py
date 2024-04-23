import sys
import os
import unittest
import torch

sys.path.append("src/")

from utils import load, params


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

    def test_data_quantity(self):
        self.assertEqual(sum(image.size(0) for image, _ in self.dataloader), 18)

    def test_train_dataset(self):
        data, _ = next(iter(self.train_dataloader))
        self.assertEqual(data.size(), torch.Size([1, 3, 256, 256]))

    def test_test_dataloader(self):
        data, _ = next(iter(self.test_dataloader))
        self.assertEqual(data.size(), torch.Size([4, 3, 256, 256]))


if __name__ == "__main__":
    unittest.main()
