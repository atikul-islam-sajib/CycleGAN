import joblib
import yaml
import torch
import torch.nn as nn


def dump(value, filename):
    joblib.dump(value=value, filename=filename)


def load(filename):
    return joblib.load(filename=filename)


def params():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        return torch.device("cpu")
