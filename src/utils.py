import joblib
import yaml


def dump(value, filename):
    joblib.dump(value=value, filename=filename)


def load(filename):
    return joblib.load(filename=filename)


def params():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
