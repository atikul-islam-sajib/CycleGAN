import joblib


def dump(value, filename):
    joblib.dump(value=value, filename=filename)


def load(filename):
    return joblib.load(filename=filename)
