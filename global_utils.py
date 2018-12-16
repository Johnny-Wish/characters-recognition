import pickle as pkl


def dump(obj, dest):
    with open(dest, "wb") as f:
        pkl.dump(obj, f)


def load(src):
    with open(src, "rb") as f:
        obj = pkl.load(f)
    return obj
