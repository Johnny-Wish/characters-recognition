import pickle as pkl
import json
import time


def dump(obj, dest):
    with open(dest, "wb") as f:
        pkl.dump(obj, f)


def load(src):
    with open(src, "rb") as f:
        obj = pkl.load(f)
    return obj


class JsonMetricQueueWriter:
    def __init__(self, metric, itr, time_interval=0):
        self.metric = metric
        self._jsons = [json.dumps({"metric": self.metric, "value": it, "step": index}) for index, it in enumerate(itr)]
        self.time_interval = time_interval

    def write(self, ):
        for json in self._jsons:
            print(json)
            if self.time_interval > 0:
                time.sleep(self.time_interval)

    @property
    def jsons(self):
        return self._jsons

    def extend(self, other):
        self._jsons.extend(
            [json.dumps({"metric": self.metric, "value": it, "step": idx+len(self)}) for idx, it in enumerate(other)]
        )

    def __len__(self):
        return len(self._jsons)

    def __iter__(self):
        return self._jsons
