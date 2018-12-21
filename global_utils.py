import pickle as pkl
import json
import time


def dump(obj, dest):
    """
    a trvial tool for dumping object on disk
    :param obj: object to be dumped
    :param dest: dump path
    :return: None
    """
    with open(dest, "wb") as f:
        pkl.dump(obj, f)


def load(src):
    """
    a trivial tool for loading object on disk
    :param src: path to dumped object
    :return: an object
    """
    with open(src, "rb") as f:
        obj = pkl.load(f)
    return obj


class JsonMetricQueueWriter:
    def __init__(self, metric, itr, time_interval=0):
        """
        a writer for metrics that flushes metrics, along with steps, in json format
        :param metric: str, name of the metric
        :param itr: a finite iterable of metric values
        :param time_interval: time interval between two flushes, in secs
        """
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
