import pickle as pkl
import json
import time


def dump(obj, dest):
    """
    a trivial tool for dumping object on disk
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


def flush_json_metrics(d: dict, step=None):
    """
    writes json dicts (from a python dict of metrics) to be parsed by FloydHub servers
    :param d: a dict whose keys are metric names, and values are corresponding values
    :param step: global step count, default is None
    :return: None
    """
    for key in d:
        if step is None:
            print(json.dumps({"metric": key, "value": d[key]}))
        else:
            print(json.dumps({"metric": key, "value": d[key], "step": step}))


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
        """
        flushes the metrics to stdout in .json format
        :return: None
        """
        for json in self._jsons:
            print(json)
            if self.time_interval > 0:
                time.sleep(self.time_interval)

    @property
    def jsons(self):
        return self._jsons

    def extend(self, other):
        """
        extend the list of jsons with `other`
        :param other: a finite iterable
        :return: self
        """
        self._jsons.extend(
            [json.dumps({"metric": self.metric, "value": it, "step": idx + len(self)}) for idx, it in enumerate(other)]
        )
        return self

    def __len__(self):
        return len(self._jsons)

    # REVIEW does __iter__ works this way?
    def __iter__(self):
        return self._jsons
