import unittest
from global_utils import JsonMetricQueueWriter


class TestJsonMetricQueueWriter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestJsonMetricQueueWriter, self).__init__(*args, **kwargs)
        self.writer = JsonMetricQueueWriter("f1-score", [0.1 * n for n in range(10)])

    def test_write(self):
        self.writer.write()