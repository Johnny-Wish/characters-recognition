import unittest
from global_utils import JsonMetricQueueWriter as Writer


class TestJsonMetricQueueWriter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestJsonMetricQueueWriter, self).__init__(*args, **kwargs)
        self.writer = Writer("f1-score", [0.1 * n for n in range(10)])

    def test_write(self):
        self.writer.write()

    def test_extend_list(self):
        self.writer.extend([0.15, 0.17, 0.33])
        self.writer.write()
        self.writer.extend([0.235, 0.5437, 0.323233])
        self.writer.write()
