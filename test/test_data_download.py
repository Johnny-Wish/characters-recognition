import os
import unittest
from dataset_download import attemptive_download, unzip


class TestAttemptDownload(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAttemptDownload, self).__init__(*args, **kwargs)
        self.url = "https://github.com/Johnny-Wish/siamese-optimization-for-neural-nets/raw/master/resources/optimization-objective-breakdown.png"
        self.location = "../test.png"
        if os.path.exists(self.location):
            os.remove(self.location)

    def test_attemptive_download_no_duplicate(self):
        self.assertTrue(attemptive_download(self.url, self.location))

    def test_attemptive_download_duplicate(self):
        self.assertTrue(attemptive_download(self.url, self.location, force=True))
        self.assertFalse(attemptive_download(self.url, self.location, force=False))
        self.assertTrue(attemptive_download(self.url, self.location, force=True))
        if os.path.exists(self.location):
            os.remove(self.location)
