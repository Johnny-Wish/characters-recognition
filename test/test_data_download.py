import os
import shutil
import unittest
from dataset_download import attemptive_download, attemptive_unzip


class TestDatasetDownload(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDatasetDownload, self).__init__(*args, **kwargs)
        self.url = "https://github.com/Johnny-Wish/url-test-site/raw/master/test.zip"
        self.location = "../test1.zip"
        self.extraction = "../test2"
        if os.path.exists(self.location):
            os.remove(self.location)
        if os.path.exists(self.extraction):
            shutil.rmtree(self.extraction)

    def test_attemptive_download(self):
        self.assertTrue(attemptive_download(self.url, self.location))
        self.assertFalse(attemptive_download(self.url, self.location, force=False))
        self.assertTrue(attemptive_download(self.url, self.location, force=True))
        if os.path.exists(self.location):
            os.remove(self.location)

    def test_attemptive_unzip(self):
        attemptive_download(self.url, self.location)
        self.assertTrue(attemptive_unzip(self.location, self.extraction))
        self.assertFalse(attemptive_unzip(self.location, self.extraction, force=False))
        self.assertTrue(attemptive_unzip(self.location, self.extraction, force=True))
        if os.path.exists(self.location):
            os.remove(self.location)
        if os.path.exists(self.location):
            shutil.rmtree(self.location)
