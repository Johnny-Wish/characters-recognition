import os
import argparse
import zipfile
from urllib.request import urlretrieve


def attemptive_download(url, location, force=False):
    """
    download from `url` to `location`, overwrite existent file if `force`
    suggested to run in a different thread
    :param url: URL for downloading
    :param location: location to save downloaded file
    :param force: whether to overwrite exitent file
    :return: bool, state of successful download
    """
    if os.path.exists(location) and not force:
        return False

    urlretrieve(url, location)
    return True


def attemptive_unzip(zipped_path, extracted_path, force=False):
    """
    unzip a file from `zipped_path` to `extracted_path`
    :param zipped_path: path of the zipped file
    :param extracted_path: path for extracted file/folder
    :return: bool, state of successful zipping
    """
    if os.path.exists(extracted_path) and not force:
        return False
    with zipfile.ZipFile(zipped_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip",
                        help="URL for downloading")
    parser.add_argument("--location", default="/Users/liushuheng/Desktop/dataset.zip",
                        help="location to save downloaded file")
    opt = parser.parse_args()

    attemptive_download(opt.url, opt.location)
