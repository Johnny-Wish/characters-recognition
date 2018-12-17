import os
import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip",
                        help="URL for downloading")
    parser.add_argument("--location", default="/Users/liushuheng/Desktop/dataset.zip",
                        help="location to save downloaded file")
    opt = parser.parse_args()

    attemptive_download(opt.url, opt.location)
