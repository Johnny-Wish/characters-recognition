import os
import sys
import shutil
import argparse
import zipfile
from os.path import join
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
                        help="URL for zipped dataset, better to use the default")
    parser.add_argument("--location", default=".", help="location to save downloaded dataset")
    opt = parser.parse_args()
    url = opt.url
    zipped = join(opt.location, "dataset.zip")
    extracted = join(opt.location, "dataset")

    print("Downloading dataset")
    if attemptive_download(url, zipped):
        print("Downloading successful")
    else:
        print("Failed to download resource from {} to {}. Try manual downloading instead.".format(url, zipped))
        sys.exit(1)

    print("Extracting dataset")
    if attemptive_unzip(zipped, extracted):
        print("Extraction successful")
    else:
        print("Failed to extract resource from {} to {}. Try manual unzipping instead.".format(zipped, extracted))
        sys.exit(1)

    for file in os.listdir(join(extracted, "matlab")):  # after extraction, files are contained in a `matlab` folder
        shutil.move(join(extracted, "matlab", file), join(extracted, file))
    shutil.rmtree(join(extracted, "matlab"))
