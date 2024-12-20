import os
import gdown
import shutil


def load_data_google_drive(url, target_name):
    """
    Utility function for downloading zipped data from Google Drive.
    
    Args:
        url (str): url to the zipped file on Google Drive.
        target_name (str): name of the folder for the downloaded and extracted data.

    Returns:
        str: directory where the data is downloaded and extracted.
    """
    root = "../data/"

    # download from Google Drive
    path = root + target_name + ".zip"
    if not os.path.isfile(path):
        gdown.download(url=url, output=path, fuzzy=True)

    # unzip files
    dir = root + target_name
    if not os.path.isdir(dir) or not os.listdir(dir):
        shutil.unpack_archive(path, dir)

    return dir
