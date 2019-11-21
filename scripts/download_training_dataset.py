import os
from zipfile import ZipFile

import wget

DATASET = "http://files.heuritech.com/raw_files/dataset_surfrider_cleaned.zip"

dataset_name = wget.download(DATASET)

with ZipFile(dataset_name, 'r') as zipObj:
    zipObj.extractall()

os.remove(dataset_name)
