import os
from zipfile import ZipFile

import wget

DATASET = "http://files.heuritech.com/raw_files/dataset_surfrider_cleaned.zip"

if not os.path.isdir("dataset_surfrider_cleaned"):
    dataset_name = wget.download(DATASET)
    with ZipFile(dataset_name, 'r') as zipObj:
        zipObj.extractall()
    os.remove(dataset_name)
else:
    print("Dataset already here")
