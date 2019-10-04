import wget
import os

FILES = {
    # download test video
    "http://files.heuritech.com/raw_files/surfrider/test_video.mp4" : "tests/resources/test_video.mp4"

}

def download_from_drive(urls_to_output):
    for k,v in urls_to_output.items():
        if not os.path.isfile(v):
            wget.download(k, v)
            print("\ndownloaded to", v)

download_from_drive(FILES)
