import wget
import os

FILES = {
    # download test video
    "http://files.heuritech.com/raw_files/surfrider/test_video.mp4" : ".mot/tests/test_video.mp4",
    "http://files.heuritech.com/raw_files/surfrider/test_video_folder.zip" : ".mot/tests/test_video_folder.zip"
}

home = os.path.expanduser("~")
if not os.path.isdir(os.path.join(home, ".mot/")):
    os.mkdir(os.path.join(home, ".mot/"))
    os.mkdir(os.path.join(home, ".mot/tests"))

def download_from_drive(urls_to_output):
    for k,v in urls_to_output.items():
        path = os.path.join(home, v)
        if not os.path.isfile(path):
            wget.download(k, path)
            print("\ndownloaded to ", path)

download_from_drive(FILES)
