import os

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


def save_file(file: FileStorage, upload_folder: str):
    filename = secure_filename(file.filename)
    if not filename:
        raise ValueError("You must choose a file before uploading it.")

    full_filepath = os.path.join(upload_folder, filename)
    os.makedirs(upload_folder, exist_ok=True)
    if os.path.isfile(full_filepath):
        os.remove(full_filepath)
    file.save(full_filepath)
    return filename, full_filepath
