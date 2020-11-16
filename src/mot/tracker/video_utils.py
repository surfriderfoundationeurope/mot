import os

import ffmpeg


def split_video(input_path, output_folder, fps=1.5, resolution=(1024, 768)):
    """Splits a video into frames

    Arguments:

    - *input_path*: string of video full path
    - *output_folder*: folder to store images
    - *fps*: float for number of frames per second
    - *resolution*: integer tuple for resolution

    """
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    (
        ffmpeg.input(input_path).filter(
            "scale", width="{}".format(resolution[0]), height="{}".format(resolution[1])
        ).filter("fps", fps=fps, round="up").trim(
            start_frame=0
        ).output(os.path.join(output_folder, "frame_%4d.jpeg"), format="image2",
                 vcodec="mjpeg").run()
    )


def read_folder(input_path):
    # for now, read directly from images in folder ; later from json outputs
    return [os.path.join(input_path, file) for file in sorted(os.listdir(input_path))]
