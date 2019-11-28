import ffmpeg
import os
import cv2

def split_video(input_path, output_folder, fps = 1.5, resolution=(1024,768)):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    (
        ffmpeg
        .input(input_path)
        .filter('fps', fps=fps, round='up')
        .trim(start_frame=0)
        .output(output_folder + 'frame_%4d.jpeg', format='image2', vcodec='mjpeg')
        .run()
    )
# vf='fps={}'.format(fps), format='image2', vcodec='mjpeg')
def read_folder(input_path):
    # for now, read directly from images in folder ; later from json outputs
    return [os.path.join(input_path, file) for file in  sorted(os.listdir(input_path))]

def open_images(list_path_images):
    return [cv2.cvtColor(cv2.imread(im_path),cv2.COLOR_BGR2GRAY)
           for im_path in list_path_images]
