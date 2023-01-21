import glob
import os
from pathlib import Path

import cv2

from torchyolo.utils.file_utils import create_dir

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes


class LoadData:
    def __init__(self, path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "**/*.*"), recursive=True))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise FileNotFoundError(f"Invalid path {p}")
        imgp = [i for i in files if i.split(".")[-1] in IMG_FORMATS]
        vidp = [v for v in files if v.split(".")[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = "image"
        if any(vidp):
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None

    @staticmethod
    def checkext(path):
        file_type = "image" if path.split(".")[-1].lower() in IMG_FORMATS else "video"
        return file_type

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.checkext(path) == "video":
            self.type = "video"
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
        return img, path, self.cap

    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def create_video_writer(video_path, output_path, fps=None) -> cv2.VideoWriter:
    """
    This function is used to create video writer.
    Args:
        video_path: video path
        output_path: output path
        fps: fps
    Returns:
        video writer
    """
    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(output_path, fourcc, fps, size)
    return videoWriter


def get_video_formats():
    """
    This function is used to get video formats.
    Returns:
        video formats
    """
    extensions = [".asf", ".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv"]
    video_formats = [x.upper() for x in extensions] + extensions
    return video_formats
