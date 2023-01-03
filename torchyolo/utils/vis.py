import cv2
import numpy as np


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


def tracker_vis(
    track_id,
    label,
    frame,
    tracker_box,
) -> np.ndarray:
    x, y, w, h = int(tracker_box[0]), int(tracker_box[1]), int(tracker_box[2]), int(tracker_box[3])
    MIN_FONT_SCALE = 0.7
    colors = Colors()  # create instance for 'from yolov5.utils.plots import colors'
    color = colors(track_id % 10)
    txt_color = (0, 0, 0) if np.mean(color) > 0.5 else (255, 255, 255)
    font_scale = max(MIN_FONT_SCALE, 0.3 * (w + h) / 600)
    thickness = 2
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    cv2.rectangle(frame, (x, y), (w, h), color, thickness)  # object box
    cv2.rectangle(frame, (x, y - txt_size[1]), (x + txt_size[0], y), color, -1)  # object label box
    cv2.putText(
        frame,
        label,
        (x, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        txt_color,
        thickness,
        cv2.LINE_AA,
    )


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
    from pathlib import Path

    from file_utils import create_dir

    save_dir = create_dir(output_path)
    save_path = str(Path(save_dir) / Path(video_path).name)
    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)
    return videoWriter
