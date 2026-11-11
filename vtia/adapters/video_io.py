"""Reading videos (or folders of frames) with OpenCV."""
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def video_info(source: Path, default_fps: float = 30.0) -> tuple[float, int]:
    """(frames per second, frame count) as reported by the container."""
    cap = cv2.VideoCapture(str(source))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (fps if fps and fps > 0 else default_fps), frames


class FfmpegWriter:
    """H.264 through the ffmpeg binary bundled by imageio-ffmpeg: plays in browsers (the dashboard)
    and, unlike OpenCV's writers, lets us set the quality, so files stay close to the camera's size."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], crf: int = 23):
        import imageio_ffmpeg

        w, h = size
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 4:2:0 needs even sides
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.path = path

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}: {self.proc.stderr.read().decode(errors='replace')}")


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> FfmpegWriter:
    return FfmpegWriter(Path(path), fps, size)


def read_frames(source: Path) -> Iterator[np.ndarray]:
    """RGB frames from a video file or a folder of images (sorted by name, like Ultralytics)."""
    source = Path(source)
    if source.is_dir():
        for p in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            yield cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
