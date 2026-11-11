"""Use case: where to blur, frame by frame, to anonymise a video before it leaves the server PC."""
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from vtia.application.ports import Detector
from vtia.domain.models import Box
from vtia.domain.privacy import private_regions


def regions_per_frame(source: Path, detector: Detector, hold_frames: int = 3) -> Iterator[list[Box]]:
    """Private regions of each frame, plus those of the previous `hold_frames` frames: a person or
    vehicle the detector misses for a frame or two stays blurred instead of flashing into view."""
    recent: deque[list[Box]] = deque(maxlen=hold_frames + 1)
    for frame in detector.detect(source):
        recent.append(private_regions(frame))
        yield [r for regions in recent for r in regions]
