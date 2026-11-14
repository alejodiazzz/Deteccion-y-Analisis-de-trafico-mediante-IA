"""Ports: what the use cases need from the outside world. Adapters implement them."""
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from vtia.domain.models import Box, Detection


class Detector(Protocol):
    """YOLO, RT-DETR, Faster R-CNN... Yields one list per frame, without track ids."""
    name: str

    def detect(self, source: Path) -> Iterator[list[Detection]]: ...


class PrivateFinder(Protocol):
    """Plate and face models. Given one frame and areas of it (frame pixels), returns for each
    area the plates / faces found inside it, in frame pixels (an empty list when none)."""

    def find(self, image: Any, plate_areas: list[Box], face_areas: list[Box]
             ) -> tuple[list[list[Box]], list[list[Box]]]: ...


class Tracker(Protocol):
    """Takes per-frame detections and yields the same frames with track ids assigned.

    Detections the tracker does not confirm are dropped. Each call starts fresh.
    `frame_rate` is that of the source, so time-based limits mean the same at any fps.
    """

    def track(self, frames: Iterable[list[Detection]], frame_rate: float) -> Iterator[list[Detection]]: ...
