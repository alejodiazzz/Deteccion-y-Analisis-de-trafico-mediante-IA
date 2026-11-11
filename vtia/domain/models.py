"""Domain entities. Pure data: no I/O, no third-party libraries."""
from dataclasses import dataclass
from datetime import time

from vtia.domain.classes import RoadClass

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


@dataclass(frozen=True)
class Detection:
    """One object found in one frame. `track_id` is None until a tracker assigns it."""
    frame: int
    road_class: RoadClass
    confidence: float
    box: Box
    track_id: int | None = None

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box: where the object touches the ground."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


@dataclass(frozen=True)
class GroundTruth:
    """One annotated object in one frame; `object_id` is stable across frames."""
    frame: int
    object_id: int
    road_class: RoadClass
    box: Box


@dataclass(frozen=True)
class Line:
    """Virtual counting line from `start` to `end`, in pixels.

    Walking from start to end as seen on screen, an object that crosses to the
    right-hand side is labelled `to_right`, otherwise `to_left`. Example: for a
    horizontal line drawn left to right, `to_right` means moving down the image.
    """
    name: str
    start: Point
    end: Point
    to_right: str
    to_left: str
    allowed: str | None = None  # if set, crossing in the other direction raises a wrong-way alert


@dataclass(frozen=True)
class Zone:
    """Region of interest (polygon in pixels) with optional alert rules."""
    name: str
    polygon: tuple[Point, ...]
    forbidden: frozenset[RoadClass] = frozenset()  # classes that raise an alert when inside
    max_dwell_s: float | None = None               # staying longer raises an alert
    max_occupancy: int | None = None               # this many objects at once raises an alert


@dataclass(frozen=True)
class Restriction:
    """A class that must not circulate during some local-time windows (e.g. trucks at rush hour)."""
    road_class: RoadClass
    windows: tuple[tuple[time, time], ...]  # [start, end); a window may cross midnight


@dataclass(frozen=True)
class PointConfig:
    """Everything configured for one camera point."""
    lines: tuple[Line, ...]
    zones: tuple[Zone, ...] = ()
    restrictions: tuple[Restriction, ...] = ()


@dataclass(frozen=True)
class Visit:
    """A track's stay inside a zone, in frames (both ends included)."""
    track_id: int
    road_class: RoadClass
    zone: str
    enter_frame: int
    exit_frame: int


@dataclass(frozen=True)
class Alert:
    frame: int
    kind: str  # wrong_way | restricted_hours | forbidden_in_zone | long_dwell | crowding
    where: str  # line or zone name
    track_id: int | None
    road_class: RoadClass | None
    detail: str


@dataclass(frozen=True)
class CrossingEvent:
    frame: int
    track_id: int
    road_class: RoadClass
    line: str
    direction: str
