"""Zones of interest: who is inside each one, and for how long."""
from collections.abc import Iterable, Iterator

from vtia.domain.counting import ClassVotes
from vtia.domain.models import Detection, Point, Visit, Zone


def inside(p: Point, polygon: tuple[Point, ...]) -> bool:
    """Ray casting: count polygon edges crossed by a ray going right from p."""
    x, y = p
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


class ZoneMonitor:
    """Fed frame by frame with tracked detections; an object is inside a zone when its anchor is.

    A visit stays open while the track keeps being seen inside; it closes after
    `gap_frames` without seeing it there (it left, or the tracker lost it briefly).
    """

    def __init__(self, zones: Iterable[Zone], gap_frames: int):
        self.zones = tuple(zones)
        self.gap = gap_frames
        self._open: dict[tuple[str, int], list[int]] = {}  # (zone, track) -> [enter frame, last frame seen]
        self._seen_outside: set[tuple[str, int]] = set()   # open visits whose track was last seen out of the zone
        self._votes = ClassVotes()
        self.visits: list[Visit] = []

    def update(self, frame: int, tracked: Iterable[Detection]) -> None:
        for d in tracked:
            self._votes.add(d)
            for z in self.zones:
                key = (z.name, d.track_id)
                if inside(d.anchor, z.polygon):
                    self._open.setdefault(key, [frame, frame])[1] = frame
                    self._seen_outside.discard(key)
                elif key in self._open:
                    self._seen_outside.add(key)
        for key, (_, last) in list(self._open.items()):
            if frame - last > self.gap:
                self._close(key)

    def open_visits(self, zone: str) -> Iterator[tuple[int, int, int]]:
        """(track id, enter frame, last frame seen) of everyone currently inside `zone`."""
        for (z, track), (enter, last) in self._open.items():
            if z == zone:
                yield track, enter, last

    def occupancy(self, zone: str) -> int:
        """Objects in `zone` now. One the tracker lost for a moment still counts; one seen outside does not."""
        return sum((zone, track) not in self._seen_outside for track, _, _ in self.open_visits(zone))

    def track_class(self, track_id: int):
        return self._votes[track_id]

    def finish(self) -> list[Visit]:
        """Closes every open visit (end of video) and returns all visits."""
        for key in list(self._open):
            self._close(key)
        return self.visits

    def _close(self, key: tuple[str, int]) -> None:
        zone, track = key
        enter, last = self._open.pop(key)
        self._seen_outside.discard(key)
        self.visits.append(Visit(track, self._votes[track], zone, enter, last))
