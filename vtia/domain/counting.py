"""Line-crossing counting: turns per-frame tracked detections into events."""
from collections import Counter, defaultdict
from collections.abc import Iterable
from math import dist

from vtia.domain.models import CrossingEvent, Detection, Line, Point


class ClassVotes:
    """Majority class of each track so far: absorbs frame-to-frame flicker (car/truck) from the detector."""

    def __init__(self):
        self._votes: defaultdict[int, Counter] = defaultdict(Counter)

    def add(self, d: Detection) -> None:
        self._votes[d.track_id][d.road_class] += 1

    def __getitem__(self, track_id: int):
        return self._votes[track_id].most_common(1)[0][0]


def _side(a: Point, b: Point, p: Point) -> float:
    """Cross product: > 0 if p is right of a->b on screen (y grows down), < 0 left, 0 on it."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class LineCounter:
    """Stateful counter fed frame by frame.

    A crossing needs the anchor to change side between two positions clear of the line,
    with the movement passing through the segment (not its extension). "Clear" means outside
    a dead band of `min_offset` x box height on each side: a stopped object's box jitters, and
    without the band that jitter alone would count as a crossing.
    The class of an event is the track's majority class so far (ClassVotes).
    """

    def __init__(self, lines: list[Line], min_offset: float = 0.1):
        self.lines = lines
        self.min_offset = min_offset
        self._last: dict[tuple[int, str], tuple[bool, Point]] = {}  # (track, line) -> (right side?, anchor)
        self._counted: set[tuple[int, str]] = set()
        self._votes = ClassVotes()

    def update(self, detections: Iterable[Detection]) -> list[CrossingEvent]:
        """`detections` must already carry track ids (output of a tracker)."""
        events = []
        for d in detections:
            self._votes.add(d)
            p = d.anchor
            band = self.min_offset * (d.box[3] - d.box[1])
            for line in self.lines:
                s = _side(line.start, line.end, p)
                if abs(s) <= band * dist(line.start, line.end):  # |s| = distance to the line x its length
                    continue  # within the dead band: wait until it clears it to decide
                key = (d.track_id, line.name)
                prev = self._last.get(key)
                self._last[key] = (s > 0, p)
                if prev is None or prev[0] == (s > 0) or key in self._counted:
                    continue
                # Endpoints of the line on opposite sides of the movement -> crossed within the segment.
                if _side(prev[1], p, line.start) * _side(prev[1], p, line.end) > 0:
                    continue
                # ponytail: one count per track per line, so jitter on the line doesn't double count;
                # an object that genuinely crosses back is missed. Revisit if dwell/return trips matter.
                self._counted.add(key)
                road_class = self._votes[d.track_id]
                direction = line.to_right if s > 0 else line.to_left
                events.append(CrossingEvent(d.frame, d.track_id, road_class, line.name, direction))
        return events


def count_by(events: Iterable[CrossingEvent]) -> Counter:
    """Totals keyed by (line, direction, class)."""
    return Counter((e.line, e.direction, e.road_class) for e in events)
