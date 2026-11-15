"""Proposes a camera point (counting lines, carriageway and pedestrian zones) from tracked objects.

1. Vehicles that really move are grouped into flows: first by direction, then, within a
   direction, by their position across it (separate lanes or carriageways).
2. Each flow gets a counting line across it, where most of its tracks are visible, oriented so
   that its own direction is `to_right` and allowed (the other one raises a wrong-way alert).
3. Each flow's carriageway zone is the convex hull of where its vehicles touch the ground.
4. Pedestrians are grouped by proximity; each group with a few people becomes a zone.

A proposal to review, not ground truth: rules (forbidden classes, dwell, occupancy) are left for
the user to add.
"""
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from vtia.domain.classes import RoadClass
from vtia.domain.models import Detection, Line, Point, PointConfig, Zone
from vtia.domain.zones import inside

VEHICLES = {RoadClass.CAR, RoadClass.BUS, RoadClass.TRUCK, RoadClass.MOTORCYCLE, RoadClass.BICYCLE}
MIN_TRACKS = 3          # tracks a flow (or a pedestrian group) needs: fewer is noise
MIN_POINTS = 5          # detections a track needs to say where it goes
MIN_MOVE = 2.0          # displacement, in box heights, for a vehicle to count as moving
MIN_STRAIGHT = 0.6      # displacement / path length: jittering boxes in a jam go nowhere
SAME_DIRECTION = 35.0   # degrees: tracks closer than this in heading share a direction
NOISE_SHARE = 0.1       # a flow with fewer tracks than this share of the biggest one is noise
                        # (lane changes, identity swaps)
LINE_MARGIN = 0.75      # line overhang beyond the outermost tracks, in box widths
WALK_CELL = 2.0         # pedestrian grid cell, in person-box heights
ACROSS = 60.0           # degrees from the flow: a person on the road moving more across than
                        # this is crossing it (crosswalk); otherwise it rides with the traffic
MAX_WALK = 2.0          # person-box heights per second: faster "pedestrians" are riders

OPPOSITE = {"toward_camera": "away_from_camera", "away_from_camera": "toward_camera",
            "left_to_right": "right_to_left", "right_to_left": "left_to_right"}


@dataclass
class _Track:
    points: list[Point]      # anchors (bottom-centre), in frame order
    widths: list[float]
    heights: list[float]
    road_class: RoadClass
    frames: tuple[int, int]  # first, last

    @property
    def length(self) -> float:
        """Path length over ~10 spaced points: frame-to-frame box jitter would inflate it."""
        step = max(1, len(self.points) // 10)
        pts = self.points[::step] + ([self.points[-1]] if (len(self.points) - 1) % step else [])
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))

    @property
    def move(self) -> tuple[float, float]:
        return self.points[-1][0] - self.points[0][0], self.points[-1][1] - self.points[0][1]

    @property
    def size(self) -> float:
        return _median(self.heights)


def suggest_point(frames: Iterable[list[Detection]], fps: float = 30.0) -> PointConfig:
    """`frames` must carry track ids (tracker output). Lines are empty when no flow is found."""
    tracks = _tracks(frames)
    moving = [t for t in tracks if t.road_class in VEHICLES and math.hypot(*t.move) >= MIN_MOVE * t.size
              and math.hypot(*t.move) >= MIN_STRAIGHT * t.length]
    flows = [f for d in _by_direction(moving) for f in _by_road(d)]
    biggest = max((len(f) for f in flows), default=0)
    flows = [f for f in flows if len(f) >= max(MIN_TRACKS, NOISE_SHARE * biggest)]
    flows.sort(key=lambda f: _mean([p[0] for t in f for p in t.points]))  # left to right: stable names

    lines, zones, roads, seen = [], [], [], Counter()
    for flow in flows:
        u = _unit(_mean([t.move[0] / math.hypot(*t.move) for t in flow]),
                  _mean([t.move[1] / math.hypot(*t.move) for t in flow]))
        label = _label(u)
        seen[label] += 1
        name = f"{label}_{seen[label]}"
        start, end = _line_across(flow, u)
        lines.append(Line(name, start, end, label, OPPOSITE[label], label))
        corners = [c for t in flow for p, w in zip(t.points, t.widths) for c in ((p[0] - w / 2, p[1]), (p[0] + w / 2, p[1]))]
        zones.append(Zone(f"calzada_{name}", _rounded(_hull(corners))))
        roads.append((zones[-1].polygon, u))
    walkers = [t for t in tracks if t.road_class == RoadClass.PEDESTRIAN
               and math.hypot(*t.move) / max(1, t.frames[1] - t.frames[0]) * fps <= MAX_WALK * t.size
               and not _rides(t, roads)]
    for k, group in enumerate(_walk_groups(walkers), 1):
        zones.append(Zone(f"peatonal_{k}", _rounded(_hull(group))))
    return PointConfig(tuple(lines), tuple(z for z in zones if len(z.polygon) >= 3))


def _rides(t: _Track, roads: list[tuple[tuple[Point, ...], tuple[float, float]]]) -> bool:
    """A "pedestrian" mostly on a carriageway who is not crossing it is a rider the detector boxed
    apart from its motorcycle or bicycle (stopped in traffic, or moving along the flow)."""
    for polygon, u in roads:
        if sum(inside(p, polygon) for p in t.points) < 0.5 * len(t.points):
            continue
        if math.hypot(*t.move) < t.size or _angle(_unit(*t.move), u) < ACROSS or _angle(_unit(*t.move), u) > 180 - ACROSS:
            return True
    return False


def _tracks(frames: Iterable[list[Detection]]) -> list[_Track]:
    dets: dict[int, list[Detection]] = defaultdict(list)
    for frame in frames:
        for d in frame:
            if d.track_id is not None:
                dets[d.track_id].append(d)
    out = []
    for ds in dets.values():
        if len(ds) >= MIN_POINTS:
            ds.sort(key=lambda d: d.frame)
            cls = Counter(d.road_class for d in ds).most_common(1)[0][0]
            out.append(_Track([d.anchor for d in ds], [d.box[2] - d.box[0] for d in ds],
                              [d.box[3] - d.box[1] for d in ds], cls, (ds[0].frame, ds[-1].frame)))
    return out


def _by_direction(tracks: list[_Track]) -> list[list[_Track]]:
    groups: list[tuple[list[float], list[_Track]]] = []  # (summed unit heading, tracks)
    for t in sorted(tracks, key=lambda t: -math.hypot(*t.move)):  # longest first: steadier headings
        h = _unit(*t.move)
        for acc, members in groups:
            if _angle(h, _unit(*acc)) < SAME_DIRECTION:
                acc[0] += h[0]
                acc[1] += h[1]
                members.append(t)
                break
        else:
            groups.append(([h[0], h[1]], [t]))
    return [members for _, members in groups]


def _by_road(tracks: list[_Track]) -> list[list[_Track]]:
    """Splits one direction into separate carriageways: tracks whose ground paths touch (on a grid
    of one box width) share a road. Unlike distances across the heading, this is not fooled by
    perspective, where parallel lanes converge toward the horizon."""
    cell = _median([w for t in tracks for w in t.widths])
    owner: dict[tuple[int, int], int] = {}  # grid cell -> index of a track passing over it
    parent = list(range(len(tracks)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, t in enumerate(tracks):
        for c in _swept_cells(t.points, cell):
            for nb in ((c[0] + dx, c[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                if nb in owner:
                    parent[root(i)] = root(owner[nb])
            owner.setdefault(c, i)
    roads: dict[int, list[_Track]] = defaultdict(list)
    for i, t in enumerate(tracks):
        roads[root(i)].append(t)
    return list(roads.values())


def _swept_cells(points: list[Point], cell: float) -> set[tuple[int, int]]:
    cells = set()
    for a, b in zip(points, points[1:] or points):
        steps = max(1, int(math.dist(a, b) / (cell / 2)))
        for k in range(steps + 1):
            x, y = a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps
            cells.add((int(x // cell), int(y // cell)))
    return cells


def _line_across(flow: list[_Track], u: tuple[float, float]) -> tuple[Point, Point]:
    """Perpendicular to `u`, where the most tracks of the flow are visible (ties: nearest the middle)."""
    n = (-u[1], u[0])
    spans = [(min(_dot(p, u) for p in t.points), max(_dot(p, u) for p in t.points)) for t in flow]
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    mid = (lo + hi) / 2
    best = max((lo + (hi - lo) * k / 40 for k in range(1, 40)),
               key=lambda s: (sum(a <= s <= b for a, b in spans), -abs(s - mid)))
    offsets, widths = [], []
    for t in flow:
        hit = _at(t, u, best)
        if hit is not None:
            offsets.append(_dot(hit[0], n))
            widths.append(hit[1])
    offsets.sort()
    lo_n, hi_n = _percentile(offsets, 0.05), _percentile(offsets, 0.95)
    half = (hi_n - lo_n) / 2 + LINE_MARGIN * _median(widths)
    centre_n = (lo_n + hi_n) / 2
    c = (best * u[0] + centre_n * n[0], best * u[1] + centre_n * n[1])
    # Walking start -> end, the flow (+u) must cross to the right: that is end - start = -n on screen.
    start = (c[0] + n[0] * half, c[1] + n[1] * half)
    end = (c[0] - n[0] * half, c[1] - n[1] * half)
    return (round(start[0]), round(start[1])), (round(end[0]), round(end[1]))


def _at(t: _Track, u: tuple[float, float], s: float) -> tuple[Point, float] | None:
    """Where track `t` is when its position along `u` equals `s` (linear between detections)."""
    for (p, w), (q, v) in zip(zip(t.points, t.widths), zip(t.points[1:], t.widths[1:])):
        a, b = _dot(p, u), _dot(q, u)
        if min(a, b) <= s <= max(a, b):
            k = 0.0 if a == b else (s - a) / (b - a)
            return (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k), w + (v - w) * k
    return None


def _walk_groups(people: list[_Track]) -> list[list[Point]]:
    """Anchors of pedestrians, grouped by grid cells that touch; groups with few people are dropped."""
    if not people:
        return []
    cell = WALK_CELL * _median([t.size for t in people])
    cells: dict[tuple[int, int], list[tuple[int, Point]]] = defaultdict(list)
    for i, t in enumerate(people):
        for p in t.points:
            cells[(int(p[0] // cell), int(p[1] // cell))].append((i, p))
    # Only cells two different people walked over: lone tracks must not chain cells into one blob.
    cells = defaultdict(list, {c: m for c, m in cells.items() if len({i for i, _ in m}) >= 2})
    groups, done = [], set()
    for start in cells:
        if start in done:
            continue
        stack, members = [start], []
        done.add(start)
        while stack:
            cx, cy = stack.pop()
            members.extend(cells[(cx, cy)])
            for nb in ((cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                if nb in cells and nb not in done:
                    done.add(nb)
                    stack.append(nb)
        if len({i for i, _ in members}) >= MIN_TRACKS:
            groups.append([p for _, p in members])
    return groups


def _hull(points: list[Point]) -> list[Point]:
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _label(u: tuple[float, float]) -> str:
    # Perspective: a road leaving the camera climbs the image diagonally, so any clear vertical
    # component (> ~25 degrees from horizontal) names it by depth, not by side.
    if abs(u[1]) >= 0.42:
        return "toward_camera" if u[1] > 0 else "away_from_camera"  # down the image = closer
    return "left_to_right" if u[0] > 0 else "right_to_left"


def _rounded(points: list[Point]) -> tuple[Point, ...]:
    out = []
    for p in points:
        q = (float(round(p[0])), float(round(p[1])))
        if not out or q != out[-1]:
            out.append(q)
    return tuple(out)


def _unit(x: float, y: float) -> tuple[float, float]:
    n = math.hypot(x, y) or 1.0
    return x / n, y / n


def _dot(p: Point, u: tuple[float, float]) -> float:
    return p[0] * u[0] + p[1] * u[1]


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def _percentile(sorted_xs: list[float], q: float) -> float:
    return sorted_xs[min(len(sorted_xs) - 1, int(q * (len(sorted_xs) - 1) + 0.5))]
