"""Rule-based alerts, evaluated frame by frame. Each one fires once per object (or per episode)."""
from collections.abc import Iterable
from datetime import datetime, time

from vtia.domain.models import Alert, CrossingEvent, PointConfig
from vtia.domain.timeline import clock
from vtia.domain.zones import ZoneMonitor


def in_window(t: time, start: time, end: time) -> bool:
    """[start, end), also for windows that cross midnight (22:00-05:00)."""
    return start <= t < end if start <= end else t >= start or t < end


class AlertRules:
    """Rules come from the point config:

    - wrong_way: crossing a line against its `allowed` direction
    - restricted_hours: a restricted class crossing a line inside a forbidden window (needs the clock)
    - forbidden_in_zone: a forbidden class inside a zone for at least `min_inside_s`
    - long_dwell: staying in a zone longer than its `max_dwell_s`
    - crowding: a zone reaching `max_occupancy` objects (fires again only after it clears)
    """

    def __init__(self, config: PointConfig, fps: float, start: datetime | None = None, min_inside_s: float = 0.5):
        self.config = config
        self.fps = fps
        self.start = start
        self.min_inside_s = min_inside_s  # filters one-frame flicker of a box over a zone edge
        self._lines = {line.name: line for line in config.lines}
        self._fired: set[tuple] = set()
        self._crowded: set[str] = set()

    def update(self, frame: int, crossings: Iterable[CrossingEvent], zones: ZoneMonitor) -> list[Alert]:
        alerts = []
        for e in crossings:
            line = self._lines[e.line]
            if line.allowed and e.direction != line.allowed:
                alerts.append(Alert(frame, "wrong_way", e.line, e.track_id, e.road_class, f"crossed {e.direction}"))
            if self.start:
                now = clock(self.start, frame, self.fps)
                for r in self.config.restrictions:
                    if r.road_class == e.road_class and any(in_window(now.time(), a, b) for a, b in r.windows):
                        alerts.append(Alert(frame, "restricted_hours", e.line, e.track_id, e.road_class,
                                            f"{e.road_class} at {now:%H:%M}"))

        for zone in self.config.zones:
            for track, enter, last in zones.open_visits(zone.name):
                seconds = (last - enter + 1) / self.fps
                road_class = zones.track_class(track)
                if road_class in zone.forbidden and seconds >= self.min_inside_s:
                    alerts += self._once((zone.name, track, "forbidden"), Alert(
                        frame, "forbidden_in_zone", zone.name, track, road_class, f"{road_class} inside {zone.name}"))
                if zone.max_dwell_s is not None and seconds > zone.max_dwell_s:
                    alerts += self._once((zone.name, track, "dwell"), Alert(
                        frame, "long_dwell", zone.name, track, road_class, f"more than {zone.max_dwell_s:g} s"))
            if zone.max_occupancy is not None:
                n = zones.occupancy(zone.name)
                if n >= zone.max_occupancy and zone.name not in self._crowded:
                    self._crowded.add(zone.name)
                    alerts.append(Alert(frame, "crowding", zone.name, None, None, f"{n} objects"))
                elif n < zone.max_occupancy:
                    self._crowded.discard(zone.name)
        return alerts

    def _once(self, key: tuple, alert: Alert) -> list[Alert]:
        if key in self._fired:
            return []
        self._fired.add(key)
        return [alert]
