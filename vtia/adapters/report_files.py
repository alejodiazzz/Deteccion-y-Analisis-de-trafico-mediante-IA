"""Writes a VideoAnalysis as CSV tables (raw data for later analysis)."""
from pathlib import Path

from vtia.adapters.table_files import write_csv
from vtia.application.process_video import VideoAnalysis
from vtia.application.summary import Summary
from vtia.domain.counting import count_by
from vtia.domain.timeline import clock


def write_tables(a: VideoAnalysis, out: Path) -> None:
    """events.csv, counts.csv, visits.csv and alerts.csv. `clock` columns are empty if the start time is unknown."""
    def at(frame: int) -> tuple[float, str]:
        return round(frame / a.fps, 2), (f"{clock(a.start, frame, a.fps):%Y-%m-%d %H:%M:%S}" if a.start else "")

    write_csv(out / "events.csv", ["frame", "seconds", "clock", "track_id", "class", "line", "direction"],
              [(e.frame, *at(e.frame), e.track_id, e.road_class, e.line, e.direction) for e in a.events])
    write_csv(out / "counts.csv", ["line", "direction", "class", "count"],
              [(*key, n) for key, n in sorted(count_by(a.events).items())])
    write_csv(out / "visits.csv", ["zone", "track_id", "class", "enter_seconds", "enter_clock", "dwell_seconds"],
              [(v.zone, v.track_id, v.road_class, *at(v.enter_frame), round((v.exit_frame - v.enter_frame + 1) / a.fps, 2))
               for v in a.visits])
    write_csv(out / "alerts.csv", ["frame", "seconds", "clock", "kind", "where", "track_id", "class", "detail"],
              [(x.frame, *at(x.frame), x.kind, x.where, x.track_id if x.track_id is not None else "",
                x.road_class or "", x.detail) for x in a.alerts])


def write_summary_tables(s: Summary, out: Path) -> None:
    """Aggregates: by_class.csv, by_interval.csv and dwell.csv."""
    write_csv(out / "by_class.csv", ["class", "count", "share"], [(c, n, round(share, 4)) for c, n, share in s.by_class])
    write_csv(out / "by_interval.csv", [f"interval_{s.interval_s}s", "count"], s.by_interval)
    write_csv(out / "dwell.csv", ["zone", "visits", "mean_seconds", "max_seconds"],
              [(z, n, round(m, 2), round(x, 2)) for z, n, m, x in s.dwell])
