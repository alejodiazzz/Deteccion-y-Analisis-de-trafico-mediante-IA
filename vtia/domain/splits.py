"""Train/test partition by capture session (plan §2.3 and capture protocol §1).

Clips cut from the same session share light, place and even vehicles: they must all
land on the same side, or test metrics come out inflated. Sessions are shuffled within
each stratum (camera point x time band) so the test set covers every point and band.
"""
import random
import re
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

_NAME = re.compile(r"^(P[^_]+)_(\d{8})_(\d{2})(\d{2})(?:_s\d+)?$")
# Proposed capture-protocol windows (§4), represented as minutes after midnight.
# Outside these windows (including night) is "other".
BANDS = (("morning", 6 * 60 + 30, 9 * 60),
         ("midday", 11 * 60 + 30, 14 * 60),
         ("afternoon", 16 * 60 + 30, 18 * 60 + 30))


@dataclass(frozen=True)
class Clip:
    name: str      # file name without extension
    session: str   # P<point>_<date>_<HHMM>
    stratum: str   # <point>/<band>


def parse_clip(name: str) -> Clip:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"{name}: expected P<point>_<YYYYMMDD>_<HHMM>[_s<SSSS>] (capture protocol §5)")
    point, date, hh, mm = m.groups()
    try:
        datetime.strptime(date + hh + mm, "%Y%m%d%H%M")
    except ValueError as exc:
        raise ValueError(f"{name}: invalid capture date/time") from exc
    minute_of_day = int(hh) * 60 + int(mm)
    band = next((b for b, lo, hi in BANDS if lo <= minute_of_day < hi), "other")
    return Clip(name, f"{point}_{date}_{hh}{mm}", f"{point}/{band}")


def split(names: list[str], test_share: float = 0.3, folds: int = 5, seed: int = 0) -> dict[str, str]:
    """clip name -> "test" or "fold<i>" (i in 1..folds). Whole sessions move together.

    Per stratum, round(test_share x sessions) go to test (at least one when the stratum has
    two or more sessions); the rest are dealt to the folds round-robin, continuing across
    strata so folds stay balanced. Deterministic for a given seed.
    """
    if not 0 < test_share < 1:
        raise ValueError("test_share must be strictly between 0 and 1")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(names) != len(set(names)):
        raise ValueError("clip names must be unique")
    clips = [parse_clip(n) for n in names]
    sessions_by_stratum: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        sessions_by_stratum[c.stratum].add(c.session)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    next_fold = 0
    for stratum in sorted(sessions_by_stratum):
        sessions = sorted(sessions_by_stratum[stratum])
        rng.shuffle(sessions)
        n_test = round(len(sessions) * test_share)
        if len(sessions) >= 2:
            n_test = max(1, n_test)
        for s in sessions[:n_test]:
            assignment[s] = "test"
        for s in sessions[n_test:]:
            assignment[s] = f"fold{next_fold % folds + 1}"
            next_fold += 1
    return {c.name: assignment[c.session] for c in clips}


def validate_split_assignment(names: list[str], assignment: dict[str, str], folds: int = 5) -> list[str]:
    """Validate coverage, fixed test partition, fold labels and session isolation."""
    problems = []
    expected = set(names)
    actual = set(assignment)
    if expected - actual:
        problems.append(f"clips missing from manifest: {sorted(expected - actual)}")
    if actual - expected:
        problems.append(f"manifest has unknown clips: {sorted(actual - expected)}")
    allowed = {"test", *(f"fold{i}" for i in range(1, folds + 1))}
    invalid = sorted({v for v in assignment.values() if v not in allowed})
    if invalid:
        problems.append(f"invalid partition labels: {invalid}")
    sessions = defaultdict(set)
    for name, part in assignment.items():
        try:
            sessions[parse_clip(name).session].add(part)
        except ValueError as exc:
            problems.append(str(exc))
    for session, parts in sessions.items():
        if len(parts) > 1:
            problems.append(f"capture session {session} spans partitions {sorted(parts)}")
    if "test" not in assignment.values():
        problems.append("fixed test partition is empty")
    present_folds = {p for p in assignment.values() if p.startswith("fold")}
    if len(present_folds) < folds:
        problems.append(f"only {len(present_folds)} of {folds} validation folds are populated")
    return problems
