"""Pure validation rules for dataset partition manifests."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json


def read_split_manifest(path: Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"clip", "split"}.issubset(reader.fieldnames):
            raise ValueError("split manifest must contain clip,split columns")
        assignment = {}
        for line, row in enumerate(reader, 2):
            clip, part = (row.get("clip") or "").strip(), (row.get("split") or "").strip()
            if not clip or part not in {"test"} and not part.startswith("fold"):
                raise ValueError(f"invalid split manifest row {line}: clip={clip!r}, split={part!r}")
            if clip in assignment:
                raise ValueError(f"duplicate clip in split manifest: {clip}")
            assignment[clip] = part
    if not assignment:
        raise ValueError("split manifest is empty")
    sidecar = Path(path).with_suffix(".manifest.json")
    if sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            expected_hash = metadata.get("assignment_sha256")
            actual_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if expected_hash and expected_hash != actual_hash:
                raise ValueError("split CSV SHA-256 does not match its manifest; regenerate the split")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid split metadata JSON: {sidecar}") from exc
    return assignment


def validate_session_partition(names: list[str], assignment: dict[str, str]) -> list[str]:
    from vtia.domain.splits import parse_clip
    problems = []
    sessions: dict[str, set[str]] = defaultdict(set)
    for name in names:
        if name not in assignment:
            problems.append(f"{name}: missing from split manifest")
            continue
        sessions[parse_clip(name).session].add(assignment[name])
    for session, parts in sessions.items():
        if len(parts) > 1:
            problems.append(f"session {session} appears in multiple partitions: {sorted(parts)}")
    extras = sorted(set(assignment) - set(names))
    if extras:
        problems.append(f"manifest contains clips not present in dataset: {extras}")
    return problems


def select_training_sequences(sequences, assignment: dict[str, str], fold: int):
    """Return (train, validation) sequences; the fixed test partition is never returned."""
    if fold not in range(1, 6):
        raise ValueError("validation fold must be between 1 and 5")
    missing = sorted(seq.name for seq in sequences if seq.name not in assignment)
    if missing:
        raise ValueError(f"sequences missing from split manifest: {missing}")
    validation_part = f"fold{fold}"
    train = [seq for seq in sequences if assignment[seq.name].startswith("fold") and assignment[seq.name] != validation_part]
    validation = [seq for seq in sequences if assignment[seq.name] == validation_part]
    if not train or not validation:
        raise ValueError("need at least one training and one validation sequence in the selected fold")
    return train, validation
