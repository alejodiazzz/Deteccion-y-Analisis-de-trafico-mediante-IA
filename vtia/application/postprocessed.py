"""Detector decorator applying a per-frame clean-up step (duplicate removal, rider merge)."""
from collections.abc import Callable, Iterator
from pathlib import Path

from vtia.application.ports import Detector
from vtia.domain.models import Detection


class PostProcessed:
    def __init__(self, inner: Detector, step: Callable[[list[Detection]], list[Detection]], tag: str):
        self.inner = inner
        self.step = step
        self.name = f"{inner.name}+{tag}"  # keeps experiments apart in evaluation outputs

    def detect(self, source: Path) -> Iterator[list[Detection]]:
        for frame in self.inner.detect(source):
            yield self.step(frame)
