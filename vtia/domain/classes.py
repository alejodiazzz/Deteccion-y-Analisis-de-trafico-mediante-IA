"""The six road-user classes and how pretrained-model labels map onto them."""
from enum import StrEnum


class RoadClass(StrEnum):
    PEDESTRIAN = "pedestrian"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"


# COCO label -> project class. Known mismatches (riders boxed as a separate
# "person", pickups labelled "truck") are handled in later phases.
COCO_TO_ROAD = {
    "person": RoadClass.PEDESTRIAN,
    "bicycle": RoadClass.BICYCLE,
    "motorcycle": RoadClass.MOTORCYCLE,
    "car": RoadClass.CAR,
    "bus": RoadClass.BUS,
    "truck": RoadClass.TRUCK,
}

# Any label a detector may emit -> project class: COCO names (pretrained) or our own (fine-tuned models).
MODEL_LABELS = COCO_TO_ROAD | {c.value: c for c in RoadClass}
