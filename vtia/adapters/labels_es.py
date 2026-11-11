"""Spanish labels for what end users read (PDF, dashboard). Code and data files stay in English."""
from vtia.domain.classes import RoadClass

CLASS = {
    RoadClass.PEDESTRIAN: "Peatón",
    RoadClass.BICYCLE: "Bicicleta",
    RoadClass.MOTORCYCLE: "Motocicleta",
    RoadClass.CAR: "Automóvil",
    RoadClass.BUS: "Bus",
    RoadClass.TRUCK: "Camión",
}

ALERT = {
    "wrong_way": "Contraflujo",
    "restricted_hours": "Horario restringido",
    "forbidden_in_zone": "Clase no permitida en zona",
    "long_dwell": "Permanencia prolongada",
    "crowding": "Aglomeración",
}


def duration(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s" if m else f"{s} s"
