import pandas as pd
from pathlib import Path


class Room:
    """
    Represents a physical room in the building.

    Attributes:
        room_num (int): The unique room identifier/number.
        room_type (str): The classification of the room
        room_capacity (int): Maximum occupancy of the room.
        room_bucket (str): The categorized room size for the room.
        room_size_sf (float): The room area in square feet.
    """

    def __init__(self, room_num: int, room_type: str, room_capacity: int, room_bucket: str,
                 room_size_sf: float = 0.0):
        self.room_num = room_num
        self.room_type = room_type
        self.room_capacity = room_capacity
        self.room_bucket = room_bucket
        self.room_size_sf = room_size_sf

    def __repr__(self) -> str:
        return f"Room({self.room_num}, {self.room_type!r}, cap={self.room_capacity}, {self.room_bucket!r})"

    def get_room_weight(self) -> float:
        """
        Returns the proportional share weight based on the room size class. Sub-strata such as S3 or M1
        use the weight of their base class (S, M, L, XL).
        """
        weights = {'S': 1.0, 'M': 2.0, 'L': 4.0, 'XL': 8.0,
                   'S1': 1, 'S2': 1.5, 'S3': 2.0, 'S4': 2.5, 'S5': 2.9, 'S6': 3.5,
                   'M1': 4.8, 'M2': 6.6, 'M3': 9.0, 'M4': 12.0,
                   'L1': 18.0,
                   'XL1': 30.0
                   }

        bucket = self.room_bucket.strip().upper()
        return weights[bucket] if bucket in weights else 1.0


class VAV:
    """
    Represents a Variable Air Volume (VAV) box that regulates air to specific rooms.

    Attributes:
        vav_name (str): The unique identifier for the VAV.
        ahu_ref (str): The short reference ID of the Air Handling Unit (AHU) supplying this VAV.
        rooms_associated (list[Room]): A list of Room objects serviced by this VAV.
    """

    def __init__(self, vav_name: str, ahu_ref: str):
        self.vav_name = vav_name
        self.ahu_ref = ahu_ref
        self.rooms_associated: list[Room] = []

    def __str__(self) -> str:
        return f'{self.vav_name}'

    def __repr__(self) -> str:
        return f'VAV({self.vav_name!r}, ahu={self.ahu_ref!r}, rooms={len(self.rooms_associated)})'

    def __eq__(self, other: object) -> bool:
        """Determines equality based strictly on the unique VAV name."""
        if not isinstance(other, VAV):
            return NotImplemented
        return self.vav_name == other.vav_name

    def __hash__(self) -> int:
        """Allows VAV objects to be stored in sets or used as dictionary keys."""
        return hash(self.vav_name)

    def add_room(self, room: Room) -> None:
        """Assigns a Room to this VAV's service list."""
        self.rooms_associated.append(room)


# Accepted header names (case-insensitive) for each field in the mapping file.
_COLUMN_ALIASES = {
    "vav_name": ("vavname", "vav_name", "vav", "vav_id"),
    "ahu_ref": ("ahuref", "ahu_ref", "ahu", "ahu_id", "ahu_system"),
    "room_num": ("room_number", "room_num", "room"),
    "room_type": ("room_type", "type"),
    "room_capacity": ("room_capacity", "capacity"),
    "room_bucket": ("room_bucket", "bucket"),
    "room_size_sf": ("room_size_sf", "size_sf", "room_area_sf", "area_sf"),
}


def _resolve_columns(columns: list[str]) -> dict[str, str | None]:
    """
    Returns a mapping of the standard field names to actual CSV column names.
    Raises a ValueError if a required column ('vav_name' or 'ahu_ref') is missing.
    """
    lowered = {c.strip().lower(): c for c in columns}
    resolved = {}

    for field, aliases in _COLUMN_ALIASES.items():
        resolved[field] = next((lowered[a] for a in aliases if a in lowered), None)

    for required in ("vav_name", "ahu_ref"):
        if resolved[required] is None:
            raise ValueError(
                f"Mapping file needs a '{required}' column (accepted names: {_COLUMN_ALIASES[required]}); "
                f"found {columns}"
            )
    return resolved


def _short_ahu_ref(ref: str) -> str:
    """
    Extracts the base AHU ID from a given descriptive string. Example: 'Bren Hall AH-01' -> 'AH-01'
    """
    parts = ref.split()
    return parts[-1] if parts else ""


def get_building_vavs(file: Path) -> set[VAV]:
    """
    Parses the given VAV-to-room mapping CSV file and returns a set of all unique VAV objects that
    represents the building's HVAC zones with their associated rooms.
    Raises ValueError if a VAV is mapped to multiple different AHUs, or if room data is malformed.
    """
    df = pd.read_csv(file, dtype = str, keep_default_na = False)
    cols = _resolve_columns(list(df.columns))

    def cell(row: pd.Series, field: str) -> str:
        col = cols[field]
        return row[col].strip() if col else ""

    vavs: dict[str, VAV] = {}

    for line_no, row in df.iterrows():
        vav_name = cell(row, "vav_name")
        if not vav_name:
            continue

        ahu_ref = _short_ahu_ref(cell(row, "ahu_ref"))
        vav = vavs.get(vav_name)

        if vav is None:
            vav = vavs[vav_name] = VAV(vav_name, ahu_ref)
        elif ahu_ref and vav.ahu_ref != ahu_ref:
            raise ValueError(
                f"{file}: VAV {vav_name!r} maps to two AHUs ({vav.ahu_ref!r} and {ahu_ref!r})"
            )

        room_num = cell(row, "room_num")
        if not room_num:
            continue

        try:
            vav.add_room(Room(
                room_num = int(float(room_num)),
                room_type = cell(row, "room_type"),
                room_capacity = int(float(cell(row, "room_capacity") or 0)),
                room_bucket = cell(row, "room_bucket"),
                room_size_sf = float(cell(row, "room_size_sf") or 0),
            ))
        except ValueError as e:
            raise ValueError(f"{file}: bad room data on row {line_no + 2}: {e}") from e

    return set(vavs.values())


def group_vavs_by_ahu(vavs: set[VAV]) -> dict[str, list[VAV]]:
    """
    Groups a collection of VAVs by their parent Air Handling Unit (AHU).
    Given a set of VAV objects, returns a dictionary mapping AHU references to lists of VAVs.
    """
    groups: dict[str, list[VAV]] = {}
    for vav in vavs:
        groups.setdefault(vav.ahu_ref, []).append(vav)
    return groups
