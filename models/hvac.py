"""
Shared HVAC Data Models and Loaders.

This module provides data structures and a memory-efficient generator to parse HVAC CSV data
line-by-line. It defines the physical constants and equipment mappings required for
energy consumption calculations.
"""

import csv
import re
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from itertools import chain, islice
from pathlib import Path
from typing import Dict, List, Generator, Any, Union, Set, Iterable, Optional, Tuple
from dateutil import parser, tz


# Physical and Conversion Constants
SENSIBLE_HEAT_CONSTANT = 1.08    # BTU / (hr * cfm * °F)
WATER_HEAT_CONSTANT = 500        # BTU / (hr * gpm * °F)
MIN_DELTA_T = 0.5                # °F, noise floor for temperature differentials
KWH_TO_KBTU = 3.412              # kBTU per kWh
TONS_TO_KBTU_HR = 12.0           # kBTU/hr per Ton of refrigeration
HP_TO_KW = 0.7457                # kW per horsepower



# Timestamps are used exactly as they are written in the data, so a day is the date shown in the file
# Set CONVERT_TO_LOCAL_TIME = True to convert timestamps that carry a UTC offset into LOCAL_TZ first,
# which makes days and hours local midnight-to-midnight.
CONVERT_TO_LOCAL_TIME = False
LOCAL_TZ = tz.gettz("America/Los_Angeles")


# Buildings not listed here have their fans detected from the data, so an entry is only
# needed when detection can't tell which AHU a fan belongs to and you know better.
FAN_MAP: Dict[str, Dict[str, List[str]]] = {
    "Bren Hall": {
        "AH-01": ["SF-1", "SF-2", "RF-1", "RF-2"],
        "AH-02": ["SF-3", "SF-4", "RF-3", "RF-4"],
    },
}
FAN_POWER_COLUMN = "{p} {f} elec power"
FAN_NAME_PATTERN = re.compile(r"(?:^|[.\-_])(?:sf|rf)-?\d")    # supply/return fan names like 'SF-1', 'A01.RF1B.VFD'

# Column-name templates ({p} = prefix, {v} = VAV name, {a} = AHU name). Each field lists its possible names in
# order of preference; the first one the data actually has is used, so new naming styles only need a new entry here.
COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "vav_saf": ["{p} {v} supply air flow cfm", "{p} {v} supply air flow.1", "{p} {v} supply air flow"],
    "vav_sat": ["{p} {v} supply air temp"],
    "vav_zt": ["{p} {v} zone temp"],
    "vav_hhw": ["{p} {v} hhw htg power (calculated)"],
    "ahu_sat": ["{p} {a} supply air temp"],
    "electric_kw": ["{p} Whole Building Electric Meter total elec power",
                    "{p} Whole Building Electric Meter elec power",
                    "{p} Whole Building Electric Meter elec power (NA removed)"],
    "chw_tons": ["{p} CHW chw clg power", "{p} chw btu meter chw clg power"],
    "chw_flow": ["{p} CHW chw flow", "{p} chw btu meter chw flow"],
    "chw_return": ["{p} CHW chw return temp", "{p} chw btu meter chw return temp"],
    "chw_supply": ["{p} CHW chw supply temp", "{p} chw btu meter chw supply temp"],
}
# A candidate is rejected if it holds values that don't carry this unit
COLUMN_UNITS: Dict[str, str] = {"vav_saf": "cfm"}

# Fields that make the results meaningless if their columns are missing
REQUIRED_FIELDS = ("vav_saf", "vav_sat", "vav_zt", "ahu_sat", "electric_kw")


@dataclass
class HVACRecord:
    """
    Represents a stream for a single timestamp's HVAC data across the entire building.

    Attributes:
        timestamp (datetime): The naive datetime of the current record.
        interval_hours (float): The duration of this reading in hours.
        vav_data (Dict[str, Dict[str, float]]): Data keyed by VAV name.
        ahu_data (Dict[str, Dict[str, float]]): Data keyed by AHU reference.
        building_data (Dict[str, float]): Building-level metered data.
    """
    timestamp: datetime
    interval_hours: float
    vav_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ahu_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    building_data: Dict[str, float] = field(default_factory=dict)

    @property
    def day_of_week(self) -> int:
        """Returns the day of the week as an integer (0 = Monday, 6 = Sunday)."""
        return self.timestamp.weekday()

    @property
    def hour(self) -> int:
        """Returns the hour of the day (0-23)."""
        return self.timestamp.hour


class EnergyAccumulator:
    """
    Accumulates calculated energy consumption over time. Groups calculated ILM and NILM energy
    values by day of the week and hour for reporting or downstream visualization.
    """

    def __init__(self) -> None:
        """Dictionary structure: consumption[day_of_week, hour][vav_name] = [ilm_energy, nilm_energy]"""
        self.consumption = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        # Distinct calendar dates seen for each (day_of_week, hour); equals the number of weeks of data.
        self._dates_seen = defaultdict(set)

    def add_record(self, record: HVACRecord, ilm_results: Dict[Any, float], nilm_results: Dict[Any, float]) -> None:
        """
        Adds a single timestep's calculations to the running total given the HVACRecord, ilm_results,
        and nilm_results per VAV.
        """
        key = (record.day_of_week, record.hour)
        self._dates_seen[key].add(record.timestamp.date())
        for vav, energy in ilm_results.items():
            self.consumption[key][vav.vav_name][0] += energy
        for vav, energy in nilm_results.items():
            self.consumption[key][vav.vav_name][1] += energy


    def average(self) -> Dict[Any, Dict[str, List[float]]]:
        """
        Returns consumption averaged per week: each (day_of_week, hour) bucket is divided by the number
        of distinct dates that contributed to it. Same structure as self.consumption.
        """
        averaged = {}
        for key, per_vav in self.consumption.items():
            n_weeks = max(len(self._dates_seen[key]), 1)
            averaged[key] = {name: [ilm / n_weeks, nilm / n_weeks] for name, (ilm, nilm) in per_vav.items()}
        return averaged


def _parse_timestamp(raw: str) -> datetime:
    """
    Parses a timestamp cell into a naive datetime holding the date and time as written in the data.
    """
    text = raw.strip()
    head, _, tail = text.rpartition(" ")
    if head and re.fullmatch(r"[A-Za-z_/]+", tail):
        text = head
    dt = parser.parse(text)
    if dt.tzinfo is not None and CONVERT_TO_LOCAL_TIME:
        dt = dt.astimezone(LOCAL_TZ)
    return dt.replace(tzinfo = None)


def _safe_float(val: Any) -> float:
    """
    Safely casts a raw CSV string to a float and returns it, stripping units if present.
    Returns 0.0 if parsing fails or value is None.
    """
    if not val:
        return 0.0
    try:
        cleaned = re.sub(r"[^\d\.\-]", "", str(val))
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def _power_kw(val: Any) -> float:
    """Parses a power reading and returns kW, converting from hp when the raw string carries an 'hp' unit."""
    kw = _safe_float(val)
    return kw * HP_TO_KW if "hp" in str(val).lower() else kw


def _normalize(text: str) -> str:
    """Lowercases and collapses whitespace, the form header lookups are keyed by."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def detect_prefix(headers: Iterable[str], vav_names: List[str]) -> str:
    """
    Infers the column prefix from the data: it is the text in front of a VAV name in the VAV columns, e.g.
    'Bren Hall' in 'Bren Hall VAV1.1 zone temp'. Each VAV votes once and the most common prefix wins.
    """
    cleaned = [re.sub(r"\s+", " ", str(h).strip()) for h in headers if h]
    votes: Counter = Counter()
    for vav in vav_names:
        pattern = re.compile(rf"^(.+?) {re.escape(vav)} ", re.IGNORECASE)
        for header in cleaned:
            match = pattern.match(header)
            if match:
                votes[match.group(1)] += 1
                break
    if not votes:
        raise ValueError(
            f"Could not infer the column prefix: none of the {len(vav_names)} VAV names in the mapping "
            f"(e.g. {vav_names[:2]}) appear in the HVAC data's column headers (e.g. {cleaned[1:3]}). "
            f"Check that the mapping's vavName values match the data."
        )
    return votes.most_common(1)[0][0]


def _resolve_columns(prefix: str, vav_names: List[str], ahu_names: List[str], header_map: dict,
                     sample_rows: List[dict]) -> Dict[str, str]:
    """
    Picks, for every field, the first candidate template in COLUMN_CANDIDATES whose columns exist in the data
    (the one found for the most VAVs/AHUs wins; ties go to the earlier candidate). Candidates whose values
    visibly lack the field's expected unit are skipped.
    """
    resolved = {}
    for field_name, templates in COLUMN_CANDIDATES.items():
        best, best_found = templates[0], 0
        for template in templates:
            if "{v}" in template:
                names = [template.format(p = prefix, v = v) for v in vav_names]
            elif "{a}" in template:
                names = [template.format(p = prefix, a = a) for a in ahu_names]
            else:
                names = [template.format(p = prefix)]
            found = [header_map[_normalize(n)] for n in names if _normalize(n) in header_map]

            unit = COLUMN_UNITS.get(field_name)
            if unit and found:
                values = [str(row.get(c, "")).lower() for c in found[:20] for row in sample_rows if row.get(c)]
                if values and not any(unit in v for v in values):
                    continue
            if len(found) > best_found:
                best, best_found = template, len(found)
        resolved[field_name] = best
    return resolved


def _detect_fans(prefix: str, header_map: dict, ahu_names: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Finds supply/return fan power columns by name. A fan is assigned to an AHU when its name starts
    with the AHU's name. Fans that can't be tied to an AHU are returned  as a pooled list and
    their power is shared between the AHUs by airflow.
    """
    pattern = re.compile(rf"^{re.escape(_normalize(prefix))} (\S+) elec power$")
    per_ahu: Dict[str, List[str]] = {a: [] for a in ahu_names}
    pooled: List[str] = []
    for key in header_map:
        match = pattern.match(key)
        if not match or not FAN_NAME_PATTERN.search(match.group(1)):
            continue
        fan = match.group(1)
        owner = next((a for a in ahu_names if re.match(rf"{re.escape(a.lower())}[._]", fan)), None)
        (per_ahu[owner] if owner else pooled).append(fan)
    return per_ahu, pooled


def _warn_missing_columns(prefix: str, vav_names: List[str], ahu_names: List[str], columns: Dict[str, str],
                          header_map: dict, building_fans: Dict[str, List[str]], pooled_fans: List[str]) -> None:
    """
    Prints a warning for every required data column the file does not contain. A missing column is
    otherwise read as 0, which silently turns the predictions into zeros.
    """
    def exists(name: str) -> bool:
        return _normalize(name) in header_map

    expected = {
        "vav_saf": [columns["vav_saf"].format(p = prefix, v = v) for v in vav_names],
        "vav_sat": [columns["vav_sat"].format(p = prefix, v = v) for v in vav_names],
        "vav_zt": [columns["vav_zt"].format(p = prefix, v = v) for v in vav_names],
        "ahu_sat": [columns["ahu_sat"].format(p = prefix, a = a) for a in ahu_names],
        "electric_kw": [columns["electric_kw"].format(p = prefix)],
    }
    for field_name in REQUIRED_FIELDS:
        missing = [name for name in expected[field_name] if not exists(name)]
        if missing:
            print(f"Warning: {len(missing)} of {len(expected[field_name])} '{field_name}' columns not found in the "
                  f"data (read as 0), e.g. {missing[0]!r}")

    all_fans = [f for fans in building_fans.values() for f in fans] + pooled_fans
    if not all_fans:
        print(f"Warning: no fan power columns found for {prefix!r}; fan energy is 0")
    missing_fans = [f for f in all_fans if not exists(FAN_POWER_COLUMN.format(p = prefix, f = f))]
    if missing_fans:
        print(f"Warning: fan columns not found in the data: {missing_fans}")


def hvac_record_generator(filepath: Union[str, Path], active_vavs: Iterable[Any], prefix: Optional[str] = None,
                          interval_hours: float = 0.25) -> Generator[HVACRecord, None, None]:
    """
    A memory-efficient generator that parses HVAC data line-by-line to yield an HVACRecord of
    all the necessary sensor data for the current timestamp.

    Args:
        filepath (str): The file path to the CSV data.
        active_vavs (List[VAV]): List of VAV objects to parse data for.
        prefix (str): The building prefix used in the CSV column headers. If None (default), it is detected
            from the data using the VAV names.
        interval_hours (float): The time interval of each row (default is 15 minutes).
    """
    active_vavs = list(active_vavs)
    vav_names = [v.vav_name for v in active_vavs]
    ahu_names = list({v.ahu_ref for v in active_vavs})
    vavs_of_ahu: Dict[str, List[str]] = defaultdict(list)
    for v in active_vavs:
        vavs_of_ahu[v.ahu_ref].append(v.vav_name)

    with open(filepath, 'r', encoding = 'utf-8') as f:
        reader = csv.DictReader(f)

        header_map = {}
        for k in reader.fieldnames or []:
            if k:
                header_map[_normalize(k)] = k

        if prefix is None:
            prefix = detect_prefix(reader.fieldnames or [], vav_names)
            print(f"Detected column prefix: {prefix!r}")

        head = list(islice(reader, 200))    # peeked at to check column units, then processed like the rest
        columns = _resolve_columns(prefix, vav_names, ahu_names, header_map, head)

        fan_override = {k.lower(): v for k, v in FAN_MAP.items()}.get(prefix.lower())
        if fan_override is not None:
            building_fans, pooled_fans = fan_override, []
        else:
            building_fans, pooled_fans = _detect_fans(prefix, header_map, ahu_names)
            print(f"Detected fans: { {a: fans for a, fans in building_fans.items() if fans} } pooled: {pooled_fans}")

        def col(field_name: str, **names: str) -> str:
            return columns[field_name].format(p = prefix, **names)

        _warn_missing_columns(prefix, vav_names, ahu_names, columns, header_map, building_fans, pooled_fans)

        def get_val(row_dict: dict, expected_col: str) -> Any:
            """Fetches a value ignoring case and extra spacing."""
            actual_col = header_map.get(_normalize(expected_col))
            return row_dict.get(actual_col, 0) if actual_col else 0

        for row in chain(head, reader):
            actual_ts_col = header_map.get('timestamp')
            if not actual_ts_col or not row.get(actual_ts_col):
                continue

            dt = _parse_timestamp(row[actual_ts_col])
            record = HVACRecord(timestamp = dt, interval_hours = interval_hours)

            # Parse VAV Data
            for vav in vav_names:
                record.vav_data[vav] = {
                    'saf': _safe_float(get_val(row, col("vav_saf", v = vav))),
                    'sat': _safe_float(get_val(row, col("vav_sat", v = vav))),
                    'zt': _safe_float(get_val(row, col("vav_zt", v = vav))),
                    'hhw': _safe_float(get_val(row, col("vav_hhw", v = vav)))
                }

            # Fans that can't be tied to one AHU are shared between the AHUs by their airflow
            pooled_kw = sum(_power_kw(get_val(row, FAN_POWER_COLUMN.format(p = prefix, f = fan)))
                            for fan in pooled_fans)
            ahu_saf = {a: sum(record.vav_data[v]['saf'] for v in vavs) for a, vavs in vavs_of_ahu.items()}
            total_saf = sum(ahu_saf.values())

            # Parse AHU Data
            for ahu in ahu_names:
                fan_power_kw = sum(
                    _power_kw(get_val(row, FAN_POWER_COLUMN.format(p = prefix, f = fan)))
                    for fan in building_fans.get(ahu, [])
                )
                if pooled_kw:
                    fan_power_kw += pooled_kw * (ahu_saf[ahu] / total_saf if total_saf > 0 else 1 / len(ahu_names))
                record.ahu_data[ahu] = {
                    'sat': _safe_float(get_val(row, col("ahu_sat", a = ahu))),
                    'fan_kw': fan_power_kw
                }

            # Parse Building Data
            for key in ('electric_kw', 'chw_tons', 'chw_flow', 'chw_return', 'chw_supply'):
                record.building_data[key] = _safe_float(get_val(row, col(key)))

            yield record
