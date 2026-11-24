"""
VAV damper position. Each VAV's 'supply air damper position' (%) is collected over the data and the VAV is put in
a bucket by its average position. The bucket belongs to the VAV: rooms are stratified through the VAV that feeds them.

    closed        average position below 10 %
    low           10 % up to 30 %
    moderate      30 % up to 45 %     (where most Bren Hall VAVs sit: parked near their minimum flow)
    partly_open   45 % up to 65 %
    mostly_open   65 % up to 95 %     (where most SBSG VAVs sit)
    fully_open    95 % and above      (damper pinned open)
    no data       no readings for the VAV

The edges are absolute positions, not percentiles of one building, so the buckets mean the same thing in every
building and in every period of data.

Damper swing (how far a damper moves, see swing_pattern) is kept as a separate method and is not used for the
buckets or plots; call DamperProfile.swing_summary() to get it.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from models.hvac import HVACRecord

# (name, lower edge in %, inclusive); a VAV is in the last bucket whose lower edge its average position reaches
BUCKETS = (("closed", 0.0), ("low", 10.0), ("moderate", 30.0), ("partly_open", 45.0),
           ("mostly_open", 65.0), ("fully_open", 95.0))
NO_DATA = "no data"
BUCKET_NAMES = tuple(name for name, _ in BUCKETS) + (NO_DATA,)


def position_bucket(mean_position: float) -> str:
    """The bucket for an average damper position in %."""
    if mean_position is None or math.isnan(mean_position):
        return NO_DATA
    bucket = BUCKETS[0][0]
    for name, lower in BUCKETS:
        if mean_position >= lower:
            bucket = name
    return bucket


def position_summary(readings: Iterable[float]) -> Dict[str, float]:
    """Bucket and statistics of one VAV's damper readings (NaN readings are ignored)."""
    x = np.array([r for r in readings if not math.isnan(r)], dtype = float)
    if x.size == 0:
        return {"bucket": NO_DATA, "readings": 0}
    mean = float(x.mean())
    return {"bucket": position_bucket(mean), "readings": int(x.size), "mean": mean, "median": float(np.median(x)),
            "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "share_closed": float((x < 10).mean()), "share_fully_open": float((x >= 95).mean())}


# ─── Damper swing (not used by the buckets or plots) ───────────────────────────────────
CLOSED_PCT, OPEN_PCT, STUCK_SHARE, FLAT_SWING, MODULATING_SWING = 10.0, 95.0, 0.90, 5.0, 15.0


def swing_pattern(readings: Iterable[float]) -> Dict[str, float]:
    """
    How the damper moves, apart from where it sits: swing = 95th percentile - 5th percentile of the VAV's own
    readings. Pattern: stuck_open / closed (pinned for over 90 % of readings), flat (swing < 5 points),
    min_mostly (5-15 points) or modulating (over 15 points).
    """
    x = np.array([r for r in readings if not math.isnan(r)], dtype = float)
    if x.size == 0:
        return {"swing_pattern": NO_DATA, "readings": 0}
    swing = float(np.percentile(x, 95) - np.percentile(x, 5))
    if (x >= OPEN_PCT).mean() > STUCK_SHARE:
        pattern = "stuck_open"
    elif (x < CLOSED_PCT).mean() > STUCK_SHARE:
        pattern = "closed"
    elif swing < FLAT_SWING:
        pattern = "flat"
    elif swing <= MODULATING_SWING:
        pattern = "min_mostly"
    else:
        pattern = "modulating"
    return {"swing_pattern": pattern, "swing": swing, "readings": int(x.size)}


class DamperProfile:
    """Collects damper readings of every mapped VAV and summarizes them after all records were added."""

    def __init__(self) -> None:
        self.readings: Dict[str, List[float]] = defaultdict(list)
        self._by_time = defaultdict(lambda: defaultdict(list))     # (day, hour) -> vav -> readings

    def add_record(self, record: HVACRecord) -> None:
        key = (record.day_of_week, record.hour)
        for vav, d in record.vav_data.items():
            value = d.get('damper', math.nan)
            self.readings[vav].append(value)
            if not math.isnan(value):
                self._by_time[key][vav].append(value)

    def vav_summary(self) -> Dict[str, Dict[str, float]]:
        """Bucket and position statistics of each VAV: {vav_name: {...}}."""
        return {vav: position_summary(values) for vav, values in self.readings.items()}

    def buckets(self) -> Dict[str, str]:
        """The bucket of every VAV with readings: {vav_name: bucket}. A VAV is in the bucket its own damper gives."""
        return {vav: info["bucket"] for vav, info in self.vav_summary().items()}

    def swing_summary(self) -> Dict[str, Dict[str, float]]:
        """Swing and swing pattern of each VAV. Not used by the buckets or plots."""
        return {vav: swing_pattern(values) for vav, values in self.readings.items()}

    def average_by_time(self) -> Dict[tuple, Dict[str, float]]:
        """Average damper position per (day of week, hour) for each VAV, same keys as the energy averages."""
        return {key: {vav: float(np.mean(v)) for vav, v in per_vav.items()} for key, per_vav in self._by_time.items()}

    def room_members(self, building_vavs, bucket_of: Dict[str, str]) -> Dict[tuple, List[str]]:
        """
        Pairs of (room, bucket) with the VAVs behind them: {(room, bucket): [vav names]}. A room fed by VAVs in
        different buckets has one pair per bucket, holding only the VAVs (and so the energy) of that bucket.
        """
        members: Dict[tuple, List[str]] = defaultdict(list)
        for vav in building_vavs:
            bucket = bucket_of.get(vav.vav_name, NO_DATA)
            for room in vav.rooms_associated:
                members[(room.room_num, bucket)].append(vav.vav_name)
        return members

    def member_average_by_time(self, members: Dict[tuple, List[str]]) -> Dict[tuple, Dict[tuple, float]]:
        """Average damper position per (day, hour) for each (room, bucket): mean over its VAVs with readings."""
        out = {}
        for key, per_vav in self.average_by_time().items():
            row = {}
            for pair, vavs in members.items():
                vals = [per_vav[v] for v in vavs if v in per_vav]
                if vals:
                    row[pair] = float(np.mean(vals))
            out[key] = row
        return out