"""
Non-Intrusive Load Monitoring (NILM) Predictor.

Calculates whole-building energy usage and disaggregates it to the VAV level proportionally
using a calculated thermal signal. The building's energy is shared between every VAV in the data, including
VAVs that are not in the mapping (they are not reported, but they take their share), so the energy given to the
mapped VAVs is not inflated when the mapping only covers part of the building.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from buildings.building_data import VAV
from models.hvac import HVACRecord, KWH_TO_KBTU, TONS_TO_KBTU_HR, WATER_HEAT_CONSTANT

STREAMS = ("cooling", "heating", "electric")


@dataclass
class NILMResult:
    """
    Result of disaggregating one timestep.

    per_vav: kBTU for each mapped VAV (all three streams summed).
    per_vav_streams: the same per stream, {stream: {VAV: kBTU}}.
    streams: for each stream, the building's kBTU this timestep and where it went:
        {'building': metered total, 'mapped': given to mapped VAVs, 'unmapped': given to VAVs not in the mapping,
         'unallocated': left over because no VAV had a signal to share it by}
    cooling_source: 'tons' (metered tonnage), 'flow' (flow x temperature difference) or 'none'.
    heating_source: 'VAV hhw' (sum of the VAVs' metered HHW power) or 'none'.
    """
    per_vav: Dict[VAV, float]
    streams: Dict[str, Dict[str, float]]
    cooling_source: str
    heating_source: str = "none"
    per_vav_streams: Dict[str, Dict[VAV, float]] = field(default_factory = dict)


def disaggregate_nilm(vavs: Iterable[VAV], record: HVACRecord) -> NILMResult:
    """
    Disaggregates the building-level metered energy to every VAV in the data and returns the share of each
    mapped VAV together with how the building's energy was split (see NILMResult).
    """
    vavs = list(vavs)
    h = record.interval_hours

    ahu_sats = [d.get('sat', 0.0) for d in record.ahu_data.values() if d.get('sat', 0.0) > 0]
    mean_ahu_sat = sum(ahu_sats) / len(ahu_sats) if ahu_sats else 0.0

    # One entry per VAV in the data: (mapped VAV or None, saf, sat, zt, ahu_sat, hhw). VAVs outside the mapping
    # have no known AHU, so the average AHU supply temperature stands in for theirs.
    entries = []
    for vav in vavs:
        d = record.vav_data.get(vav.vav_name, {})
        ahu_sat = record.ahu_data.get(vav.ahu_ref, {}).get('sat', 0.0)
        entries.append((vav, d.get('saf', 0.0), d.get('sat', 0.0), d.get('zt', 0.0), ahu_sat, d.get('hhw', 0.0)))
    for d in record.extra_vav_data.values():
        entries.append((None, d.get('saf', 0.0), d.get('sat', 0.0), d.get('zt', 0.0), mean_ahu_sat,
                        d.get('hhw', 0.0)))

    # Thermal signal of every VAV, one per stream
    signals = {s: [] for s in STREAMS}
    total_hhw_metered = 0.0
    for _, saf, sat, zt, ahu_sat, hhw in entries:
        d_cool = max(zt - sat, 0) if (zt > 0 and sat > 0) else 0
        d_reheat = max(sat - ahu_sat, 0) if (sat > 0 and ahu_sat > 0) else 0
        signals["cooling"].append(saf * d_cool)
        signals["heating"].append(saf * d_reheat)
        signals["electric"].append(saf)
        if hhw > 0:
            total_hhw_metered += hhw * h

    # Building streams
    bldg = record.building_data
    building = {"electric": bldg.get('electric_kw', 0.0) * h * KWH_TO_KBTU,
                "heating": total_hhw_metered,   # metered HHW only, from every VAV in the data
                "cooling": 0.0}

    cooling_source = "none"
    if bldg.get('chw_tons_valid', 0.0) > 0:
        # A metered tonnage column exists: a zero or negative reading means no cooling, not a missing meter
        cooling_source = "tons"
        building["cooling"] = max(bldg.get('chw_tons', 0.0), 0.0) * TONS_TO_KBTU_HR * h
    else:
        chw_flow = bldg.get('chw_flow', 0.0)
        chw_ret = bldg.get('chw_return', 0.0)
        chw_sup = bldg.get('chw_supply', 0.0)
        if chw_flow > 0 and chw_ret > chw_sup:
            cooling_source = "flow"
            building["cooling"] = WATER_HEAT_CONSTANT * chw_flow * (chw_ret - chw_sup) * h / 1000

    # Share each stream by signal; the final prediction of a VAV is the sum of its three shares
    per_vav = {vav: 0.0 for vav in vavs}
    per_vav_streams = {stream: {vav: 0.0 for vav in vavs} for stream in STREAMS}
    streams = {}
    for stream in STREAMS:
        total_signal = sum(signals[stream])
        mapped = unmapped = 0.0
        if total_signal > 0:
            for (vav, *_), sig in zip(entries, signals[stream]):
                share = sig / total_signal * building[stream]
                if vav is None:
                    unmapped += share
                else:
                    per_vav[vav] += share
                    per_vav_streams[stream][vav] += share
                    mapped += share
        streams[stream] = {"building": building[stream], "mapped": mapped, "unmapped": unmapped,
                           "unallocated": building[stream] - mapped - unmapped}

    return NILMResult(per_vav = per_vav, streams = streams, cooling_source = cooling_source,
                      heating_source = "VAV hhw" if total_hhw_metered > 0 else "none",
                      per_vav_streams = per_vav_streams)


def predict_nilm_consumption(vavs: Iterable[VAV], record: HVACRecord) -> Dict[VAV, float]:
    """
    Disaggregates building-level metered energy to individual VAVs and returns a mapping of
    each mapped VAV to its disaggregated energy in kBTU.
    """
    return disaggregate_nilm(vavs, record).per_vav


@dataclass
class NILMAudit:
    """
    Adds up the building's metered cooling, heating and electric energy over all the data and where NILM gave
    it, so the result can be checked: for NILM, the energy given to the rooms must add up to the building's
    energy minus the share of VAVs outside the mapping and any energy that no VAV could be given.
    """
    totals: Dict[str, Dict[str, float]] = field(default_factory = lambda: {
        s: defaultdict(float) for s in STREAMS})
    intervals: int = 0
    cooling_sources: Dict[str, int] = field(default_factory = lambda: defaultdict(int))
    heating_sources: Dict[str, int] = field(default_factory = lambda: defaultdict(int))

    def add(self, result: NILMResult) -> None:
        self.intervals += 1
        self.cooling_sources[result.cooling_source] += 1
        self.heating_sources[result.heating_source] += 1
        for stream, parts in result.streams.items():
            for key, value in parts.items():
                self.totals[stream][key] += value

    def report(self, n_mapped_vavs: int, n_data_vavs: int, room_totals: Dict[str, float]) -> List[str]:
        """
        Prints the check and returns the problems it found.

        room_totals is the NILM energy of all the rooms for each stream, added up from the results the rest of
        the program uses (not from this audit). For NILM it must match the building: the building's energy equals
        what the rooms received, plus what VAVs outside the mapping received, plus what no VAV could be given.
        """
        problems = []

        def kbtu(value: float) -> str:
            return f"{0.0 if abs(value) < 0.5 else value:,.0f}"    # no '-0' from floating point leftovers

        print(f"\nNILM check ({self.intervals} intervals, kBTU; {n_data_vavs} VAVs in the data, "
              f"{n_mapped_vavs} in the mapping)")
        print("  Building totals compared with the room totals:")
        print(f"  {'':10}{'building':>14}{'rooms':>14}{'unmapped VAVs':>15}{'not shared':>12}{'difference':>12}")

        sums = defaultdict(float)
        for stream in STREAMS:
            t = self.totals[stream]
            rooms = room_totals.get(stream, 0.0)
            difference = t['building'] - rooms - t['unmapped'] - t['unallocated']
            print(f"  {stream:10}{kbtu(t['building']):>14}{kbtu(rooms):>14}{kbtu(t['unmapped']):>15}"
                  f"{kbtu(t['unallocated']):>12}{kbtu(difference):>12}")
            for key, value in (("building", t['building']), ("rooms", rooms), ("unmapped", t['unmapped']),
                               ("unallocated", t['unallocated']), ("difference", difference)):
                sums[key] += value

            if abs(difference) > 1e-6 * max(t['building'], 1.0):
                problems.append(f"{stream}: building {t['building']:,.1f} is not rooms {rooms:,.1f} + unmapped VAVs "
                                f"{t['unmapped']:,.1f} + not shared {t['unallocated']:,.1f}")
            if t['building'] <= 0:
                problems.append(f"no {stream} energy in the data, so NILM has no {stream} in it")
            elif t['unallocated'] > 0.01 * t['building']:
                problems.append(f"{t['unallocated'] / t['building']:.0%} of the {stream} energy was not given to any "
                                f"VAV (no VAV had airflow/signal at those times)")
        print(f"  {'total':10}{kbtu(sums['building']):>14}{kbtu(sums['rooms']):>14}{kbtu(sums['unmapped']):>15}"
              f"{kbtu(sums['unallocated']):>12}{kbtu(sums['difference']):>12}")
        print(f"  The rooms received {sums['rooms'] / max(sums['building'], 1e-9):.0%} of the building's energy; "
              f"{sums['unmapped'] / max(sums['building'], 1e-9):.0%} went to VAVs outside the mapping.")
        for label, sources in (("cooling", self.cooling_sources), ("heating", self.heating_sources)):
            print(f"  {label} taken from: " + ", ".join(f"{k} in {v} intervals" for k, v in sources.items()))
        for problem in problems:
            print(f"  Warning: {problem}")
        return problems