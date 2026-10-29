"""
Author: Aadi Agrawal
Purpose: A program to predict/plot room energy consumption for a smart building from HVAC data
"""

import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Set

import buildings
import models
import plot
from buildings.building_data import VAV


# ─── Configuration ─────────────────────────────────────────────────────────────────────
"""
If chosen to not use the command line, can configure the program using these global variables
"""
BUILDING = 'BrenHall'    # the smart building in which room energy consumption is investigated
PURPOSE = 'predict'    # predict = 0, plot = 1, summarize = 2
ROOMS = None             # None = all rooms, STR = specific room or room type
# ───────────────────────────────────────────────────────────────────────────────────────


def _get_rooms_to_predict(vavs: Set[VAV], rooms: str | None) -> Set[str]:
    """
    None = all rooms. Otherwise, rooms is a room number if it matches one in the building,
    else a room type.
    """
    all_rooms = {room.room_num: room for vav in vavs for room in vav.rooms_associated}
    if rooms is None:
        return set(all_rooms)
    if str(rooms).strip() in all_rooms:
        return {str(rooms).strip()}
    return {rm for rm, room in all_rooms.items() if room.room_type == rooms}


def _write_room_energy_consumption(room_consumption, building: str) -> None:
    """
    Writes the average ILM and NILM energy (kBTU per hour) of each room for every (day of week, hour) of the
    average week to output/{building}_average_hourly_consumption.txt, the same values that are plotted.
    """
    DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    lines = []
    for (day, hr), rooms_data in sorted(room_consumption.items()):
        lines.append(f"--- {DAYS_OF_WEEK[day]} {hr:02d}:00 ---")
        for rm, energy in sorted(rooms_data.items(), key = lambda kv: buildings.room_sort_key(kv[0])):
            lines.append(f" Room {rm}: ILM = {energy[0]:.2f} kBTU, NILM = {energy[1]:.2f} kBTU")

    out_path = Path(f'output/{building}_average_hourly_consumption.txt')
    out_path.parent.mkdir(parents = True, exist_ok = True)
    out_path.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    print(f"Wrote room profiles to {out_path}")


def _vavs_to_rooms(vav_data, building_vavs: Set[VAV], target_rooms: Set[str]) -> dict:
    """
    Splits each VAV's [ILM, NILM] energy across its rooms by room weight and sums the shares for rooms served
    by several VAVs. Returns {room_num: [ilm, nilm]} for the rooms in target_rooms.
    """
    rooms_energy = defaultdict(lambda: [0.0, 0.0])

    for vav in building_vavs:
        if vav.vav_name not in vav_data:
            continue

        ilm_energy, nilm_energy = vav_data[vav.vav_name]
        total_weight = sum(r.get_room_weight() for r in vav.rooms_associated)

        for room in vav.rooms_associated:
            if room.room_num in target_rooms:
                room_share = room.get_room_weight() / total_weight
                rooms_energy[room.room_num][0] += (ilm_energy * room_share)
                rooms_energy[room.room_num][1] += (nilm_energy * room_share)

    return rooms_energy


def _write_daily_csv(daily: dict, day_info: dict, building_vavs: Set[VAV], target_rooms: Set[str],
                     building: str) -> None:
    """
    Writes a single CSV with one row per date in the data. Each row gives the desired rooms' ILM and NILM
    energy (kBTU) summed over the 15-minute intervals present for that date with how many intervals those were
    and the first and last interval time.
    """
    mapped_rooms = {room.room_num for vav in building_vavs for room in vav.rooms_associated}
    room_cols = sorted(target_rooms & mapped_rooms, key = buildings.room_sort_key)
    if not room_cols:
        print("No matching rooms to write")
        return

    table_rows = []
    for day, vav_data in sorted(daily.items()):
        count, first, last = day_info[day]
        rooms_energy = _vavs_to_rooms(vav_data, building_vavs, set(room_cols))
        row = {'Date': day.isoformat(), 'intervals': count,
               'first_time': first.strftime('%H:%M'), 'last_time': last.strftime('%H:%M')}
        for rm in room_cols:
            ilm_energy, nilm_energy = rooms_energy[rm] if rm in rooms_energy else (0.0, 0.0)
            row[f"Room {rm} ILM (kBTU)"] = round(ilm_energy, 4)
            row[f"Room {rm} NILM (kBTU)"] = round(nilm_energy, 4)
        table_rows.append(row)

    out_path = Path(f'output/{building}_room_daily_consumption.csv')
    out_path.parent.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(table_rows).to_csv(out_path, index = False)
    partial = sum(1 for r in table_rows if r['intervals'] != 96)
    print(f"Wrote {out_path} ({len(table_rows)} days x {len(room_cols)} rooms, ILM and NILM columns, kBTU; "
          f"{partial} day(s) without the entire day's 96 intervals)")


def main(building: str, purpose: int, rooms: None | str) -> None:
    # Get Building VAV information
    vav_room_mapping = Path(f'data/{building}_vav_room_mapping.csv')
    hvac_file = Path(f'data/{building}2025_combined.csv')

    if not vav_room_mapping.is_file():
        print(f'Mising {vav_room_mapping} file')
        return
    if not hvac_file.is_file():
        print(f'Missing {hvac_file} file')
        return

    try:
        building_vavs = buildings.get_building_vavs(vav_room_mapping)
    except ValueError as e:
        print(e)
        return

    target_rooms = _get_rooms_to_predict(building_vavs, rooms)

    accumulator = models.EnergyAccumulator()
    daily = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0]))  # daily[date][vav_name] = [ilm, nilm]
    day_info = {}                                 # day_info[date] = [intervals counted, first time, last time]

    generator = models.hvac_record_generator(hvac_file, building_vavs)
    for record in generator:
        ilm_results = models.predict_ilm_consumption(building_vavs, record)
        nilm_results = models.predict_nilm_consumption(building_vavs, record)

        accumulator.add_record(record, ilm_results, nilm_results)

        if purpose == 2:
            day = record.timestamp.date()
            info = day_info.setdefault(day, [0, record.timestamp, record.timestamp])
            info[0] += 1
            info[1], info[2] = min(info[1], record.timestamp), max(info[2], record.timestamp)
            for vav, energy in ilm_results.items():
                daily[day][vav.vav_name][0] += energy
            for vav, energy in nilm_results.items():
                daily[day][vav.vav_name][1] += energy

    if purpose == 2:
        _write_daily_csv(daily, day_info, building_vavs, target_rooms, building)
        return

    room_consumption = {time_key: _vavs_to_rooms(vav_data, building_vavs, target_rooms)
                        for time_key, vav_data in accumulator.average().items()}
    if purpose == 0:
        _write_room_energy_consumption(room_consumption, building)
    else:
        print("Generating plots...")
        plot.generate_plots(room_consumption, building_vavs, building)


if __name__ == '__main__':
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        BUILDING = sys.argv[1]
        PURPOSE = sys.argv[2]
        ROOMS = sys.argv[3] if len(sys.argv) == 4 else None

    elif len(sys.argv) != 1:
        print("Usage: main.py building purpose rooms")
        exit(1)

    purposes = {'predict': 0, 'plot': 1, 'summarize': 2}
    if PURPOSE not in purposes:
        print("PURPOSE must be predict, plot, or summarize")
        exit(1)
    PURPOSE = purposes[PURPOSE]

    main(BUILDING, PURPOSE, ROOMS)