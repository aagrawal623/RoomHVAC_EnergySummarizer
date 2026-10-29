"""
Author: Aadi Agrawal
Purpose: A program to predict/plot room energy consumption for a smart building from HVAC data
"""

import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Set, Union, Optional

import buildings
import models
import plot
from buildings.building_data import VAV


# ─── Configuration ─────────────────────────────────────────────────────────────────────
"""
If chosen to not use the command line, can configure the program using these global variables
"""
BUILDING = 'BrenHall'    # the smart building in which room energy consumption is investigated
PURPOSE = 'summarize'         # 0 = predict, 1 = plot, 2 = summarize
ROOMS = None             # None = all rooms, INT = specific room, STR = room type
# ───────────────────────────────────────────────────────────────────────────────────────


def _get_rooms_to_predict(vavs: Set[VAV], rooms: Union[None, int, str]) -> Set[int]:
    if rooms is None:
        return {room.room_num for vav in vavs for room in vav.rooms_associated}
    if isinstance(rooms, str):
        return {room.room_num for vav in vavs for room in vav.rooms_associated if
                room.room_type == rooms}
    return {rooms}


def _print_room_energy_consumption(room_consumption):
    DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for time_key, rooms_data in sorted(room_consumption.items()):
        day, hr = time_key
        print(f"--- {DAYS_OF_WEEK[day]} {hr:02d}:00 ---")
        for rm, energy in sorted(rooms_data.items()):
            print(f" Room {rm}: ILM = {energy[0]:.2f} kBTU, NILM = {energy[1]:.2f} kBTU")


def _vavs_to_rooms(vav_data, building_vavs: Set[VAV], target_rooms: Set[int]) -> dict:
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


def _write_daily_csvs(daily: dict, building_vavs: Set[VAV], target_rooms: Set[int],
                      building: str) -> None:
    """
    Writes a single CSV where each row gives the desired room ILM and NILM energy consumption for a
    date.
    """
    mapped_rooms = {room.room_num for vav in building_vavs for room in vav.rooms_associated}
    room_cols = sorted(target_rooms & mapped_rooms)
    if not room_cols:
        print("No matching rooms to write")
        return

    daily_rooms = {day: _vavs_to_rooms(vav_data, building_vavs, set(room_cols))
                   for day, vav_data in sorted(daily.items())}

    table_rows = []
    for day, rooms_energy in daily_rooms.items():
        row = {'date': day.isoformat()}
        for rm in room_cols:
            ilm_energy, nilm_energy = rooms_energy[rm] if rm in rooms_energy else (0.0, 0.0)
            row[f"Room {rm} ILM (kBTU)"] = round(ilm_energy, 4)
            row[f"Room {rm} NILM (kBTU)"] = round(nilm_energy, 4)
        table_rows.append(row)

    out_path = Path(f'data/{building}_room_daily_consumption.csv')
    pd.DataFrame(table_rows).to_csv(out_path, index = False)
    print(
        f"Wrote {out_path} ({len(table_rows)} days x {len(room_cols)} rooms, ILM and NILM columns, kBTU)")


def main(building: str, purpose: int, rooms: None | int | str) -> None:
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
    prefix = "Bren Hall" if building in ("Bren Hall", "BrenHall") else building

    generator = models.hvac_record_generator(hvac_file, building_vavs, prefix = prefix)
    for record in generator:
        ilm_results = models.predict_ilm_consumption(building_vavs, record)
        nilm_results = models.predict_nilm_consumption(building_vavs, record)

        accumulator.add_record(record, ilm_results, nilm_results)

        day = record.timestamp.date()
        for vav, energy in ilm_results.items():
            daily[day][vav.vav_name][0] += energy
        for vav, energy in nilm_results.items():
            daily[day][vav.vav_name][1] += energy

    if purpose == 2:
        _write_daily_csvs(daily, building_vavs, target_rooms, building)
        return

    room_consumption = {time_key: _vavs_to_rooms(vav_data, building_vavs, target_rooms)
                        for time_key, vav_data in accumulator.average().items()}

    if purpose == 0:
        _print_room_energy_consumption(room_consumption)
    else:
        print("Generating plots...")
        plot.generate_plots(room_consumption, building_vavs)


if __name__ == '__main__':
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        BUILDING = sys.argv[1]
        PURPOSE = sys.argv[2]
        try:
            ROOMS = int(sys.argv[3]) if len(sys.argv) == 4 else None
        except ValueError:
            ROOMS = sys.argv[3]

    elif len(sys.argv) != 1:
        print("Usage: main.py building purpose rooms")
        exit(1)

    purposes = {'predict': 0, 'plot': 1, 'summarize': 2}
    if PURPOSE not in purposes:
        print("PURPOSE must be predict, plot, or summarize")
        exit(1)
    PURPOSE = purposes[PURPOSE]

    main(BUILDING, PURPOSE, ROOMS)
