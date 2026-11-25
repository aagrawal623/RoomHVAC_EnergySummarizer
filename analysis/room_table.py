"""
STEP 2: Build one table per energy method with one row per "profile" (a room, or a group of rooms that
share exactly the same VAVs), its labels, and the target: daytime energy per square foot per weekday.

Why "profiles"? In SBSG one VAV often serves 2-6 rooms. The teammate's file splits that VAV's energy between
its rooms (mostly equally), so the per-room numbers inside a group are not independent measurements.
We add the group's rooms back together: total energy / total area. That number is right no matter how the
VAV's energy was split.

Put this file in:  <repo>/analysis/room_table.py

In:   output/analysis/energy_daytime_<METHOD>.pkl   (from step 1)
      data/<building>_vav_room_mapping.csv
Out:  output/analysis/profiles_<METHOD>.csv          (one row per profile: labels + target)

Run (from the repo main folder):  python3 analysis/room_table.py
"""
import re
import pandas as pd
from config import RAW, PROCESSED, FILES, METHODS, TYPE_MAP, DBH_RENAME

# ---------------------------------------------------------------------------
# 1. One row per room with its labels (the mapping file has one row per VAV-room pair)
# ---------------------------------------------------------------------------
rooms = []
for building, (_, map_file) in FILES.items():
    m = pd.read_csv(RAW / map_file)
    room_col = "room_num" if "room_num" in m.columns else "room_number"   # the two file versions differ
    m[room_col] = m[room_col].astype(str)
    if building == "DBH":
        m[room_col] = m[room_col].replace(DBH_RENAME)
    for room, g in m.groupby(room_col):
        floor = re.search(r"Floor-(\d+)", str(g["floorRef"].iloc[0]))
        rooms.append({
            "room_id": f"{building}|{room}",
            "building": building,
            "room": room,
            "type_raw": g["room_type"].iloc[0],
            "sq_ft": g["room_size_sf"].dropna().iloc[0] if g["room_size_sf"].notna().any() else None,
            "size_fine": g["room_bucket"].iloc[0],
            "floor": int(floor.group(1)) if floor else None,
            "ahu": str(g["ahuRef"].iloc[0]).split()[-1],        # "Bren Hall AH-01" -> "AH-01"
            "vavs": "|".join(sorted(g["vavName"].unique())),     # the set of VAVs serving this room
        })
rooms = pd.DataFrame(rooms)
rooms["room_type"] = rooms["type_raw"].map(TYPE_MAP).fillna(rooms["type_raw"])
print(f"Rooms in the mapping files: {len(rooms)}  {rooms.building.value_counts().to_dict()}")


def largest(group, col):
    """Label of the largest room in a group (used when rooms on one VAV have different labels)."""
    return group.loc[group["sq_ft"].idxmax(), col]


# ---------------------------------------------------------------------------
# 2. For each method: attach energy, drop unusable rooms, merge into profiles, compute the target
# ---------------------------------------------------------------------------
for method in METHODS:
    print(f"\n{'=' * 60}\n{method}")
    energy = pd.read_pickle(PROCESSED / f"energy_daytime_{method}.pkl")
    n_days = energy.index.normalize().nunique()

    r = rooms.copy()
    in_energy_not_map = sorted(set(energy.columns) - set(r["room_id"]))
    print(f"  rooms in the energy file but not in the mapping (ignored): {len(in_energy_not_map)} {in_energy_not_map}")
    no_size = r["sq_ft"].isna()
    no_energy = ~r["room_id"].isin(energy.columns)
    print(f"  dropped, no square footage: {no_size.sum()} {sorted(r.loc[no_size, 'room_id'])}")
    print(f"  dropped, not in the energy file: {(no_energy & ~no_size).sum()} {sorted(r.loc[no_energy & ~no_size, 'room_id'])}")
    r = r[~no_size & ~no_energy].copy()
    r["daytime_kbtu"] = r["room_id"].map(energy.sum())          # total over all 53 days
    zero = r["daytime_kbtu"] <= 0
    print(f"  dropped, zero energy: {zero.sum()} {sorted(r.loc[zero, 'room_id'])}")
    r = r[~zero]

    profiles = []
    for (building, vavs), g in r.groupby(["building", "vavs"]):
        profiles.append({
            "profile_id": f"{building}|" + "+".join(sorted(g["room"])),
            "building": building,
            "rooms": ";".join(sorted(g["room"])),
            "n_rooms": len(g),
            "room_type": largest(g, "room_type"),
            "mixed_types": g["room_type"].nunique() > 1,
            "size_fine": largest(g, "size_fine"),
            "floor": largest(g, "floor"),
            "ahu": largest(g, "ahu"),
            "n_vavs": len(vavs.split("|")),
            "sq_ft": g["sq_ft"].sum(),
            "daytime_kbtu": g["daytime_kbtu"].sum(),
        })
    p = pd.DataFrame(profiles)
    p["size_coarse"] = p["size_fine"].str[0]                     # S1 -> S, M2 -> M, L1 -> L
    p.loc[p["size_fine"].str.startswith("XL"), "size_coarse"] = "XL"

    # The target: daytime energy per square foot, per weekday
    p["kbtu_per_sqft_day"] = p["daytime_kbtu"] / p["sq_ft"] / n_days

    print(f"\n  Profiles: {len(p)}  {p.building.value_counts().to_dict()}   "
          f"(profiles mixing room types: {p.mixed_types.sum()})")
    print("  Target, daytime kBTU per sq ft per weekday:")
    print(p.groupby("building")["kbtu_per_sqft_day"].describe()[["count", "min", "25%", "50%", "75%", "max"]].round(3).to_string())
    out = PROCESSED / f"profiles_{method}.csv"
    p.to_csv(out, index=False)
    print(f"  saved {out}")

# Room types per building (same for both methods)
print("\nProfiles per room type:")
print(pd.crosstab(p.room_type, p.building).to_string())

# Check yourself:
#   - Open profiles_ILM.csv and find an SBSG profile with n_rooms > 1. Why is it fine to merge those rooms?
#   - Which building uses more energy per sq ft in the daytime, by ILM? By NILM? Do the two methods agree?
#   - Which rooms are in the energy file but not in the mapping, and why can't we use them?