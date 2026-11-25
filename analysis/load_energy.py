"""
STEP 1: Load the 15-minute room energy, fix the time zone, and keep only weekday daytime readings.
Runs once per energy method in config.METHODS (ILM and NILM).

Put this file in:  <repo>/analysis/load_energy.py

In:   data/<building>_room_15min_consumption.csv   (one column per room per method, e.g. "Room 1100 ILM (kBTU)")
Out:  output/analysis/energy_daytime_ILM.pkl, output/analysis/energy_daytime_NILM.pkl
      (rows = 15-min timestamps, columns = "DBH|2045", "SBSG|1203", ...)

Run (from the repo main folder):  python3 analysis/load_energy.py
"""
import pandas as pd
from config import RAW, PROCESSED, FILES, METHODS, LOCAL_TZ, START, END, HOLIDAYS, DAY_START, DAY_END, DBH_RENAME

# 1. Read each building's file once, fix time, keep the right days and hours ----------------
raw = {}
for building, (energy_file, _) in FILES.items():
    df = pd.read_csv(RAW / energy_file)
    print(f"\n=== {building}: raw file {df.shape[0]} rows x {df.shape[1]} columns")

    # The file is in UTC with no zone written, so we tell pandas it is UTC, then convert to Pacific.
    df.index = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    df = df.drop(columns="timestamp")
    print(f"  first/last reading (Pacific): {df.index.min()}  ->  {df.index.max()}")

    # Date window, weekdays only (Mon=0 ... Fri=4), no holidays
    local_date = df.index.tz_localize(None).normalize()
    keep = (local_date >= START) & (local_date <= END) & (df.index.dayofweek < 5) \
        & ~local_date.isin(pd.to_datetime(HOLIDAYS))
    df = df[keep]

    # Daytime window: 7:00 <= time < 22:30
    df = df.between_time(DAY_START, DAY_END, inclusive="left")
    n_days = df.index.normalize().nunique()
    print(f"  after window: {n_days} weekdays, {len(df)} readings ({len(df) // n_days} per day)")
    raw[building] = df

# 2. Split out each method, rename columns, join buildings, save ----------------------------
for method in METHODS:
    print(f"\n--- {method} ---")
    parts = []
    for building, df in raw.items():
        # "Room 1100 ILM (kBTU)" -> "1100". The leading space stops " ILM" from also matching "NILM".
        cols = [c for c in df.columns if c.endswith(f" {method} (kBTU)")]
        part = df[cols].copy()
        part.columns = [c.replace("Room ", "").replace(f" {method} (kBTU)", "") for c in cols]
        if building == "DBH":
            part = part.rename(columns=DBH_RENAME)
        part.columns = [f"{building}|{c}" for c in part.columns]
        print(f"  {building}: {part.shape[1]} rooms, largest 15-min value {part.values.max():.1f} kBTU")
        parts.append(part)

    energy = pd.concat(parts, axis=1)
    print(f"  combined table: {energy.shape[0]} timestamps x {energy.shape[1]} rooms, "
          f"missing values: {int(energy.isna().sum().sum())}")
    print(energy.iloc[:3, :4].round(3))
    out = PROCESSED / f"energy_daytime_{method}.pkl"
    energy.to_pickle(out)
    print(f"  saved {out}")

# Check yourself:
#   - Compare the "largest 15-min value" lines for ILM and NILM. Which one looks suspicious, and in which building?
#   - Why does the script add a space before the method name when picking columns? (Try it without the space in your head.)
