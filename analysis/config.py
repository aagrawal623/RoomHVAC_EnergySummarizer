"""
Settings shared by every step of the analysis. Change things here, not inside the scripts.

Put this file in:  <repo>/analysis/config.py
You never run this file directly; the step scripts import it.
"""
from pathlib import Path

# Folder layout. This file lives in <repo>/analysis/, so the repo's main folder is the one above analysis/.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data"                          # the 4 input CSVs go in the repo's data/ folder
PROCESSED = ROOT / "output" / "analysis"     # each step writes its in-between results here
OUTPUTS = ROOT / "output" / "analysis"       # figures and tables for slides
for folder in (PROCESSED, OUTPUTS):
    folder.mkdir(parents=True, exist_ok=True)

# Input files: building code -> (15-min energy file, VAV-room mapping file)
FILES = {
    "DBH":  ("BrenHall_room_15min_consumption.csv", "BrenHall_vav_room_mapping.csv"),
    "SBSG": ("SBSG_room_15min_consumption.csv",     "SBSG_vav_room_mapping.csv"),
}

# Which energy estimates to analyze. Every step runs once per method and names its files with it.
#   ILM  = physics-based, from each VAV's own airflow and temperatures
#   NILM = building meters split between VAVs by airflow (SBSG NILM has very large spikes)
METHODS = ["ILM", "NILM"]

# The energy files are stamped in UTC (they start at 08:00 = midnight Pacific).
# We convert to Pacific so "7:00am" means 7:00am in the building.
LOCAL_TZ = "America/Los_Angeles"

# Date window both buildings cover, weekdays only, holidays removed
START, END = "2025-01-13", "2025-03-28"          # inclusive
HOLIDAYS = ["2025-01-20", "2025-02-17"]          # MLK Day, Presidents' Day

# Mentor's daytime window: 7:00am up to (not including) 10:30pm
DAY_START, DAY_END = "07:00", "22:30"

# Room bucket file uses DBH's room numbers 3024/4024/5024; the energy file calls them 3028/4028/5028
DBH_RENAME = {"3024": "3028", "4024": "4028", "5024": "5028"}

# One shared set of room-type names for both buildings
TYPE_MAP = {
    "office": "Office", "Faculty Office": "Office", "Graduate Office": "Office",
    "student lab": "Lab/Study", "Graduate Study Space": "Lab/Study", "workroom/lab": "Lab/Study",
    "meeting room": "Meeting", "Seminar Room": "Meeting",
    "classroom": "Classroom", "General Assignment Classroom": "Classroom",
    "conference room": "Conference", "Conference Room": "Conference",
    "lecture room": "Lecture", "telecom": "Telecom",
    "subject room": "Subject room", "lobby/reception": "Lobby",
    "food/break": "Food/Break", "corridor": "Corridor", "Support Room": "Support",
}
