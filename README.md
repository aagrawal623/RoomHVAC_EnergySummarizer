# Room HVAC Energy Summarizer

A program to estimate and plot room energy consumption for a smart building given the building's HVAC data. This is not the predictor
for energy consumption based on room semantics but a tool to analyze the relationship between VAVs, room semantics, and building
HVAC usage.

## Structure

```text
/
├── buildings/                 Building structure
├── data/                      Building data
├── format_data/               Scripts to combine/filter building data
├── models/                    Energy estimation models
│   ├── nilm_predictor.py      NILM estimator
│   ├── ilm_predictor.py       ILM estimator
├── output/                    Generated output directory
├── plot/                      Plotting output
└── main.py                    Executable script 
```

## Getting started
In the data/ subdirectory, for every building that is to be analyzed, ensure two files are present.
If the files are not present or no data/ subdirectory exists, add them respectively. 

The first file should be named: {building_name}_vav_room_mapping.csv.

This file should contain every VAV in the smart building with each room they map to. A VAV can be present on multiple columns if multiple rooms
are with that VAV's zone. It should have the following headers in the first column (example data will be provided): 

```text
floorRef, vavName, ahuRef, equipRef, room_num, room_type, room_capacity, room_bucket, room_size_sf
```

The next file should be named: {building_name}_combined.csv.

This file contains the HVAC data of a smart building.

## Run the Repository

There are two options to run the repository: commandline and editing configurations.

To run the repository, need to determine the following input:
- building_name: the name of the building analyzed
- purpose: the purpose of running the script. Should be one of
  - predict (creates a file named {building_name}_average_hourly_consumption.txt in output/ of rooms average hourly consumption in the week)
  - plot (creates subdirectories in output/ of the relationship between energy usage and some room feature)
  - summarize (creates a csv file of every room's hourly estimated energy usage)
- rooms (optional to specify the specific room to analyze)
  - room_name (the one room to analyze. e.g. 2011)
  - room_type (the room type to analyze. e.g. 'lecture hall')
  - No value given (analyze all rooms)

### Commandline
Run the following in the home directory:

```python3
python3 main.py building_name purpose [rooms]
```

### Editing Configurations
Go inside main.py and edit the configuration variables


## Methodology

To analyze the relationship between energy usage and room semantics, we need to first convert the building data to room energy usage.
The process to estimate is taken in two ways: NILM and ILM

### NILM Energy Estimation
In the building HVAC data, there is data that presents the building energy used for heating, cooling, and electric respectively.
At a high level, these values are taken and disaggregated to each zone (VAV) based on how much air the respective VAV is asking for.
That zone energy calculation is then split across the rooms in the zone.

### ILM Proxy (Physics-based) Energy Estimation
In the building HVAC data, there is data present at each zone in the building (described as a VAV). That zone data is used
to calculate how much energy the zone is taking and split across the rooms in the zone.
