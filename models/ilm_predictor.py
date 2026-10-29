"""
Intrusive Load Monitoring (ILM) Predictor.

Calculates the physics-based energy consumption of each VAV using explicit sensor data
(Airflow, Supply Temps, Zone Temps, and AHU Fans).
"""

from typing import Dict, List, Iterable
from buildings.building_data import VAV
from models.hvac import HVACRecord, SENSIBLE_HEAT_CONSTANT, KWH_TO_KBTU


def predict_ilm_consumption(vavs: Iterable[VAV], record: HVACRecord) -> Dict[VAV, float]:
    """
    Calculates ILM energy consumption for a single timestep based on thermodynamics and returns
    a mapping of each VAV to its calculated energy consumption in kBTU.
    """
    result = {vav: 0.0 for vav in vavs}
    h = record.interval_hours

    # Group VAVs by AHU to calculate system-level metrics
    ahu_vavs_map: Dict[str, List[VAV]] = {}
    for vav in vavs:
        ahu_vavs_map.setdefault(vav.ahu_ref, []).append(vav)

    for ahu_ref, ahu_vavs in ahu_vavs_map.items():
        # Calculate proportional fan share distribution requirements
        system_saf = sum(record.vav_data.get(v.vav_name, {}).get('saf', 0.0) for v in ahu_vavs)
        fan_kbtu = record.ahu_data.get(ahu_ref, {}).get('fan_kw', 0.0) * h * KWH_TO_KBTU
        ahu_sat = record.ahu_data.get(ahu_ref, {}).get('sat', 0.0)

        for vav in ahu_vavs:
            vav_dat = record.vav_data.get(vav.vav_name, {})
            saf = vav_dat.get('saf', 0.0)
            sat = vav_dat.get('sat', 0.0)
            zt = vav_dat.get('zt', 0.0)

            q_cool, q_reheat = 0.0, 0.0

            # Physics-based cooling and reheat derivation
            if saf > 0:
                if zt > sat:
                    q_cool = SENSIBLE_HEAT_CONSTANT * saf * (zt - sat) * h / 1000
                if sat > ahu_sat > 0:
                    q_reheat = SENSIBLE_HEAT_CONSTANT * saf * (sat - ahu_sat) * h / 1000

            # Fan energy attributed to this VAV based on proportional airflow
            saf_share = (saf / system_saf) if system_saf > 0 else 0.0

            result[vav] = q_cool + q_reheat + (saf_share * fan_kbtu)

    return result
