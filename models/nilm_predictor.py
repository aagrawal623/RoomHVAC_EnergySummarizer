"""
Non-Intrusive Load Monitoring (NILM) Predictor.

Calculates whole-building energy usage and disaggregates it  to the VAV level proportionally
using a calculated thermal signal.
"""

from typing import Dict, Iterable
from buildings.building_data import VAV
from models.hvac import HVACRecord, KWH_TO_KBTU, TONS_TO_KBTU_HR, WATER_HEAT_CONSTANT


def predict_nilm_consumption(vavs: Iterable[VAV], record: HVACRecord) -> Dict[VAV, float]:
    """
    Disaggregates building-level metered energy to individual VAVs and returns a mapping of
    each VAV to its disaggregated energy in kBTU.
    """
    result = {vav: 0.0 for vav in vavs}

    signals_cool: Dict[VAV, float] = {}
    signals_heat: Dict[VAV, float] = {}
    signals_elec: Dict[VAV, float] = {}

    total_sig_cool = 0.0
    total_sig_heat = 0.0
    total_sig_elec = 0.0

    h = record.interval_hours
    total_hhw_metered = 0.0
    has_hhw_meter = False

    # Calculate each VAV Thermal Signal
    for vav in vavs:
        vav_dat = record.vav_data.get(vav.vav_name, {})
        ahu_dat = record.ahu_data.get(vav.ahu_ref, {})

        saf = vav_dat.get('saf', 0.0)
        sat = vav_dat.get('sat', 0.0)
        zt = vav_dat.get('zt', 0.0)
        ahu_sat = ahu_dat.get('sat', 0.0)
        hhw_power = vav_dat.get('hhw', 0.0)

        d_cool = max(zt - sat, 0) if (zt > 0 and sat > 0) else 0
        d_reheat = max(sat - ahu_sat, 0) if (sat > 0 and ahu_sat > 0) else 0

        # Cooling Signal
        sig_c = saf * d_cool
        signals_cool[vav] = sig_c
        total_sig_cool += sig_c

        # Heating Signal
        sig_h = saf * d_reheat
        signals_heat[vav] = sig_h
        total_sig_heat += sig_h

        # Electric Signal
        sig_e = saf
        signals_elec[vav] = sig_e
        total_sig_elec += sig_e

        if hhw_power > 0:
            has_hhw_meter = True
            total_hhw_metered += (hhw_power * h)

    # Find whole building energy
    bldg = record.building_data

    # Electric Stream
    building_electric_kbtu = bldg.get('electric_kw', 0.0) * h * KWH_TO_KBTU

    # Cooling Stream
    building_cooling_kbtu = 0.0
    tons = bldg.get('chw_tons', 0.0)
    if tons > 0:
        building_cooling_kbtu = tons * TONS_TO_KBTU_HR * h
    else:
        chw_flow = bldg.get('chw_flow', 0.0)
        chw_ret = bldg.get('chw_return', 0.0)
        chw_sup = bldg.get('chw_supply', 0.0)
        if chw_flow > 0 and chw_ret > chw_sup:
            building_cooling_kbtu = WATER_HEAT_CONSTANT * chw_flow * (chw_ret - chw_sup) * h / 1000

    # Heating Streams
    building_heating_kbtu = total_hhw_metered if has_hhw_meter else 0.0

    # Disaggregate Streams independently and sum streams for the VAV's final NILM prediction
    for vav in vavs:
        vav_cool = 0.0
        vav_heat = 0.0
        vav_elec = 0.0

        if total_sig_cool > 0:
            vav_cool = (signals_cool[vav] / total_sig_cool) * building_cooling_kbtu

        if total_sig_heat > 0:
            vav_heat = (signals_heat[vav] / total_sig_heat) * building_heating_kbtu

        if total_sig_elec > 0:
            vav_elec = (signals_elec[vav] / total_sig_elec) * building_electric_kbtu

        result[vav] = vav_cool + vav_heat + vav_elec

    return result