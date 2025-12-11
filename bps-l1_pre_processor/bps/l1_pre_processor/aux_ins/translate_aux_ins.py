# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX INS Translation
-------------------
"""

from bps.common import AcquisitionMode, MissionPhaseID, Polarization, Swath
from bps.common.io import translate_common
from bps.l1_pre_processor.aux_ins.aux_ins import (
    AcqModeParameters,
    AuxInsParameters,
    IntCalParameters,
)
from bps.l1_pre_processor.io import aux_ins_models


def _translate_acquisition_mode_string(
    acquisition_mode: aux_ins_models.AcquisitionModeIdtype,
) -> AcquisitionMode:
    if acquisition_mode == aux_ins_models.AcquisitionModeIdtype.RX_ONLY:
        swath, mission_phase = Swath.RO, None
    else:
        swath, mission_phase = acquisition_mode.value.split()
        mission_phase = MissionPhaseID(mission_phase.lower())
        swath = Swath(swath)

    return swath, mission_phase


def _translate_acquisition_mode_parameters_from_model(
    model: aux_ins_models.AcquisitionModeType,
) -> AcqModeParameters:
    assert model.acquisition_mode is not None
    assert model.swath is not None
    assert model.int_cal_parameters_list is not None

    swath = Swath(model.swath.value)
    acquisition_mode = _translate_acquisition_mode_string(model.acquisition_mode)

    int_cal_params: dict[Polarization, IntCalParameters] = {}
    for param in model.int_cal_parameters_list.int_cal_parameters:
        assert param.polarisation is not None
        int_cal_params[Polarization(param.polarisation.value)] = _translate_int_cal_parameters(param)

    return AcqModeParameters(
        acquisition_mode=acquisition_mode,
        swath=swath,
        int_cal_params=int_cal_params,
    )


def _translate_int_cal_parameters(
    model: aux_ins_models.IntCalParametersType,
) -> IntCalParameters:
    assert model.polarisation is not None
    assert model.reference_drift is not None
    assert model.tx_power_tracking is not None
    assert model.noise_power is not None and model.noise_power.value is not None
    return IntCalParameters(
        polarization=Polarization(model.polarisation.value),
        reference_drift=translate_common.translate_complex(model.reference_drift),
        tx_power_tracking=translate_common.translate_complex(model.tx_power_tracking),
        noise_power=model.noise_power.value,
    )


def translate_model_to_aux_ins_parameters(
    model: aux_ins_models.AuxiliaryInstrumentParameters,
) -> AuxInsParameters:
    """Translate aux ins to the corresponding structure"""
    assert model.radar_frequency is not None and model.radar_frequency.value is not None
    assert model.roll_bias is not None and model.roll_bias.value is not None
    assert model.tx_start_time is not None and model.tx_start_time.value is not None
    assert model.calibration_signals_swp is not None and model.calibration_signals_swp.value is not None
    assert model.acquisition_mode_list is not None
    assert model.raw_data_decoding_parameters is not None

    assert model.acquisition_mode_list.count is not None

    acq_mode_parameters: dict[AcquisitionMode, AcqModeParameters] = {}
    for acq_mode in model.acquisition_mode_list.acquisition_mode:
        assert acq_mode.acquisition_mode is not None
        acquisition_mode = _translate_acquisition_mode_string(acq_mode.acquisition_mode)
        acq_mode_parameters[acquisition_mode] = _translate_acquisition_mode_parameters_from_model(acq_mode)

        # Commissioning phase uses tomographic parameters
        (swath, phase_id) = acquisition_mode
        if phase_id == MissionPhaseID.TOMOGRAPHIC:
            acq_mode_parameters[(swath, MissionPhaseID.COMMISSIONING)] = acq_mode_parameters[acquisition_mode]

    return AuxInsParameters(
        radar_frequency=translate_common.translate_float_with_unit(model.radar_frequency),
        roll_bias=translate_common.translate_float_with_unit(model.roll_bias),
        tx_start_time=translate_common.translate_float_with_unit(model.tx_start_time),
        calibration_signals_swp=translate_common.translate_float_with_unit(model.calibration_signals_swp),
        parameters=acq_mode_parameters,
    )
