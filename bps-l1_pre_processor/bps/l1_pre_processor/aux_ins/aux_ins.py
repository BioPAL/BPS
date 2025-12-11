# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX INS
-------
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from bps.common import (
    AcquisitionMode,
    MissionPhaseID,
    Polarization,
    Swath,
    retrieve_aux_product_data_content,
)

DriftNormalizationID = tuple[Swath, MissionPhaseID, Polarization]

EXPECTED_NUMBER_OF_CHIRP_FILES = 6


class InvalidAuxInsProduct(RuntimeError):
    """Raised when an Aux Ins product is invalid"""


def _retrieve_acquisition_mode_from_file(file: Path) -> AcquisitionMode:
    """Retrieve the acquisition mode from the filename"""
    swath_str, mission_phase_id_str = file.stem[-6:].split("_")
    swath = Swath(swath_str.upper())
    mission_phase_id = MissionPhaseID(mission_phase_id_str)
    return (swath, mission_phase_id)


@dataclass
class AuxInsProduct:
    """Auxiliary instrument information"""

    instrument_file: Path
    antenna_pattern_file: Path
    chirp_files: dict[AcquisitionMode, Path]

    @classmethod
    def from_product(cls, aux_ins_product: Path) -> Self:
        """Retrieve content of aux ins product and fill the corresponding structure

        Parameters
        ----------
        aux_ins_product : Path
            path to the aux ins product

        Returns
        -------
        AuxInsContent
            structure containing paths to the data contained in the auxins

        Raises
        ------
        InvalidAuxInsProduct
            In case of unexpected or missing files
        """
        aux_ins_raw_content = retrieve_aux_product_data_content(aux_ins_product)

        antenna_pattern_file: Path | None = None
        instrument_file: Path | None = None
        chirp_files: dict[AcquisitionMode, Path] = {}

        for file in aux_ins_raw_content:
            if "antenna_patterns" in file.stem:
                if antenna_pattern_file is not None:
                    raise InvalidAuxInsProduct(aux_ins_product)

                antenna_pattern_file = file

            elif "chirp_replica" in file.stem:
                chirp_id = _retrieve_acquisition_mode_from_file(file)
                if chirp_id in chirp_files:
                    raise InvalidAuxInsProduct(aux_ins_product)

                chirp_files[chirp_id] = file

                # Commissioning phase uses tomographic chirp
                swath, phase_id = chirp_id
                if phase_id == MissionPhaseID.TOMOGRAPHIC:
                    chirp_files[(swath, MissionPhaseID.COMMISSIONING)] = file

            elif "ins.xml" in file.name:
                if instrument_file is not None:
                    raise InvalidAuxInsProduct(aux_ins_product)

                instrument_file = file
            else:
                raise InvalidAuxInsProduct(aux_ins_product)

        if (
            antenna_pattern_file is None
            or instrument_file is None
            or len(set(chirp_files.values())) != EXPECTED_NUMBER_OF_CHIRP_FILES
        ):
            raise InvalidAuxInsProduct(aux_ins_product)

        return cls(
            instrument_file=instrument_file,
            antenna_pattern_file=antenna_pattern_file,
            chirp_files=chirp_files,
        )


@dataclass
class IntCalParameters:
    """Internal calibration parameters"""

    polarization: Polarization

    reference_drift: complex

    tx_power_tracking: complex

    noise_power: float


@dataclass
class AcqModeParameters:
    """Acquisition mode based parameters"""

    acquisition_mode: AcquisitionMode
    swath: Swath
    int_cal_params: dict[Polarization, IntCalParameters]


@dataclass
class AuxInsParameters:
    """Auxiliary instrument parameters"""

    radar_frequency: float
    """Carrier frequency [Hz]"""

    roll_bias: float
    """Roll bias [deg]"""

    tx_start_time: float
    """TX start time (T9) [s]"""

    calibration_signals_swp: float
    """Calibration signals Sampling Window Position (SWP) [s]"""

    parameters: dict[AcquisitionMode, AcqModeParameters]
