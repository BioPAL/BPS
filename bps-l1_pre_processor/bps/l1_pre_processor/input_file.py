# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 PreProcessor input file structure
------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from arepytools.timing.precisedatetime import PreciseDateTime


@dataclass
class L1PreProcessorInputFile:
    """Input file of the L1 PreProcessor app"""

    input_l0s_product: Path
    """Input L0S product"""

    input_aux_orb_file: Path
    """Input auxiliary orbit file"""
    input_aux_att_file: Path
    """Input auxiliary attitude file"""

    input_aux_ins_file: Path
    """Input auxiliary instrument parameters file"""
    input_configuration_file: Path
    """Input auxiliary L1 processing parameters file"""
    input_iersbullettin_file: Path | None
    """Input IERS bullettin file"""

    time_of_interest: tuple[PreciseDateTime, PreciseDateTime] | None
    """Time of interest for smart read"""

    bps_configuration_file: Path
    """BPS common configuration file"""
    bps_log_file: Path
    """BPS log file"""

    output_raw_data_product: Path
    """Output RAW data product"""

    input_l0m_product: Path | None = None
    """Input L0 monitoring product"""

    intermediate_dyn_cal_product: Path | None = None
    """Intermediate DynCal product"""
    intermediate_pgpproduct: Path | None = None
    """Intermediate pgp product"""

    output_per_line_correction_factors_product: Path | None = None
    """Output per line correction factors product"""
    output_chirp_replica_product: Path | None = None
    """Output chirp replica product"""
    output_channel_delays_file: Path | None = None
    """Output channel delays file"""
    output_channel_imbalance_file: Path | None = None
    """Output channel imbalance file"""
    output_tx_power_tracking_product: Path | None = None
    """Output TX power tracking product"""
    output_est_noise_product: Path | None = None
    """Output estimated noise product"""
    output_ssp_headers_file: Path | None = None
    """Output ssp headers file"""
    output_report_file: Path | None = None
    """Output report file"""
