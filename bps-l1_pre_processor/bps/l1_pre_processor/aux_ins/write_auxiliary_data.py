# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Write auxiliary data
--------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bps.common import AcquisitionMode, bps_logger
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsProduct
from bps.l1_pre_processor.aux_ins.chirp import (
    transcode_input_chirp_replica_to_product_folder,
)
from bps.l1_pre_processor.aux_ins.drift_normalization import (
    normalize_drift_product,
    retrieve_reference_drifts,
)
from bps.l1_pre_processor.aux_ins.estimated_noise import (
    retrieve_noise_power,
    write_estimated_noise_product,
)
from bps.l1_pre_processor.aux_ins.excitation_coefficients import (
    retrieve_tx_power_tracking,
    write_excitation_coefficient_product,
)
from bps.l1_pre_processor.aux_ins.pattern import (
    transcode_input_antenna_patterns_to_product_folder,
)
from bps.l1_pre_processor.io.parsing import parse_aux_ins


@dataclass
class RequestedProducts:
    """Products to write/update"""

    chirp_replica_product: Path | None
    """If set, chirp replica is extracted from aux ins to given path"""

    amp_phase_drift_product: Path | None
    """If set, given drift are normalized with reference drift from aux ins"""

    tx_power_tracking_product: Path | None
    """If set, tx power tracking is extracted from aux ins to given path"""

    est_noise_product: Path | None
    """If set, est noise is extracted from aux ins to given path"""

    @dataclass
    class AntennaPatterns:
        """Antenna patterns"""

        ant_d1_h_product: Path
        ant_d1_v_product: Path
        ant_d2_h_product: Path
        ant_d2_v_product: Path

    antenna_patterns: AntennaPatterns | None
    """If set, antenna patterns are extracted from aux ins to given paths"""


def write_core_auxiliary_input_products(
    aux_ins_product_path: Path,
    requested_products: RequestedProducts,
    extracted_raw_product: Path,
    acquisition_mode: AcquisitionMode,
):
    """Prepare all the required auxiliary input products for the core processor

    * Chirp (if required)
    * Update drift product (normalization, when available)
    * TX power tracking (if required)
    * Noise power (if required)
    """

    aux_ins_product = AuxInsProduct.from_product(aux_ins_product_path)

    if requested_products.antenna_patterns is not None:
        transcode_input_antenna_patterns_to_product_folder(
            aux_ins_product.antenna_pattern_file,
            ant_d1_h_product=requested_products.antenna_patterns.ant_d1_h_product,
            ant_d1_v_product=requested_products.antenna_patterns.ant_d1_v_product,
            ant_d2_h_product=requested_products.antenna_patterns.ant_d2_h_product,
            ant_d2_v_product=requested_products.antenna_patterns.ant_d2_v_product,
        )

    if requested_products.chirp_replica_product is not None:
        transcode_input_chirp_replica_to_product_folder(
            aux_ins_product.chirp_files,
            requested_products.chirp_replica_product,
            acquisition_mode,
        )

    aux_ins_parsing_required = (
        requested_products.amp_phase_drift_product is not None
        or requested_products.tx_power_tracking_product is not None
        or requested_products.est_noise_product is not None
    )

    aux_ins_parameters = (
        parse_aux_ins(aux_ins_product.instrument_file.read_text(encoding="utf-8")) if aux_ins_parsing_required else None
    )

    if requested_products.amp_phase_drift_product is not None:
        assert aux_ins_parameters is not None

        reference_drifts = retrieve_reference_drifts(aux_ins_parameters, acquisition_mode)

        normalize_drift_product(
            requested_products.amp_phase_drift_product.absolute(),
            reference_drifts,
        )

    if requested_products.tx_power_tracking_product is not None:
        assert aux_ins_parameters is not None

        bps_logger.info("Antenna excitation coefficients retrieved from AUX INS product")

        reference_coefficients = retrieve_tx_power_tracking(aux_ins_parameters, acquisition_mode)

        write_excitation_coefficient_product(
            extracted_raw_product,
            requested_products.tx_power_tracking_product,
            reference_coefficients,
        )

    if requested_products.est_noise_product:
        assert aux_ins_parameters is not None

        bps_logger.info("Noise power retrieved from AUX INS product")

        reference_noise_power = retrieve_noise_power(aux_ins_parameters, acquisition_mode)

        write_estimated_noise_product(
            extracted_raw_product,
            requested_products.est_noise_product,
            reference_noise_power,
        )
