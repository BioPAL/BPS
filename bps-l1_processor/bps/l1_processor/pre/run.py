# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 pre processor run
------------------------
"""

from __future__ import annotations

from pathlib import Path

from arepyextras.runner import Environment
from arepytools.io import open_product_folder
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import (
    AcquisitionMode,
    bps_logger,
    retrieve_aux_product_data_single_content,
)
from bps.common.decorators import log_elapsed_time
from bps.common.runner_helper import run_application
from bps.l1_pre_processor.attitude_utils import repair_attitude_if_needed
from bps.l1_pre_processor.aux_ins import write_auxiliary_data
from bps.l1_pre_processor.interface import (
    L1PreProcessorInterface,
    write_l1_preproc_configuration_file,
    write_l1_preproc_input_file,
)
from bps.l1_processor.pre.interface import (
    L1PreProcessorInterfaceFiles,
    fill_l1_preprocessor_configuration_file,
    fill_l1_preprocessor_input_file,
)
from bps.l1_processor.processor_interface.aux_pp1 import (
    AuxProcessingParametersL1,
    ChirpSource,
)
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1RXOnlyProducts,
    L1StripmapProducts,
)
from bps.l1_processor.settings.l1_binaries import BPS_L1PREPROC_EXE_NAME
from bps.l1_processor.settings.l1_intermediates import L1PreProcessorOutputProducts
from bps.transcoder.auxiliaryfiles.aux_attitude import write_attitude_file


@log_elapsed_time("L1PreProcessor")
def run_l1_pre_processing(
    job_order: L1JobOrder,
    iers_bulletin_file: Path,
    pre_processor_files: L1PreProcessorInterfaceFiles,
    bps_conf_file: Path,
    bps_logger_file: Path,
    l1_pre_processor_outputs: L1PreProcessorOutputProducts,
    aux_pp1: AuxProcessingParametersL1,
    acquisition_mode: AcquisitionMode,
    data_interval: tuple[PreciseDateTime, PreciseDateTime],
    env: Environment,
):
    """Execute L1 pre processing"""
    export_ssp_headers_file = True

    repaired_attitude = repair_attitude_if_needed(
        attitude_xml_file=retrieve_aux_product_data_single_content(job_order.auxiliary_files.attitude),
        processing_interval=job_order.processor_configuration.azimuth_interval or data_interval,
    )
    repaired_attitude_file = None
    if repaired_attitude is not None:
        write_attitude_file(l1_pre_processor_outputs.repaired_attitude, repaired_attitude)
        repaired_attitude_file = l1_pre_processor_outputs.repaired_attitude
        bps_logger.warning(f"Repaired attitude file written to {repaired_attitude_file}")

    pre_processor_interface = L1PreProcessorInterface(
        input_file=fill_l1_preprocessor_input_file(
            job_order=job_order,
            iers_bulletin_file=iers_bulletin_file.absolute(),
            input_conf_file=pre_processor_files.configuration_file.absolute(),
            bps_configuration_file=bps_conf_file.absolute(),
            bps_log_file=bps_logger_file.absolute(),
            output_raw_data_product=l1_pre_processor_outputs.extracted_raw_product.absolute(),
            intermediate_dyn_cal_product=l1_pre_processor_outputs.extracted_dyncal_product.absolute(),
            intermediate_pgpproduct=l1_pre_processor_outputs.pgp_product.absolute(),
            output_per_line_correction_factors_product=l1_pre_processor_outputs.amp_phase_drift_product.absolute(),
            output_chirp_replica_product=l1_pre_processor_outputs.chirp_replica_product.absolute(),
            output_channel_delays_file=l1_pre_processor_outputs.internal_delays_file.absolute(),
            output_channel_imbalance_file=l1_pre_processor_outputs.channel_imbalance_file.absolute(),
            output_tx_power_tracking_product=l1_pre_processor_outputs.tx_power_tracking_product.absolute(),
            output_est_noise_product=l1_pre_processor_outputs.est_noise_product.absolute(),
            output_ssp_headers_file=(
                l1_pre_processor_outputs.ssp_headers_file.absolute() if export_ssp_headers_file else None
            ),
            output_report_file=l1_pre_processor_outputs.report_file,
            repaired_attitude_file=repaired_attitude_file,
        ),
        conf=fill_l1_preprocessor_configuration_file(aux_pp1_obj=aux_pp1),
    )

    write_l1_preproc_input_file(pre_processor_interface.input_file, pre_processor_files.input_file)
    write_l1_preproc_configuration_file(pre_processor_interface.conf, pre_processor_files.configuration_file)

    run_application(
        env,
        BPS_L1PREPROC_EXE_NAME,
        str(pre_processor_files.input_file.absolute()),
        1,
    )

    if isinstance(job_order.io_products, L1RXOnlyProducts):
        return

    write_core_auxiliary_input_products(
        aux_pp1,
        job_order,
        l1_pre_processor_outputs,
        acquisition_mode,
    )


def _chirp_source_message(source: ChirpSource) -> str | None:
    _chirp_source_string: dict[ChirpSource, str] = {
        ChirpSource.INTERNAL: "AUX_INS product",
        ChirpSource.REPLICA: "internal calibration",
        ChirpSource.NOMINAL: "nominal parameters (ideal chirp)",
    }

    return _chirp_source_string.get(source)


def _is_est_noise_product_invalid(est_noise_product: Path) -> bool:
    if not est_noise_product.exists():
        return True

    est_noise_channels = open_product_folder(est_noise_product).get_channels_list()
    return len(est_noise_channels) == 0


def _is_noise_fallback_required(
    internal_calibration_estimation_flag: bool,
    est_noise_product: Path,
) -> bool:
    if not internal_calibration_estimation_flag:
        return True

    return _is_est_noise_product_invalid(est_noise_product)


def write_core_auxiliary_input_products(
    aux_pp1: AuxProcessingParametersL1,
    job_order: L1JobOrder,
    l1_pre_processor_outputs: L1PreProcessorOutputProducts,
    acquisition_mode: AcquisitionMode,
):
    """Write core auxiliary input products"""

    antenna_patterns_required = isinstance(job_order.io_products, L1StripmapProducts)

    aux_ins_chirp_required = aux_pp1.range_compression.range_reference_function_source == ChirpSource.INTERNAL

    noise_fallback_required = _is_noise_fallback_required(
        aux_pp1.l0_product_import.internal_calibration_estimation_flag,
        l1_pre_processor_outputs.est_noise_product,
    )

    tx_power_tracking_fallback_required = not aux_pp1.l0_product_import.internal_calibration_estimation_flag

    drift_normalization_required = l1_pre_processor_outputs.amp_phase_drift_product.exists()

    requested_products = write_auxiliary_data.RequestedProducts(
        chirp_replica_product=(l1_pre_processor_outputs.chirp_replica_product if aux_ins_chirp_required else None),
        amp_phase_drift_product=(
            l1_pre_processor_outputs.amp_phase_drift_product if drift_normalization_required else None
        ),
        tx_power_tracking_product=(
            l1_pre_processor_outputs.tx_power_tracking_product if tx_power_tracking_fallback_required else None
        ),
        est_noise_product=(l1_pre_processor_outputs.est_noise_product if noise_fallback_required else None),
        antenna_patterns=(
            write_auxiliary_data.RequestedProducts.AntennaPatterns(
                ant_d1_h_product=l1_pre_processor_outputs.antenna_patterns.ant_d1_h_product,
                ant_d1_v_product=l1_pre_processor_outputs.antenna_patterns.ant_d1_v_product,
                ant_d2_h_product=l1_pre_processor_outputs.antenna_patterns.ant_d2_h_product,
                ant_d2_v_product=l1_pre_processor_outputs.antenna_patterns.ant_d2_v_product,
            )
            if antenna_patterns_required
            else None
        ),
    )

    write_auxiliary_data.write_core_auxiliary_input_products(
        aux_ins_product_path=job_order.auxiliary_files.instrument_parameters,
        requested_products=requested_products,
        extracted_raw_product=l1_pre_processor_outputs.extracted_raw_product,
        acquisition_mode=acquisition_mode,
    )

    if antenna_patterns_required:
        bps_logger.debug("Antenna patterns retrieved from AUX_INS")

    bps_logger.info(
        "Chirp retrieved from " + f"{_chirp_source_message(aux_pp1.range_compression.range_reference_function_source)}"
    )
