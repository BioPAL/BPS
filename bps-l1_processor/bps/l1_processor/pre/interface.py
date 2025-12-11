# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Pre Processor interface functions
------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bps.common import (
    STRIPMAP_SWATHS,
    bps_logger,
    retrieve_aux_product_data_single_content,
)
from bps.common.common import Swath
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsProduct
from bps.l1_pre_processor.configuration import (
    L1PreProcessorConfiguration,
    L1PreProcessorConfigurationFile,
)
from bps.l1_pre_processor.input_file import L1PreProcessorInputFile
from bps.l1_processor.processor_interface import aux_pp1
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1StripmapProducts,
)


def product_has_index_files(product: Path) -> bool:
    """Check wether index files for smart read are available"""
    index_file_regex = r"_raw__0s.*[0-9_]_idx_rx(h|v).dat$"
    for file in product.iterdir():
        if re.search(index_file_regex, file.name) is not None:
            return True

    return False


def is_instr_attitude(attitude_content: str) -> bool:
    """Wether given attitude is marked as Instr_Attitude"""
    attitude_type_str = attitude_content.split("<Attitude_File_Type>")[1].split("</Attitude_File_Type>")[0]
    return attitude_type_str == "Instr_Attitude"


def fill_l1_preprocessor_input_file(
    *,
    job_order: L1JobOrder,
    iers_bulletin_file: Path,
    input_conf_file: Path,
    bps_configuration_file: Path,
    bps_log_file: Path,
    output_raw_data_product: Path,
    intermediate_dyn_cal_product: Path,
    intermediate_pgpproduct: Path,
    output_per_line_correction_factors_product: Path,
    output_chirp_replica_product: Path,
    output_channel_delays_file: Path,
    output_channel_imbalance_file: Path,
    output_tx_power_tracking_product: Path,
    output_est_noise_product: Path,
    output_ssp_headers_file: Path | None = None,
    output_report_file: Path | None = None,
    repaired_attitude_file: Path | None = None,
) -> L1PreProcessorInputFile:
    """Fill the input file of the L1 PreProcessor

    Parameters
    ----------
    job_order : L1JobOrder
        job order object
    iers_bulletin_file : Path
        path to the iers bulletin file
    input_conf_file : Path
        file path of the configuration file
    bps_configuration_file : Path
        bps common configuration file
    bps_log_file : Path
        bps l1 processor log file
    output_raw_data_product: Path
        destination where to save the imported RAW data in Aresys format
    intermediate_dyn_cal_product: Path
        destination where to save the imported dynamic calibration data in Aresys format
    intermediate_pgpproduct: Path
        destination where to save the imported PGP product in Aresys format
    output_per_line_correction_factors_product: Path
        destination where to save the imported amplitude and phase drifts in Aresys format
    output_chirp_replica_product: Path
        destination where to save the imported chirp replica in Aresys format
    output_channel_delays_file: Path
        destination where to save the imported channel delays in Aresys format
    output_channel_imbalance_file: Path
        destination where to save the imported channel imbalance in Aresys format
    output_tx_power_tracking_product: Path
        destination where to save the imported transmit power tracking in Aresys format
    output_est_noise_product: Path
        destination where to save the imported estimated noise in Aresys format
    output_report_file: Path
        destination where to save the annotation report xml file
    output_ssp_headers_file: Path, optional
        destination where to save the SSP headers file
    repaired_attitude_file: Path, optional
        overwrite attitude file with this path
    Returns
    -------
    L1PreProcessorInputFile
        L0ImportPreProcessor app input file object
    """
    monitoring = (
        job_order.io_products.input.input_monitoring if isinstance(job_order.io_products, L1StripmapProducts) else None
    )

    smart_read_enabled = product_has_index_files(job_order.io_products.input.input_standard)
    smart_read_requested = job_order.processor_configuration.azimuth_interval is not None

    if smart_read_requested and not smart_read_enabled:
        bps_logger.warning(
            "Smart-read disabled: missing index files in %s",
            job_order.io_products.input.input_standard,
        )

    time_of_interest = (
        job_order.processor_configuration.azimuth_interval if smart_read_requested and smart_read_enabled else None
    )

    orbit_file = retrieve_aux_product_data_single_content(job_order.auxiliary_files.orbit)
    attitude_file = repaired_attitude_file or retrieve_aux_product_data_single_content(
        job_order.auxiliary_files.attitude
    )

    aux_ins_file = AuxInsProduct.from_product(job_order.auxiliary_files.instrument_parameters).instrument_file

    iers_bulletin_optional_file = None if is_instr_attitude(attitude_file.read_text()) else iers_bulletin_file

    return L1PreProcessorInputFile(
        input_l0s_product=job_order.io_products.input.input_standard,
        input_aux_orb_file=orbit_file,
        input_aux_att_file=attitude_file,
        input_aux_ins_file=aux_ins_file,
        input_iersbullettin_file=iers_bulletin_optional_file,
        input_configuration_file=input_conf_file,
        time_of_interest=time_of_interest,
        bps_configuration_file=bps_configuration_file,
        bps_log_file=bps_log_file,
        output_raw_data_product=output_raw_data_product,
        input_l0m_product=monitoring,
        intermediate_dyn_cal_product=intermediate_dyn_cal_product,
        intermediate_pgpproduct=intermediate_pgpproduct,
        output_per_line_correction_factors_product=output_per_line_correction_factors_product,
        output_chirp_replica_product=output_chirp_replica_product,
        output_channel_delays_file=output_channel_delays_file,
        output_channel_imbalance_file=output_channel_imbalance_file,
        output_tx_power_tracking_product=output_tx_power_tracking_product,
        output_est_noise_product=output_est_noise_product,
        output_ssp_headers_file=output_ssp_headers_file,
        output_report_file=output_report_file,
    )


def fill_l1_preprocessor_configuration_file(
    *, aux_pp1_obj: aux_pp1.AuxProcessingParametersL1
) -> L1PreProcessorConfigurationFile:
    """Fill L1 Pre Processor configuration based on AUX PP1

    Parameters
    ----------
    aux_pp1 : aux_pp1.AuxPP1
        AUX PP1 object

    Returns
    -------
    L1PreProcessorConfigurationFile
        L1 pre processor configuration object
    """
    enable_channel_delays_annotation = (
        aux_pp1_obj.range_compression.range_reference_function_source
        in (
            aux_pp1.ChirpSource.NOMINAL,
            aux_pp1.ChirpSource.INTERNAL,
        )
        and aux_pp1_obj.internal_calibration_correction.delay_correction_flag
    )

    enable_internal_calibration = aux_pp1_obj.l0_product_import.internal_calibration_estimation_flag

    configurations = []
    for swath in STRIPMAP_SWATHS + [Swath.RO]:
        swath_conf = L1PreProcessorConfiguration(
            max_isp_gap=aux_pp1_obj.l0_product_import.max_isp_gap,
            raw_mean_expected=0.0,
            raw_mean_threshold=0.0,
            raw_std_expected=0.0,
            raw_std_threshold=0.0,
            correct_bias=aux_pp1_obj.raw_data_correction.bias_correction_flag,
            correct_gain_imbalance=aux_pp1_obj.raw_data_correction.gain_imbalance_correction_flag,
            correct_non_orthogonality=aux_pp1_obj.raw_data_correction.non_orthogonality_correction_flag,
            internal_calibration_source=L1PreProcessorConfiguration.Source.EXTRACTED,
            max_drift_amplitude_std_fraction=0.0,
            max_drift_phase_std_fraction=0.0,
            max_drift_amplitude_error=0.0,
            max_drift_phase_error=0.0,
            max_invalid_drift_fraction=0.0,
            enable_channel_delays_annotation=enable_channel_delays_annotation,
            enable_internal_calibration=enable_internal_calibration,
            swath=swath.name,
        )
        configurations.append(swath_conf)

    return L1PreProcessorConfigurationFile(configurations)


@dataclass
class L1PreProcessorInterfaceFiles:
    """L1 PreProcessor interface files"""

    input_file: Path
    configuration_file: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> L1PreProcessorInterfaceFiles:
        """Setup the paths of the L1PP interface files

        Parameters
        ----------
        base_dir : Path
            directory where to save the interface files

        Returns
        -------
        L1PreProcessorInterfaceFiles
            Struct with the paths of the files
        """
        base_name = "L1PreProcessor"
        return cls(
            input_file=base_dir.joinpath(f"{base_name}InputFile.xml"),
            configuration_file=base_dir.joinpath(f"{base_name}Conf.xml"),
        )

    def delete(self):
        """Delete files"""
        self.input_file.unlink()
        self.configuration_file.unlink()
