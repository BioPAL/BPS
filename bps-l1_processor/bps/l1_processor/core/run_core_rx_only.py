# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 core processor run for RX only data
------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from arepyextras.runner import Environment
from bps.common.decorators import log_elapsed_time
from bps.common.runner_helper import run_application
from bps.l1_core_processor.interface import (
    L1CoreProcessorInterface,
    write_l1_coreproc_input_file,
    write_l1_coreproc_options_file,
    write_l1_coreproc_parameters_file,
)
from bps.l1_core_processor.processing_options import BPSL1CoreProcessorStep
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsProduct
from bps.l1_pre_processor.aux_ins.swst_bias import retrieve_swst_bias
from bps.l1_processor.core.channel_imbalance import ChannelImbalanceProcessingParametersL1
from bps.l1_processor.core.input_file_utils import fill_bps_l1_core_processor_input_file
from bps.l1_processor.core.interface import L1CoreProcessorInterfaceFiles
from bps.l1_processor.core.processing_options_utils import (
    fill_bps_l1_core_processor_processing_options,
)
from bps.l1_processor.core.processing_parameters_utils import (
    fill_sarfoc_processing_parameters,
)
from bps.l1_processor.processor_interface.aux_pp1 import AuxProcessingParametersL1
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder
from bps.l1_processor.settings.intermediate_names import IntermediateProductID
from bps.l1_processor.settings.l1_binaries import BPS_L1COREPROC_EXE_NAME
from bps.l1_processor.settings.l1_intermediates import L1PreProcessorOutputProducts


@log_elapsed_time("L1CoreProcessorRXOnly")
def run_l1_core_processing_rx_only(
    job_order: L1JobOrder,
    l1_pre_processor_outputs: L1PreProcessorOutputProducts,
    intermediate_files: dict[IntermediateProductID, Path],
    output_dir: Path,
    core_processor_files: L1CoreProcessorInterfaceFiles,
    bps_conf_file: Path,
    bps_logger_file: Path,
    aux_pp1: AuxProcessingParametersL1,
    channel_imbalance: ChannelImbalanceProcessingParametersL1 | None,
    env: Environment,
):
    """Execute L1 core processing"""
    aux_ins_file = AuxInsProduct.from_product(job_order.auxiliary_files.instrument_parameters).instrument_file
    instrument_swst_bias = retrieve_swst_bias(aux_ins_file)
    assert aux_pp1.rfi_mitigation.activation_mode in ("Enabled", "Disabled")

    core_processor_interface = L1CoreProcessorInterface(
        input_file=fill_bps_l1_core_processor_input_file(
            job_order=job_order,
            input_raw_product=l1_pre_processor_outputs.extracted_raw_product.absolute(),
            processing_options=core_processor_files.options_file.absolute(),
            processing_parameters=core_processor_files.params_file.absolute(),
            bps_configuration_file=bps_conf_file.absolute(),
            bps_log_file=bps_logger_file.absolute(),
            output_dir=output_dir.absolute(),
        ),
        options=fill_bps_l1_core_processor_processing_options(
            dem_path=None,
            aux_pp1_conf=aux_pp1,
            steps={
                BPSL1CoreProcessorStep.RFI_MITIGATION: aux_pp1.rfi_mitigation.activation_mode == "Enabled",
                BPSL1CoreProcessorStep.RANGE_FOCUSER: False,
                BPSL1CoreProcessorStep.DOPPLER_CENTROID_ESTIMATOR: False,
                BPSL1CoreProcessorStep.DOPPLER_RATE_ESTIMATOR: False,
                BPSL1CoreProcessorStep.AZIMUTH_FOCUSER: False,
                BPSL1CoreProcessorStep.RANGE_COMPENSATOR: False,
                BPSL1CoreProcessorStep.POLARIMETRIC_COMPENSATOR: False,
                BPSL1CoreProcessorStep.MULTI_LOOKER: False,
                BPSL1CoreProcessorStep.NESZ_MAP_GENERATOR: False,
                BPSL1CoreProcessorStep.DENOISER: False,
                BPSL1CoreProcessorStep.SLANT2_GROUND: False,
            },
            intermediate_files=intermediate_files,
            remove_sarfoc_intermediates=True,
        ),
        params=fill_sarfoc_processing_parameters(
            pp1=aux_pp1,
            channel_imbalance=channel_imbalance,
            instrument_swst_bias=instrument_swst_bias,
        ),
    )

    write_l1_coreproc_input_file(core_processor_interface.input_file, core_processor_files.input_file)
    write_l1_coreproc_options_file(core_processor_interface.options, core_processor_files.options_file)
    write_l1_coreproc_parameters_file(core_processor_interface.params, core_processor_files.params_file)

    run_application(
        env,
        BPS_L1COREPROC_EXE_NAME,
        str(core_processor_files.input_file.absolute()),
        1,
    )
