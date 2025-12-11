# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Folder layout
-------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bps.common import bps_logger
from bps.l1_processor.core.interface import L1CoreProcessorInterfaceFiles
from bps.l1_processor.parc.parc_info import ScatteringResponse
from bps.l1_processor.pre.interface import L1PreProcessorInterfaceFiles
from bps.l1_processor.settings.intermediate_names import (
    BPS_CONF_FILE_NAME,
    BPS_L1_PROCESSOR_STATUS_FILE_NAME,
    L1_CORE_PROC_OUTPUT_FOLDER,
    L1_PARC_PROC_OUTPUT_FOLDERS,
    L1_PRE_PROC_OUTPUT_FOLDER,
)
from bps.l1_processor.settings.l1_intermediates import (
    L1CoreProcessorOutputProducts,
    L1ParcCoreAdditionalInputFiles,
    L1PreProcessorOutputProducts,
)


@dataclass
class FolderLayout:
    """Layout of the files and folders generated during the run"""

    bps_logger_file: Path
    bps_conf_file: Path
    bps_l1_processor_status_file: Path

    pre_processor_outputs: L1PreProcessorOutputProducts
    core_processor_outputs: L1CoreProcessorOutputProducts
    pre_processor_files: L1PreProcessorInterfaceFiles
    core_processor_files: L1CoreProcessorInterfaceFiles

    parc_core_processor_outputs: dict[ScatteringResponse, L1CoreProcessorOutputProducts]
    parc_core_processor_files: dict[ScatteringResponse, L1CoreProcessorInterfaceFiles]
    parc_core_processor_additional_inputs: dict[ScatteringResponse, L1ParcCoreAdditionalInputFiles]

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> FolderLayout:
        """Build the layout in a given base directory"""

        bps_logger_file = bps_logger.get_log_file()
        assert bps_logger_file is not None

        bps_conf_file = base_dir.joinpath(BPS_CONF_FILE_NAME)

        bps_l1_processor_status_file = base_dir.joinpath(BPS_L1_PROCESSOR_STATUS_FILE_NAME)

        l1_pre_processor_outputs = L1PreProcessorOutputProducts.from_output_dir(
            base_dir.joinpath(L1_PRE_PROC_OUTPUT_FOLDER)
        )

        l1_core_processor_outputs = L1CoreProcessorOutputProducts.from_output_dir(
            base_dir.joinpath(L1_CORE_PROC_OUTPUT_FOLDER)
        )

        pre_processor_files = L1PreProcessorInterfaceFiles.from_base_dir(base_dir)
        core_processor_files = L1CoreProcessorInterfaceFiles.from_base_dir(base_dir)

        parc_core_processor_outputs = {}
        parc_core_processor_files = {}
        parc_core_additional_inputs = {}
        for scattering_response in ScatteringResponse:
            parc_folder = base_dir.joinpath(L1_PARC_PROC_OUTPUT_FOLDERS[scattering_response])
            parc_core_processor_outputs[scattering_response] = L1CoreProcessorOutputProducts.from_output_dir(
                parc_folder.joinpath(L1_CORE_PROC_OUTPUT_FOLDER)
            )
            parc_core_processor_files[scattering_response] = L1CoreProcessorInterfaceFiles.from_base_dir(parc_folder)
            parc_core_additional_inputs[scattering_response] = L1ParcCoreAdditionalInputFiles.from_base_dir(parc_folder)

        return cls(
            bps_logger_file,
            bps_conf_file,
            bps_l1_processor_status_file,
            l1_pre_processor_outputs,
            l1_core_processor_outputs,
            pre_processor_files,
            core_processor_files,
            parc_core_processor_outputs,
            parc_core_processor_files,
            parc_core_additional_inputs,
        )
