# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 parc processor run
-------------------------
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from arepyextras.runner import Environment
from bps.common import bps_logger
from bps.common.decorators import log_elapsed_time
from bps.l1_core_processor.processing_options import BPSL1CoreProcessorStep
from bps.l1_processor.core.interface import L1CoreProcessorInterfaceFiles
from bps.l1_processor.core.processing_options_utils import (
    retrieve_bps_l1_core_processor_steps,
)
from bps.l1_processor.core.run import run_l1_core_processing
from bps.l1_processor.core.utils import save_reference_extracted_raw_annotation
from bps.l1_processor.folder_layout import FolderLayout
from bps.l1_processor.intermediate_footprint import (
    add_footprint_file_to_intermediate_products,
)
from bps.l1_processor.parc.parc_processing_info import Delays, Window
from bps.l1_processor.parc.parc_processing_utils import (
    ScatteringResponse,
    apply_delay_to_product,
    update_aux_pp1_for_parc_processing,
    update_job_order_for_parc_processing,
)
from bps.l1_processor.post.interface import (
    export_in_bps_format,
    fill_bps_transcoder_configuration,
)
from bps.l1_processor.post.rfi_masks_statistics import compute_rfi_masks_statistics
from bps.l1_processor.processor_interface.aux_pp1 import (
    AuxProcessingParametersL1,
    DopplerEstimationConf,
    GeneralConf,
)
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1StripmapOutputProducts,
)
from bps.l1_processor.settings.intermediate_names import IntermediateProductID
from bps.l1_processor.settings.l1_intermediates import (
    L1CoreProcessorOutputProducts,
    L1ParcCoreAdditionalInputFiles,
    L1PreProcessorOutputProducts,
)
from bps.transcoder.io.preprocessor_report import parse_annotations


@dataclass
class ParcProcessingAdditionalInputs:
    """Additional inputs for the core processor"""

    chirp_replica_product: Path | None
    per_line_correction_factors_product: Path | None
    channel_imbalance_file: Path | None
    geomagnetic_field: Path | None
    tec_map_file: Path | None
    dem_path: Path | None


@log_elapsed_time("L1ParcPostProcessor")
def run_l1_post_processing_parc(
    parc_core_processor_outputs: L1CoreProcessorOutputProducts,
    job_order: L1JobOrder,
    aux_pp1: AuxProcessingParametersL1,
    scattering_response: ScatteringResponse,
):
    """Execute L1 post processing tailored for PARC processing output in the bps format"""

    assert isinstance(job_order.io_products.output, L1StripmapOutputProducts)

    scattering_response_dict = {
        ScatteringResponse.GT1: "1",
        ScatteringResponse.GT2: "2",
        ScatteringResponse.X: "X",
        ScatteringResponse.Y: "Y",
    }
    calibration_tag = scattering_response_dict[scattering_response]

    bps_transcoder_configuration = fill_bps_transcoder_configuration(
        job_order, aux_pp1, parc_core_processor_outputs.extracted_raw_annotation
    )

    if parc_core_processor_outputs.main_slc_id is None:
        raise RuntimeError(f"Main SLC product ID not specified for parc processing {scattering_response}")

    slc_product = parc_core_processor_outputs.output_products.get(parc_core_processor_outputs.main_slc_id)
    assert slc_product

    intermediate_files = {id.name: info.path for id, info in parc_core_processor_outputs.output_products.items()}

    auxiliary_files = job_order.auxiliary_files.get_all_aux_products_paths()

    rfi_masks_statistics = compute_rfi_masks_statistics(
        rfi_frequency_mask=intermediate_files.get(IntermediateProductID.RFI_FREQ_MASK.name),
        rfi_time_mask=intermediate_files.get(IntermediateProductID.RFI_TIME_MASK.name),
    )

    export_in_bps_format(
        input_product_path=slc_product.path,
        source_product_path=job_order.io_products.input.input_standard,
        source_monitoring_product_path=job_order.io_products.input.input_monitoring,
        auxiliary_files=auxiliary_files,
        intermediate_products_dict=intermediate_files,
        output_folder=job_order.io_products.output.output_directory,
        configuration=bps_transcoder_configuration,
        add_monitoring_product=False,
        calibration_tag=calibration_tag,
        gdal_num_threads=job_order.device_resources.num_threads,
        l1_pre_proc_report=parse_annotations(parc_core_processor_outputs.preprocessor_report.read_text()),
        rfi_masks_statistics=rfi_masks_statistics,
    )


@log_elapsed_time("L1ParcProcessor")
def run_l1_parc_processing(
    parc_job_order: L1JobOrder,
    scattering_response: ScatteringResponse,
    l1_pre_processor_outputs: L1PreProcessorOutputProducts,
    additional_inputs: ParcProcessingAdditionalInputs,
    parc_core_processor_files: L1CoreProcessorInterfaceFiles,
    parc_core_processor_outputs: L1CoreProcessorOutputProducts,
    parc_core_additional_inputs: L1ParcCoreAdditionalInputFiles,
    bps_conf_file: Path,
    bps_logger_file: Path,
    parc_aux_pp1: AuxProcessingParametersL1,
    env: Environment,
):
    """Execute L1 PARC processing: Core Processing and Post Processing"""

    bps_logger.info("L1ParcProcessor started for scattering response %s", scattering_response.name)

    parc_core_processor_outputs.directory.mkdir(parents=True, exist_ok=True)

    # Force RGC product promotion, required to write geometry products after core processing
    if parc_aux_pp1.general.height_model == GeneralConf.EarthModel.ELLIPSOID:
        parc_core_processor_outputs.add_output_if_not_present(
            IntermediateProductID.RGC_DC_FR_ESTIMATOR, to_be_removed=True
        )

    # Defining core processor outputs depending on what requested in job order and aux_pp1
    parc_core_processing_steps = retrieve_bps_l1_core_processor_steps(
        job_order=parc_job_order, aux_pp1_conf=parc_aux_pp1
    )

    parc_core_processing_steps[BPSL1CoreProcessorStep.NESZ_MAP_GENERATOR] = False
    parc_core_processing_steps[BPSL1CoreProcessorStep.DENOISER] = False

    parc_core_processor_outputs.update_main_slc_output(
        bool(parc_core_processing_steps.get(BPSL1CoreProcessorStep.POLARIMETRIC_COMPENSATOR)),
        parc_aux_pp1.autofocus.autofocus_flag,
    )
    if parc_core_processing_steps.get(BPSL1CoreProcessorStep.SLANT2_GROUND):
        parc_core_processor_outputs.update_grd_output()

    parc_core_processor_outputs.update_lut_outputs(
        bool(parc_core_processing_steps.get(BPSL1CoreProcessorStep.RFI_MITIGATION)),
        parc_aux_pp1.is_ionospheric_calibration_enabled(),
        parc_aux_pp1.autofocus.autofocus_flag,
        bool(parc_core_processing_steps.get(BPSL1CoreProcessorStep.NESZ_MAP_GENERATOR)),
    )

    save_reference_extracted_raw_annotation(
        l1_pre_processor_outputs.extracted_raw_product,
        parc_core_processor_outputs.extracted_raw_annotation,
    )

    shutil.copy2(
        l1_pre_processor_outputs.report_file,
        parc_core_processor_outputs.preprocessor_report,
    )

    # Update processing configuration
    if parc_aux_pp1.doppler_estimation.method == DopplerEstimationConf.Method.COMBINED:
        bps_logger.warning("Doppler estimation method forced to GEOMETRY to process small ROI around PARC")
        parc_aux_pp1.doppler_estimation.method = DopplerEstimationConf.Method.GEOMETRY

    # L1 Core Processor
    run_l1_core_processing(
        job_order=parc_job_order,
        l1_pre_processor_outputs=l1_pre_processor_outputs,
        l1_core_processor_outputs=parc_core_processor_outputs,
        core_processing_steps=parc_core_processing_steps,
        chirp_replica_product=additional_inputs.chirp_replica_product,
        channel_imbalance_file=additional_inputs.channel_imbalance_file,
        geomagnetic_field_folder=additional_inputs.geomagnetic_field,
        tec_map_field_folder=additional_inputs.tec_map_file,
        ionospheric_height_model_file=parc_core_processor_outputs.ionospheric_height_model_file,
        input_faraday_rotation_product=parc_core_additional_inputs.faraday_rotation_product,
        input_phase_screen_product=parc_core_additional_inputs.phase_screen_product,
        per_line_correction_factors_product=additional_inputs.per_line_correction_factors_product,
        core_processor_files=parc_core_processor_files,
        bps_conf_file=bps_conf_file,
        bps_logger_file=bps_logger_file,
        aux_pp1=parc_aux_pp1,
        dem_path=additional_inputs.dem_path,
        env=env,
    )

    # L1 Post Processor
    run_l1_post_processing_parc(
        parc_core_processor_outputs=parc_core_processor_outputs,
        job_order=parc_job_order,
        aux_pp1=parc_aux_pp1,
        scattering_response=scattering_response,
    )


@contextmanager
def parc_processing_contex(
    response_data: tuple[ScatteringResponse, Delays, Window],
    job_order: L1JobOrder,
    aux_pp1: AuxProcessingParametersL1,
    layout: FolderLayout,
):
    """prepare the layout and the input parameters for parc processing of given scattering response"""
    scattering_response, parc_delays, parc_processing_roi = response_data
    parc_core_outputs = layout.parc_core_processor_outputs[scattering_response]
    parc_core_files = layout.parc_core_processor_files[scattering_response]
    parc_core_additional_inputs = layout.parc_core_processor_additional_inputs[scattering_response]

    parc_job_order = update_job_order_for_parc_processing(deepcopy(job_order), parc_processing_roi)
    parc_aux_pp1 = update_aux_pp1_for_parc_processing(deepcopy(aux_pp1), parc_delays)

    # Ionospheric height model file is taken from the main run
    parc_core_outputs.directory.mkdir(parents=True, exist_ok=True)
    if layout.core_processor_outputs.ionospheric_height_model_file.exists():
        shutil.copy(
            layout.core_processor_outputs.ionospheric_height_model_file,
            parc_core_outputs.ionospheric_height_model_file,
        )

    # Apply delay also to intermediate outputs from main run
    faraday_rotation_product = layout.core_processor_outputs.output_products.get(IntermediateProductID.FR, None)
    if faraday_rotation_product and faraday_rotation_product.path.exists():
        apply_delay_to_product(
            faraday_rotation_product.path, parc_delays, parc_core_additional_inputs.faraday_rotation_product
        )

    phase_screen_product = layout.core_processor_outputs.output_products.get(
        IntermediateProductID.PHASE_SCREEN_BB, None
    )
    if phase_screen_product and phase_screen_product.path.exists():
        apply_delay_to_product(phase_screen_product.path, parc_delays, parc_core_additional_inputs.phase_screen_product)

    yield parc_job_order, parc_aux_pp1, parc_core_files, parc_core_outputs, parc_core_additional_inputs

    if not parc_job_order.keep_intermediate():
        bps_logger.info(
            "Removing intermediate L1 core processor products"
            + f"for PARC processing scattering response {scattering_response}"
        )
        parc_core_outputs.delete()

        bps_logger.debug("Removing input L1 PARC core processor files")
        parc_core_files.delete()

        bps_logger.debug("Removing input L1 PARC core additional inputs")
        parc_core_additional_inputs.delete()

        # Remove the entire parc directory if empty
        if len(list(parc_core_files.directory.iterdir())) == 0:
            parc_core_files.directory.rmdir()


def run_parc_processing_on_all_responses(
    env: Environment,
    processing_data_per_response: dict[ScatteringResponse, tuple[Delays, Window]],
    job_order: L1JobOrder,
    aux_pp1: AuxProcessingParametersL1,
    layout: FolderLayout,
    additional_inputs: ParcProcessingAdditionalInputs,
):
    """Run parc processing on all responses"""
    for (
        scattering_response,
        processing_data,
    ) in processing_data_per_response.items():
        response_data = (
            scattering_response,
            *processing_data,
        )
        with parc_processing_contex(
            response_data,
            job_order,
            aux_pp1,
            layout,
        ) as (parc_job_order, parc_aux_pp1, parc_core_files, parc_core_outputs, parc_core_additional_inputs):
            # L1 PARC processing
            run_l1_parc_processing(
                parc_job_order=parc_job_order,
                scattering_response=scattering_response,
                l1_pre_processor_outputs=layout.pre_processor_outputs,
                additional_inputs=additional_inputs,
                parc_core_processor_files=parc_core_files,
                parc_core_processor_outputs=parc_core_outputs,
                parc_core_additional_inputs=parc_core_additional_inputs,
                bps_conf_file=layout.bps_conf_file,
                bps_logger_file=layout.bps_logger_file,
                parc_aux_pp1=parc_aux_pp1,
                env=env,
            )

            add_footprint_file_to_intermediate_products(parc_core_outputs)
