# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 processor run
--------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from arepyextras.runner import Environment
from bps.common import bps_logger, retrieve_aux_product_data_single_content
from bps.common.configuration import (
    fill_bps_configuration_file,
    write_bps_configuration_file,
)
from bps.l1_core_processor.processing_options import BPSL1CoreProcessorStep
from bps.l1_processor import __version__ as VERSION
from bps.l1_processor.aux_tec.utils import retrieve_single_decompressed_ionex_file
from bps.l1_processor.core.parsing import parse_aux_pp1
from bps.l1_processor.core.processing_options_utils import (
    retrieve_bps_l1_core_processor_steps,
)
from bps.l1_processor.core.run import run_l1_core_processing
from bps.l1_processor.core.run_core_rx_only import run_l1_core_processing_rx_only
from bps.l1_processor.core.time_validity import raise_on_inconsistent_timings
from bps.l1_processor.core.utils import (
    retrieve_rxh_data_file,
    retrieve_rxv_data_file,
    save_reference_extracted_raw_annotation,
    update_bps_l1_core_processor_status_file,
    update_bps_l1_pre_processor_report_file,
)
from bps.l1_processor.folder_layout import FolderLayout
from bps.l1_processor.intermediate_footprint import (
    add_footprint_file_to_intermediate_products,
)
from bps.l1_processor.io.l0_mph_utils import L0MainProductHeader
from bps.l1_processor.iri.ionospheric_app import run_iri_wrapper
from bps.l1_processor.parc.parc_info_utils import parse_parc_info
from bps.l1_processor.parc.parc_processing_utils import detect_parc_processing
from bps.l1_processor.parc.run import (
    ParcProcessingAdditionalInputs,
    run_parc_processing_on_all_responses,
)
from bps.l1_processor.post.run import run_l1_post_processing
from bps.l1_processor.pre.run import run_l1_pre_processing
from bps.l1_processor.processor_interface.aux_pp1 import (
    AuxProcessingParametersL1,
    ChirpSource,
    GeneralConf,
)
from bps.l1_processor.processor_interface.aux_pp1_product import AuxPP1Product
from bps.l1_processor.processor_interface.dem_db import select_dem
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1RXOnlyProducts,
    L1StripmapProducts,
)
from bps.l1_processor.processor_interface.rfi_activation_mask import (
    is_rfi_mitigation_enabled_given_footprint,
)
from bps.l1_processor.processor_interface.xsd_validation import validate_aux_pp1
from bps.l1_processor.scene_center import compute_scene_center
from bps.l1_processor.settings.intermediate_names import IntermediateProductID
from bps.l1_processor.status_system.status import (
    BPSL1ProcessorStateInfo,
    BPSL1ProcessorStep,
)
from bps.l1_processor.status_system.status_utils import (
    initialize_bps_l1_processor_status_from_file,
    write_bps_l1_processing_status,
)


def input_quad_pol_check(product: Path) -> bool:
    """Check input level 0 product is quad pol"""
    rxh_data_file = retrieve_rxh_data_file(product)
    if not rxh_data_file:
        bps_logger.warning("No rxh files found in: %s", product)
        return False

    rxv_data_file = retrieve_rxv_data_file(product)
    if not rxv_data_file:
        bps_logger.warning("No rxv files found in: %s", product)
        return False

    return True


def determine_rfi_activation(
    footprint: list[tuple[float, float]] | None, mask_file: Path | None, aux_pp1: AuxProcessingParametersL1
):
    match aux_pp1.rfi_mitigation.activation_mode:
        case "MaskBased":
            if footprint is None:
                raise RuntimeError("Missing footprint in L0 product: cannot determine RFI mitigation activation")
            if mask_file is None:
                raise RuntimeError(
                    "Missing RFI activation mask file in AUX_PP1 product: cannot determine RFI mitigation activation"
                )
            aux_pp1.rfi_mitigation.activation_mode = (
                "Enabled"
                if is_rfi_mitigation_enabled_given_footprint(
                    footprint, mask_file, aux_pp1.rfi_mitigation.activation_mask_threshold
                )
                else "Disabled"
            )
            return

        case "Enabled" | "Disabled":
            return

    raise RuntimeError(f"Unknown activation mode: {aux_pp1.rfi_mitigation.activation_mode}")


def run_l1_processing_impl(
    env: Environment,
    job_order: L1JobOrder,
    layout: FolderLayout,
    iers_bulletin_file: Path,
):
    """Performs processing as described in job order.

    Parameters
    ----------
    env: Environment
        Running environment
    job_order : L1JobOrder
        job_order object
    layout : FolderLayout
        layout of the working directory
    iers_bulletin_file : Path
        path to the iers bulletin file
    """

    if isinstance(job_order.io_products, L1RXOnlyProducts):
        bps_logger.info("RX only processing started")

    layout.pre_processor_outputs.directory.mkdir(parents=False, exist_ok=True)

    if not isinstance(job_order.io_products, L1RXOnlyProducts):
        layout.core_processor_outputs.directory.mkdir(parents=False, exist_ok=True)

    # BPS Common Configuration
    bps_common_configuration = fill_bps_configuration_file(
        job_order.processor_configuration,
        task_name="L1_P",
        processor_name="L1_P",
        processor_version=bps_logger.get_version_in_logger_format(VERSION),
        node_name=bps_logger.get_default_logger_node(),
    )

    write_bps_configuration_file(bps_common_configuration, layout.bps_conf_file)

    # Input Level0 products
    bps_logger.info(f"Input standard product: {job_order.io_products.input.input_standard}")
    main_product_header = L0MainProductHeader.from_product(job_order.io_products.input.input_standard)

    quad_pol_level0_product = input_quad_pol_check(job_order.io_products.input.input_standard)

    if isinstance(job_order.io_products, L1StripmapProducts) and not job_order.io_products.input.input_monitoring:
        bps_logger.warning("Input L0 Monitoring product not specified")
        bps_logger.warning(
            "Internal calibration parameters will be retrieved from L0 Standard product or,"
            + " if not possible, from AUX_INS product"
        )

    # Check existence of auxiliary products
    required_aux_products = {
        job_order.auxiliary_files.orbit,
        job_order.auxiliary_files.attitude,
        job_order.auxiliary_files.instrument_parameters,
        job_order.auxiliary_files.l1_processing_parameters,
    }

    missing_aux_products = list(filter(lambda f: not f.exists(), required_aux_products))
    if len(missing_aux_products) > 0:
        for product in missing_aux_products:
            bps_logger.error("Aux product '%s' not found", str(product))
        raise RuntimeError("Auxiliary products not found")

    bps_logger.info("AUX_ORB product: %s", job_order.auxiliary_files.orbit)

    bps_logger.info("AUX_ATT product: %s", job_order.auxiliary_files.attitude)

    # Input parsing: aux_pp1
    bps_logger.info("AUX_PP1 product: %s", job_order.auxiliary_files.l1_processing_parameters)
    aux_pp1_product = AuxPP1Product.from_product(job_order.auxiliary_files.l1_processing_parameters)
    aux_pp1_file = aux_pp1_product.aux_pp1_file
    validate_aux_pp1(aux_pp1_file)

    aux_pp1 = parse_aux_pp1(aux_pp1_file.read_text())
    aux_pp1.raise_if_inconsistent()

    if job_order.processing_parameters.rfi_mitigation_enabled is not None:
        job_order_setting = "Enabled" if job_order.processing_parameters.rfi_mitigation_enabled else "Disabled"
        if aux_pp1.rfi_mitigation.activation_mode != job_order_setting:
            bps_logger.info(
                f"RFI mitigation activation mode '{aux_pp1.rfi_mitigation.activation_mode}' overwritten by JobOrder setting to '{job_order_setting}'"
            )
            aux_pp1.rfi_mitigation.activation_mode = job_order_setting

    determine_rfi_activation(
        main_product_header.footprint, mask_file=aux_pp1_product.rfi_activation_mask, aux_pp1=aux_pp1
    )
    assert aux_pp1.rfi_mitigation.activation_mode in ("Enabled", "Disabled")

    if not quad_pol_level0_product:
        message = f"Input L0S product is not quad pol: {job_order.io_products.input.input_standard}"
        if aux_pp1.general.dual_polarisation_processing_flag:
            bps_logger.warning(message)
            aux_pp1.switch_off_steps_requiring_quad_pol_data()
        else:
            bps_logger.error(message)
            raise RuntimeError(
                "Cannot proceed with dual pol data: "
                + "dual pol processing can be enabled through the AUX_PP1 'DualPolarisationProcessingFlag'"
            )

    # Fallback on WGS84 in case of missing DEM DB
    dem_folder = select_dem(job_order, aux_pp1)

    # Force RGC product promotion, required to write geometry products after core processing
    if aux_pp1.general.height_model == GeneralConf.EarthModel.ELLIPSOID:
        layout.core_processor_outputs.add_output_if_not_present(
            IntermediateProductID.RGC_DC_FR_ESTIMATOR, to_be_removed=True
        )

    # Aux Ins product
    bps_logger.info("AUX_INS product: %s", job_order.auxiliary_files.instrument_parameters)

    if job_order.auxiliary_files.tec_maps:
        if aux_pp1.is_ionospheric_calibration_enabled():
            for tec_map in job_order.auxiliary_files.tec_maps:
                bps_logger.info("AUX_TEC product: %s", tec_map)
        else:
            for tec_map in job_order.auxiliary_files.tec_maps:
                bps_logger.info("AUX_TEC product (unused): %s", tec_map)

    if job_order.geomagnetic_field is not None:
        if aux_pp1.is_ionospheric_calibration_enabled():
            bps_logger.info("GMF: %s", job_order.geomagnetic_field)
        else:
            bps_logger.info("GMF (unused): %s", job_order.geomagnetic_field)

    if job_order.iri_data_folder is not None:
        if aux_pp1.is_ionospheric_calibration_enabled():
            bps_logger.info("IRI data: %s", job_order.iri_data_folder)
        else:
            bps_logger.info("IRI data (unused): %s", job_order.iri_data_folder)

    if aux_pp1.is_ionospheric_calibration_enabled():
        if job_order.iri_data_folder is None:
            raise RuntimeError(
                "IRI data folder not specified in JobOrder, but ionospheric calibration is enabled: cannot proceed"
            )

        if not job_order.iri_data_folder.exists():
            raise RuntimeError(f"IRI data folder does not exist: {job_order.iri_data_folder}")

    # Status file
    bps_l1_processor_status = initialize_bps_l1_processor_status_from_file(layout.bps_l1_processor_status_file)

    bps_logger.info("Status file: %s", layout.bps_l1_processor_status_file)

    def update_and_write_status(completed_step: BPSL1ProcessorStep):
        state_info = BPSL1ProcessorStateInfo(completed_step)
        bps_l1_processor_status.add_state(state_info)
        write_bps_l1_processing_status(bps_l1_processor_status, layout.bps_l1_processor_status_file)

    if layout.core_processor_outputs.status_file.exists():
        update_bps_l1_core_processor_status_file(
            layout.core_processor_outputs.status_file,
            layout.core_processor_files.params_file,
        )

    raise_on_inconsistent_timings(data_interval=main_product_header.phenomenon_time, job_order=job_order)

    current_step = BPSL1ProcessorStep.PRE_PROCESSOR
    if not bps_l1_processor_status.is_step_completed(step=current_step):
        # L1 Pre Processor
        run_l1_pre_processing(
            job_order=job_order,
            iers_bulletin_file=iers_bulletin_file,
            pre_processor_files=layout.pre_processor_files,
            bps_conf_file=layout.bps_conf_file,
            bps_logger_file=layout.bps_logger_file,
            l1_pre_processor_outputs=layout.pre_processor_outputs,
            aux_pp1=aux_pp1,
            acquisition_mode=main_product_header.acquisition_mode,
            data_interval=main_product_header.phenomenon_time,
            env=env,
        )

        if not job_order.keep_intermediate():
            bps_logger.debug("Removing input L1 pre processor files")
            layout.pre_processor_files.delete()

        update_and_write_status(current_step)
    else:
        bps_logger.info("L1PreProcessor already completed")

    if isinstance(job_order.io_products, L1RXOnlyProducts):
        output_dir_rx_only = layout.pre_processor_outputs.directory.parent
        intermediate_files = {
            IntermediateProductID.RAW_MITIGATED: output_dir_rx_only.joinpath(
                IntermediateProductID.RAW_MITIGATED.to_name()
            ),
            IntermediateProductID.RFI_TIME_MASK: output_dir_rx_only.joinpath(
                IntermediateProductID.RFI_TIME_MASK.to_name()
            ),
            IntermediateProductID.RFI_FREQ_MASK: output_dir_rx_only.joinpath(
                IntermediateProductID.RFI_FREQ_MASK.to_name()
            ),
        }
        run_l1_core_processing_rx_only(
            job_order=job_order,
            l1_pre_processor_outputs=layout.pre_processor_outputs,
            intermediate_files=intermediate_files,
            output_dir=layout.core_processor_outputs.directory,
            core_processor_files=layout.core_processor_files,
            bps_conf_file=layout.bps_conf_file,
            bps_logger_file=layout.bps_logger_file,
            aux_pp1=aux_pp1,
            channel_imbalance=None,
            env=env,
        )

        if not job_order.keep_intermediate():
            bps_logger.debug("Removing intermediate L1 pre processor products")
            new_raw_extracted = output_dir_rx_only.joinpath(layout.pre_processor_outputs.extracted_raw_product.name)
            layout.pre_processor_outputs.extracted_raw_product.rename(new_raw_extracted)
            layout.pre_processor_outputs.delete()

            layout.core_processor_outputs.delete()
            layout.core_processor_files.delete()

            layout.bps_conf_file.unlink()
            layout.bps_l1_processor_status_file.unlink()
        bps_logger.info("RX only processing correctly terminated")
        return

    per_line_correction_factors_product = (
        layout.pre_processor_outputs.amp_phase_drift_product.absolute()
        if aux_pp1.internal_calibration_correction.drift_correction_flag
        else None
    )

    chirp_replica_product = (
        layout.pre_processor_outputs.chirp_replica_product.absolute()
        if aux_pp1.range_compression.range_reference_function_source in (ChirpSource.REPLICA, ChirpSource.INTERNAL)
        else None
    )

    channel_imbalance_file = (
        layout.pre_processor_outputs.channel_imbalance_file.absolute()
        if aux_pp1.internal_calibration_correction.channel_imbalance_correction_flag
        and aux_pp1.range_compression.range_reference_function_source != ChirpSource.REPLICA
        else None
    )

    tec_map_file = None
    if job_order.auxiliary_files.tec_maps and aux_pp1.is_ionospheric_calibration_enabled():
        tec_map_file = retrieve_single_decompressed_ionex_file(
            job_order.auxiliary_files.tec_maps, layout.core_processor_files.ionex_files
        )

    # Defining core processor outputs depending on what requested in job order and aux_pp1
    core_processing_steps = retrieve_bps_l1_core_processor_steps(job_order=job_order, aux_pp1_conf=aux_pp1)

    layout.core_processor_outputs.update_main_slc_output(
        bool(core_processing_steps.get(BPSL1CoreProcessorStep.POLARIMETRIC_COMPENSATOR)),
        aux_pp1.autofocus.autofocus_flag,
    )
    if core_processing_steps.get(BPSL1CoreProcessorStep.SLANT2_GROUND):
        layout.core_processor_outputs.update_grd_output()

    layout.core_processor_outputs.update_lut_outputs(
        bool(core_processing_steps.get(BPSL1CoreProcessorStep.RFI_MITIGATION)),
        aux_pp1.is_ionospheric_calibration_enabled(),
        aux_pp1.autofocus.autofocus_flag,
        bool(core_processing_steps.get(BPSL1CoreProcessorStep.NESZ_MAP_GENERATOR)),
    )

    if job_order.auxiliary_files.calibration_site_information is not None:
        layout.core_processor_outputs.update_parc_required_outputs()

    current_step = BPSL1ProcessorStep.CORE_PROCESSOR
    if not bps_l1_processor_status.is_step_completed(step=current_step):
        save_reference_extracted_raw_annotation(
            layout.pre_processor_outputs.extracted_raw_product,
            layout.core_processor_outputs.extracted_raw_annotation,
        )

        if aux_pp1.range_compression.range_reference_function_source == ChirpSource.REPLICA:
            assert chirp_replica_product is not None
            update_bps_l1_pre_processor_report_file(layout.pre_processor_outputs.report_file, chirp_replica_product)

        shutil.copy2(
            layout.pre_processor_outputs.report_file,
            layout.core_processor_outputs.preprocessor_report,
        )

        time, point = compute_scene_center(
            layout.pre_processor_outputs.extracted_raw_product, job_order.processor_configuration.azimuth_interval
        )

        if aux_pp1.is_ionospheric_calibration_enabled():
            assert job_order.iri_data_folder is not None
            assert job_order.iri_data_folder.exists() is not None

            run_iri_wrapper(
                env,
                time,
                point,
                job_order.iri_data_folder,
                layout.core_processor_outputs.ionospheric_height_model_file,
            )

        # L1 Core Processor
        run_l1_core_processing(
            job_order=job_order,
            l1_pre_processor_outputs=layout.pre_processor_outputs,
            l1_core_processor_outputs=layout.core_processor_outputs,
            core_processing_steps=core_processing_steps,
            chirp_replica_product=chirp_replica_product,
            channel_imbalance_file=channel_imbalance_file,
            geomagnetic_field_folder=job_order.geomagnetic_field,
            tec_map_field_folder=tec_map_file,
            ionospheric_height_model_file=layout.core_processor_outputs.ionospheric_height_model_file,
            input_faraday_rotation_product=None,
            input_phase_screen_product=None,
            per_line_correction_factors_product=per_line_correction_factors_product,
            core_processor_files=layout.core_processor_files,
            bps_conf_file=layout.bps_conf_file,
            bps_logger_file=layout.bps_logger_file,
            aux_pp1=aux_pp1,
            dem_path=dem_folder,
            env=env,
        )

        if not job_order.keep_intermediate():
            if job_order.auxiliary_files.calibration_site_information is None:
                bps_logger.info("Removing intermediate L1 pre processor products")
                layout.pre_processor_outputs.delete()

                bps_logger.debug("Removing input L1 core processor files")
                layout.core_processor_files.delete()

        update_and_write_status(current_step)
    else:
        bps_logger.info("L1CoreProcessor already completed")

    current_step = BPSL1ProcessorStep.POST_PROCESSOR
    if not bps_l1_processor_status.is_step_completed(step=current_step):
        add_footprint_file_to_intermediate_products(
            layout.core_processor_outputs,
            [layout.pre_processor_outputs.extracted_raw_product],
        )

        run_l1_post_processing(layout.core_processor_outputs, job_order, aux_pp1)

        if not job_order.keep_intermediate():
            if job_order.auxiliary_files.calibration_site_information is None:
                bps_logger.info("Removing intermediate L1 core processor products")
                layout.core_processor_outputs.delete()

        update_and_write_status(current_step)
    else:
        bps_logger.info("L1PostProcessor already completed")

    # PARC processing
    if job_order.auxiliary_files.calibration_site_information is not None:
        bps_logger.info(
            "PARC_INFO product: %s",
            job_order.auxiliary_files.calibration_site_information,
        )
        parc_info_list = parse_parc_info(
            retrieve_aux_product_data_single_content(job_order.auxiliary_files.calibration_site_information).read_text()
        )
        assert len(parc_info_list) > 0

        azimuth_sampling_frequency = (
            aux_pp1.azimuth_compression.azimuth_resampling_frequency
            if aux_pp1.azimuth_compression.azimuth_resampling
            else None
        )

        parc_processing_info = detect_parc_processing(
            parc_info_list=parc_info_list,
            raw_product=layout.pre_processor_outputs.extracted_raw_product,
            azimuth_processing_interval=job_order.processor_configuration.azimuth_interval,
            range_processing_interval=job_order.processing_parameters.range_interval,
            chirp_replica_product=chirp_replica_product,
            block_overlap_lines=aux_pp1.azimuth_compression.block_overlap_lines,
            block_overlap_samples=aux_pp1.azimuth_compression.block_overlap_samples,
            parc_roi_lines=aux_pp1.general.parc_roi_lines,
            parc_roi_samples=aux_pp1.general.parc_roi_samples,
            azimuth_sampling_frequency=azimuth_sampling_frequency,
        )

        current_step = BPSL1ProcessorStep.PARC_PROCESSOR
        if parc_processing_info is None:
            bps_logger.info("No PARC has been detected on the scene")
        elif bps_l1_processor_status.is_step_completed(step=current_step):
            bps_logger.info(f"Detected {parc_processing_info.parc_id} on the scene")
            bps_logger.info("L1ParcProcessor already completed")
        else:
            bps_logger.info(f"Detected {parc_processing_info.parc_id} on the scene")

            additional_inputs = ParcProcessingAdditionalInputs(
                chirp_replica_product,
                per_line_correction_factors_product,
                channel_imbalance_file,
                job_order.geomagnetic_field,
                tec_map_file=tec_map_file,
                dem_path=dem_folder,
            )

            run_parc_processing_on_all_responses(
                env,
                parc_processing_info.processing_data,
                job_order,
                aux_pp1,
                layout,
                additional_inputs,
            )

            update_and_write_status(current_step)

        # Final cleanup after parc processing
        if not job_order.keep_intermediate():
            bps_logger.info("Removing intermediate L1 pre processor products")
            layout.pre_processor_outputs.delete()
            bps_logger.info("Removing intermediate L1 core processor products")
            layout.core_processor_outputs.delete()
            bps_logger.debug("Removing input L1 core processor files")
            layout.core_processor_files.delete()

    if not job_order.keep_intermediate():
        layout.bps_conf_file.unlink()
