# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Export L1c Products
-------------------
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.io.metadata import EPolarization, RasterInfo
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common, translate_common
from bps.common.io.common_types.models import (
    HeightModelType,
    PrimaryImageSelectionMethodType,
)
from bps.common.roi_utils import RegionOfInterest
from bps.stack_cal_processor.configuration import (
    AZF_NAME,
    IOB_NAME,
    MSC_NAME,
    SKP_NAME,
    StackDataSpecs,
)
from bps.stack_cal_processor.core.filtering import (
    ConvolutionBorderType,
    build_sparse_uniform_filter_matrix,
)
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.msc.utils import compute_ionosphere_axes, los_range_ground_to_altitude
from bps.stack_cal_processor.core.signal_processing import compute_coherence
from bps.stack_cal_processor.core.utils import (
    compute_spatial_azimuth_shifts,
    compute_spatial_range_shifts,
    get_time_axis,
    interpolate_on_grid_nn,
)
from bps.stack_processor import BPS_STACK_PROCESSOR_NAME
from bps.stack_processor import __version__ as VERSION
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
    L1cProductExportConf,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.utils import (
    skp_phase_correction_flag,
    skp_phase_correction_flattening_only_flag,
)
from bps.stack_processor.interface.internal.intermediates import (
    CoregistrationOutputProducts,
)
from bps.transcoder.sarproduct.biomass_l1product import BIOMASSL1Product
from bps.transcoder.sarproduct.biomass_stackproduct import (
    BIOMASSStackCoregistrationParameters,
    BIOMASSStackInSARParameters,
    BIOMASSStackProcessingParameters,
    BIOMASSStackProduct,
    BIOMASSStackProductConfiguration,
    BIOMASSStackQuality,
    BIOMASSStackQualityParameters,
)
from bps.transcoder.sarproduct.biomass_stackproduct_writer import (
    BIOMASSStackProductWriter,
    IonosphereAxes,
)
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.sarproduct.sta.quality_index import StackQualityIndex
from bps.transcoder.sarproduct.sta.stack_unique_identifier import StackUniqueID
from bps.transcoder.utils.polarization_conversions import translate_polarization
from bps.transcoder.utils.quicklook_utils import QuickLookConf
from bps.transcoder.utils.time_conversions import round_precise_datetime


def export_l1c_product(
    *,
    job_order: StackJobOrder,
    aux_pps: AuxiliaryStaprocessingParameters,
    stack_pre_proc_exec_products: dict,
    stack_coreg_exec_products: dict,
    lut_shift_exec_products: dict,
    stack_cal_proc_exec_products: dict,
    lut_nodata_mask: npt.NDArray[float],
    full_primary_axes: tuple[npt.NDArray[float], npt.NDArray[float]],
    calib_reference_image_index: int,
    primary_coreg_azimuth_shifts: npt.NDArray[float],
    primary_coreg_range_shifts: npt.NDArray[float],
    overall_product_quality_index: np.uint32,
    image_index: int,
    stack_id: StackUniqueID,
    fnf_mask_file: Path | None,
    num_threads: int,
):
    """
    Export the L1c product.

    Parameters
    ----------
    job_order: StackJobOrder
        The STA_P job order object.

    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS configuration object.

    stack_pre_proc_exec_products: dict
        The output of the StackPreProcessorExecutionManager.

    stack_coreg_exec_products: dict
        The by-products the coreg processor.

    lut_shift_exec_products: dict
        The by-products of the LUT shifting step.

    stack_cal_proc_exec_products: dict
        The output of the StackCalProcessorExecutionManager.

    lut_nodata_mask: npt.NDArray[float]
        The mask of the stack common data interpolated on the
        LUT primary grid.

    full_primary_axes: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]] [s], [s]
        The full axes of the stack.

    calib_reference_image_index: int
        The index of the calibration reference image.

    primary_coreg_azimuth_shifts: npt.NDArray[float]
        The coregistration shifts of the primary image.

    primary_coreg_range_shifts: npt.NDArray[float],
        The coregistration shifts of the secondary image.

    overall_product_quality_index: np.uint32
        The overall product quality index.

    image_index: int
        The index of the current image.

    stack_id: StackUniqueID
        The stack unique identifier.

    fnf_mask_file: Optional[Path],
        Optionally, the path to the employed FNF mask (.tiff).

    num_threads: int
        Number of threads assigned for the job.

    """
    # Just shortcuts.
    coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]
    primary_product = stack_pre_proc_exec_products["l1a_product_data"][coreg_primary_image_index]
    secondary_product = stack_pre_proc_exec_products["l1a_product_data"][image_index]

    # Compute the coherence for LUTs and stats, and possibly visualization.
    flattening_phases = stack_cal_proc_exec_products["flattening_phases"]
    if aux_pps.skp_phase_calibration.skp_phase_estimation_flag and aux_pps.skp_phase_calibration.phase_correction_flag:
        flattening_phases = np.zeros_like(flattening_phases)

    coherences = _compute_coherences(
        calibrated_stack=stack_cal_proc_exec_products["calibrated_stack_images"],
        flattening_phases=flattening_phases,
        coreg_primary_image_index=coreg_primary_image_index,
        image_index=image_index,
        num_threads=num_threads,
    )

    coh_index = 0
    if EPolarization.hh in stack_pre_proc_exec_products["stack_polarizations"]:
        coh_index = stack_pre_proc_exec_products["stack_polarizations"].index(EPolarization.hh)
    ref_coherence = coherences[coh_index]

    # The LUT for the current data.
    lut_data = lut_shift_exec_products["lut_data"][image_index]
    lut_primary_axes_indices = (
        lut_shift_exec_products["lut_primary_azm_indices"],
        lut_shift_exec_products["lut_primary_rng_indices"],
    )
    lut_axes_primary = (
        full_primary_axes[0][lut_primary_axes_indices[0]],
        full_primary_axes[1][lut_primary_axes_indices[1]],
    )

    # Populate the stack LUTs.
    stack_lut_data, start_time_shift = _fill_stack_luts(
        aux_pps=aux_pps,
        calibration_products=stack_cal_proc_exec_products["calibration_products"],
        flattening_phases=stack_cal_proc_exec_products["flattening_phases"],
        vertical_wavenumbers=stack_cal_proc_exec_products["vertical_wavenumbers"],
        stack_data_specs=stack_cal_proc_exec_products["stack_data_specs"],
        lut_nodata_mask=lut_nodata_mask,
        lut_primary_axes_indices=lut_primary_axes_indices,
        full_primary_axes=full_primary_axes,
        coreg_azimuth_shifts=lut_data["azimuthCoregistrationShifts"],
        coreg_azimuth_geo_shifts=lut_data["azimuthOrbitCoregistrationShifts"],
        coreg_range_shifts=lut_data["rangeCoregistrationShifts"],
        coreg_range_geo_shifts=lut_data["rangeOrbitCoregistrationShifts"],
        primary_coreg_azimuth_shifts=primary_coreg_azimuth_shifts,
        primary_coreg_range_shifts=primary_coreg_range_shifts,
        image_index=image_index,
        coherence=ref_coherence,
    )
    lut_data.update(stack_lut_data)

    # Update the rasters to fit the primary grid (and potentially the ROI).
    _update_sar_product_grid(
        stack_cal_proc_exec_products["calibrated_stack_images"][image_index],
        stack_sar_data=secondary_product,
        primary_sar_data=primary_product,
        azimuth_time_axis_shift=start_time_shift,
        roi=stack_pre_proc_exec_products["stack_roi"],
    )

    # The configuration.
    transcoder_configuration = _fill_bps_transcoder_configuration(
        aux_pps,
        job_order,
        stack_pre_proc_exec_products["l1a_product_data"][image_index],
    )
    # The stack processing parameters.
    stack_processing_parameters = _fill_stack_processing_parameters(
        aux_pps=aux_pps,
        primary_image_info=stack_pre_proc_exec_products["coreg_primary_selection_info"],
        processed_polarizations=stack_pre_proc_exec_products["stack_polarizations"],
        cross_pol_method=aux_pps.general.polarization_combination_method,
        l1a_product=stack_pre_proc_exec_products["l1a_product_data"][image_index],
        actualized_coregistration_parameters=stack_coreg_exec_products["actualized_coregistration_parameters"][
            image_index
        ],
    )

    # The stack coregistration parameters.
    stack_coregistration_parameters = _fill_stack_coregistration_parameters(
        aux_pps=aux_pps,
        primary_image_info=stack_pre_proc_exec_products["coreg_primary_selection_info"],
        primary_image=job_order.input_stack[coreg_primary_image_index].name,
        secondary_image=job_order.input_stack[image_index].name,
        azimuth_coreg_shifts=lut_data["azimuthCoregistrationShifts"],
        range_coreg_shifts=lut_data["rangeCoregistrationShifts"],
        normal_baseline=stack_pre_proc_exec_products["stack_spatial_baselines"][image_index],
        l1a_product=stack_pre_proc_exec_products["l1a_product_data"][image_index],
    )

    # The stack InSAR parameters.
    stack_insar_parameters = _fill_stack_insar_parameters(
        job_order=job_order,
        spatial_ordering=stack_pre_proc_exec_products["stack_spatial_ordering"],
        calibration_products=stack_cal_proc_exec_products["calibration_products"],
        stack_data_specs=stack_cal_proc_exec_products["stack_data_specs"],
        calib_reference_image_index=calib_reference_image_index,
        image_index=image_index,
        incidence_angle=np.nanmean(np.deg2rad(lut_data["incidenceAngle"])),
    )

    # The stack quality info.
    stack_quality = _fill_stack_quality(
        aux_pps=aux_pps,
        calibration_products=stack_cal_proc_exec_products["calibration_products"],
        stack_polarizations=stack_pre_proc_exec_products["stack_polarizations"],
        rfi_indices=stack_pre_proc_exec_products["rfi_indices"][image_index],
        faraday_decorrelation=stack_pre_proc_exec_products["faraday_rotations"][image_index],
        invalid_residual_shifts_ratio=lut_shift_exec_products["invalid_residual_shifts_ratios"][image_index],
        overall_product_quality_index=overall_product_quality_index,
        coherences=[_abs_or_zero(coh) for coh in coherences],
    )

    ionosphere_axes = _fill_ionosphere_axes(
        calibration_products=stack_cal_proc_exec_products["calibration_products"],
        stack_specs=stack_cal_proc_exec_products["stack_data_specs"],
        image_index=image_index,
        incidence_angle=np.nanmean(np.deg2rad(lut_data["incidenceAngle"])),
    )

    # The fnf mask product name.
    file_name_fnf = None
    if fnf_mask_file:
        # The product format is FNF_product/data/FNF_file.tiff
        file_name_fnf = fnf_mask_file.parent.parent.name

    _export_to_l1c_format(
        input_product=secondary_product,
        product_primary=primary_product,
        output_folder=job_order.output_path.output_directory,
        configuration=transcoder_configuration,
        path_primary_l1a=job_order.input_stack[coreg_primary_image_index],
        path_secondary_l1a=job_order.input_stack[image_index],
        l1c_export_conf=aux_pps.l1c_product_export,
        coherence=ref_coherence,
        lut_dict=lut_data,
        lut_axes_primary=lut_axes_primary,
        ionosphere_axes=ionosphere_axes,
        stack_processing_parameters=stack_processing_parameters,
        stack_coregistration_parameters=stack_coregistration_parameters,
        stack_in_sarparameters=stack_insar_parameters,
        stack_quality=stack_quality,
        stack_nodata_mask=stack_cal_proc_exec_products["stack_nodata_mask"],
        stack_footprint=stack_pre_proc_exec_products["stack_footprint"],
        input_stack=tuple(stack_path.name for stack_path in job_order.input_stack),
        stack_id=stack_id,
        file_name_aux_pps=job_order.auxiliary_files.name,
        file_name_fnf=file_name_fnf,
        add_monitoring_product=True,
        num_threads=num_threads,
    )


def export_l1c_products(
    *,
    job_order: StackJobOrder,
    aux_pps: AuxiliaryStaprocessingParameters,
    stack_pre_proc_exec_products: dict,
    stack_coreg_proc_output_products: tuple[CoregistrationOutputProducts, ...],
    stack_coreg_exec_products: dict,
    lut_shift_exec_products: dict,
    stack_cal_proc_exec_products: dict,
    fnf_mask_file: Path | None,
    num_threads: int = 1,
):
    """
    Export the L1c products.

    Parameters
    ----------
    job_order: StackJobOrder
        The STA_P job-order.

    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS configuration object.

    stack_pre_proc_exec_products: dict
        The STA_P pre-processor (in-memory) execution products.

    stack_coreg_proc_output_products: tuple[CoregistrationOutputProducts, ...]
        The STA_P coregistration processor products.

    stack_coreg_exec_products: dict
        The STA_P coregistration execution by-products.

    lut_shift_exec_products: dict
        The by-products of the LUT shifting step.

    stack_cal_proc_exec_products: dict
        The STA_P calibration processor (in-memory) execution products.

    fnf_mask_file Optional[Path]
        Optionally, a path to the used FNF mask (.tiff).

    num_threads: int
        Number of threads assigned for the job.

    """
    # Just a couple of shortcuts.
    coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]
    calib_reference_image_index = stack_cal_proc_exec_products["calib_reference_image_index"]

    # The stack ROI.
    stack_roi = stack_pre_proc_exec_products["stack_roi"]

    # The primary data, LUTs, etc.
    sar_primary_data = stack_pre_proc_exec_products["l1a_product_data"][coreg_primary_image_index]

    # The full primary axes.
    full_primary_axes = (
        get_time_axis(sar_primary_data.raster_info_list[0], axis=0, roi=stack_roi, absolute=True)[0],
        get_time_axis(sar_primary_data.raster_info_list[0], axis=1, roi=stack_roi, absolute=True)[0],
    )

    # Read the coregistration shifts of the coreg primary
    primary_coreg_azimuth_shifts = lut_shift_exec_products["lut_data"][coreg_primary_image_index][
        "azimuthCoregistrationShifts"
    ]
    primary_coreg_range_shifts = lut_shift_exec_products["lut_data"][coreg_primary_image_index][
        "rangeCoregistrationShifts"
    ]

    # Compute the LUT no-data mask.
    lut_primary_axes = (
        full_primary_axes[0][lut_shift_exec_products["lut_primary_azm_indices"]],
        full_primary_axes[1][lut_shift_exec_products["lut_primary_rng_indices"]],
    )

    lut_nodata_mask = interpolate_on_grid_nn(
        stack_cal_proc_exec_products["stack_nodata_mask"],
        axes_in=(
            full_primary_axes[0] - full_primary_axes[0][0],
            full_primary_axes[1] - full_primary_axes[1][0],
        ),
        axes_out=(
            lut_primary_axes[0] - lut_primary_axes[0][0],
            lut_primary_axes[1] - lut_primary_axes[1][0],
        ),
    )

    # Compute the overall product quality indices.
    quality_indices = _compute_overall_products_quality_indices(
        stack_pre_proc_exec_products=stack_pre_proc_exec_products,
        stack_coreg_exec_products=stack_coreg_exec_products,
        lut_shift_exec_products=lut_shift_exec_products,
        stack_cal_proc_exec_products=stack_cal_proc_exec_products,
        coreg_primary_image_index=coreg_primary_image_index,
        calib_reference_image_index=calib_reference_image_index,
    )

    # Populate the stack ID.
    # NOTE: _compute_stack_unique_id must be called before export_l1c_product.
    stack_id = _compute_stack_unique_id(
        job_order=job_order,
        stack_pre_proc_exec_products=stack_pre_proc_exec_products,
        lut_shift_exec_products=lut_shift_exec_products,
        stack_data_specs=stack_cal_proc_exec_products["stack_data_specs"],
    )
    bps_logger.info("Stack ID: %s", stack_id.to_id())

    # Export the L1c products.
    for image_index in range(len(job_order.input_stack)):
        export_l1c_product(
            job_order=job_order,
            aux_pps=aux_pps,
            stack_pre_proc_exec_products=stack_pre_proc_exec_products,
            stack_coreg_exec_products=stack_coreg_exec_products,
            lut_shift_exec_products=lut_shift_exec_products,
            stack_cal_proc_exec_products=stack_cal_proc_exec_products,
            lut_nodata_mask=lut_nodata_mask,
            full_primary_axes=full_primary_axes,
            calib_reference_image_index=calib_reference_image_index,
            primary_coreg_azimuth_shifts=primary_coreg_azimuth_shifts,
            primary_coreg_range_shifts=primary_coreg_range_shifts,
            overall_product_quality_index=quality_indices[image_index],
            image_index=image_index,
            stack_id=stack_id,
            fnf_mask_file=fnf_mask_file,
            num_threads=num_threads,
        )


def _compute_stack_unique_id(
    job_order: StackJobOrder,
    stack_pre_proc_exec_products: dict,
    lut_shift_exec_products: dict,
    stack_data_specs: StackDataSpecs,
) -> StackUniqueID:
    """Compute the stack's unique ID."""
    coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]
    primary_product = stack_pre_proc_exec_products["l1a_product_data"][coreg_primary_image_index]

    stack_roi = stack_pre_proc_exec_products["stack_roi"]
    if stack_roi is None:
        stack_roi = (
            0,
            0,
            primary_product.raster_info_list[0].lines,
            primary_product.raster_info_list[0].samples,
        )  # The whole frame.

    # Compute the stack's start/stop times.
    stack_start_times = []
    stack_stop_times = []
    for l1a_data, lut_data in zip(
        stack_pre_proc_exec_products["l1a_product_data"],
        lut_shift_exec_products["lut_data"],
    ):
        # The start time is the L1a product's start time, shifted by the
        # azimuth coregitration shifts (and the stack roi).
        start_time = round_precise_datetime(
            l1a_data.raster_info_list[0].lines_start
            + stack_data_specs.azimuth_sampling_step
            * (np.nanmean(lut_data["azimuthCoregistrationShifts"][0]) + stack_roi[0]),
            timespec="microseconds",
        )
        # The stop time is simply the start time increased by the extent of the
        # L1a product data.
        stop_time = round_precise_datetime(
            start_time + (stack_roi[2] - 1) * stack_data_specs.azimuth_sampling_step,
            timespec="microseconds",
        )

        stack_start_times.append(start_time)
        stack_stop_times.append(stop_time)

    return StackUniqueID(
        swath=primary_product.swath_list[0],
        stack_start_time=min(stack_start_times),
        stack_stop_time=max(stack_stop_times),
        mission_phase=primary_product.mission_phase_id,
        global_coverage_id=primary_product.global_coverage_id,
        major_cycle_id=primary_product.major_cycle_id,
        latitude_deg=np.mean(stack_pre_proc_exec_products["stack_footprint"], axis=0)[0],
        longitude_deg=np.mean(stack_pre_proc_exec_products["stack_footprint"], axis=0)[1],
        orbit_direction=primary_product.orbit_direction,
        version=job_order.output_path.product_baseline,
        creation_timestamp=stack_pre_proc_exec_products["stack_creation_stamp"],
    )


def _fill_bps_transcoder_configuration(
    aux_pps: AuxiliaryStaprocessingParameters,
    job_order: StackJobOrder,
    l1a_product: BIOMASSL1Product,
) -> BIOMASSStackProductConfiguration:
    """Populate the transcoder configuration."""
    return BIOMASSStackProductConfiguration(
        frame_number=l1a_product.frame_number,
        frame_status=l1a_product.frame_status,
        product_baseline=job_order.output_path.product_baseline,
        product_doi=aux_pps.l1c_product_export.l1_product_doi,
        product_nodata_value=aux_pps.l1c_product_export.no_pixel_value,
        product_compression_method_abs=aux_pps.l1c_product_export.abs_compression_method.value,
        product_compression_method_phase=aux_pps.l1c_product_export.phase_compression_method.value,
        product_max_z_error_abs=aux_pps.l1c_product_export.abs_max_z_error,
        product_max_z_error_phase=aux_pps.l1c_product_export.phase_max_z_error,
    )


def _update_sar_product_grid(
    calibrated_images: tuple[npt.NDArray[complex], ...],
    stack_sar_data: SARProduct,
    primary_sar_data: SARProduct,
    azimuth_time_axis_shift: float,
    *,
    roi: RegionOfInterest | None = None,
):
    """Populate the calibrated Stack product."""
    # Update the data list.
    if stack_sar_data.polarization_list != primary_sar_data.polarization_list:
        raise RuntimeError("Stack data have mismatching polarizations")

    # Update the product images with the calibrated products.
    stack_sar_data.data_list = calibrated_images

    # Get the new raster info.
    stack_num_lines = primary_sar_data.number_of_lines
    stack_lines_step = primary_sar_data.az_time_interval

    stack_num_samples = primary_sar_data.number_of_samples
    stack_samples_step = primary_sar_data.rg_time_interval
    stack_samples_start = primary_sar_data.first_sample_sr_time

    # Get the new azimuth times.
    stack_start_time = stack_sar_data.start_time
    if roi is not None:
        stack_num_lines = roi[2]
        stack_start_time = stack_sar_data.start_time + roi[0] * primary_sar_data.az_time_interval

    # Update the number of lines, also for the primary. This operation can be
    # done multiple times. It's safe.
    stack_sar_data.number_of_lines = stack_num_lines
    stack_sar_data.number_of_samples = stack_num_samples

    primary_sar_data.number_of_lines = stack_num_lines
    primary_sar_data.number_of_samples = stack_num_samples

    # Update the start/stop time.
    stack_lines_step = primary_sar_data.az_time_interval

    stack_sar_data.start_time = round_precise_datetime(
        stack_start_time + azimuth_time_axis_shift,
        timespec="microseconds",
    )
    stack_sar_data.stop_time = round_precise_datetime(
        stack_sar_data.start_time + (stack_num_lines - 1) * stack_lines_step,
        timespec="microseconds",
    )

    # Update the raster infos.
    for i in range(stack_sar_data.channels):
        stack_raster_info = _reshape_raster_info(
            stack_sar_data.raster_info_list[i],
            lines=stack_num_lines,
            samples=stack_num_samples,
        )
        stack_raster_info.set_lines_axis(
            lines_start=stack_sar_data.start_time,
            lines_start_unit="Utc",
            lines_step=stack_lines_step,
            lines_step_unit="s",
        )
        stack_raster_info.set_samples_axis(
            samples_start=stack_samples_start,
            samples_start_unit="s",
            samples_step=stack_samples_step,
            samples_step_unit="s",
        )
        stack_sar_data.raster_info_list[i] = stack_raster_info

    # Update the burst info. We expect 1 burst per channel.
    for i in range(stack_sar_data.channels):
        stack_sar_data.burst_info_list[i].clear_bursts()
        stack_sar_data.burst_info_list[i].add_burst(
            stack_samples_start,
            stack_sar_data.start_time,
            stack_num_lines,
        )

    # Update the swath info.
    for i in range(stack_sar_data.channels):
        stack_sar_data.swath_info_list[i].acquisition_start_time = stack_sar_data.start_time


def _fill_stack_luts(
    *,
    aux_pps: AuxiliaryStaprocessingParameters,
    calibration_products: dict,
    flattening_phases: tuple[npt.NDArray[float], ...],
    vertical_wavenumbers: tuple[npt.NDArray[float] | None, ...],
    stack_data_specs: StackDataSpecs,
    lut_nodata_mask: npt.NDArray[float],
    lut_primary_axes_indices: tuple[npt.NDArray[int], npt.NDArray[int]],
    full_primary_axes: tuple[npt.NDArray[float], npt.NDArray[float]],
    coreg_azimuth_shifts: npt.NDArray[float],
    coreg_azimuth_geo_shifts: npt.NDArray[float],
    coreg_range_shifts: npt.NDArray[float],
    coreg_range_geo_shifts: npt.NDArray[float],
    primary_coreg_azimuth_shifts: npt.NDArray[float],
    primary_coreg_range_shifts: npt.NDArray[float],
    image_index: int,
    coherence: npt.NDArray[complex] | None,
) -> tuple[
    dict,
    tuple[npt.NDArray[float], npt.NDArray[float]],
    float,
]:
    """Populate the stack LUTs."""
    # Just shortcuts.
    lut_azm_i, lut_rng_j = lut_primary_axes_indices

    # The STA_P LUTs.
    stack_lut = {}

    # Possibly, add coherence and inteferogram.
    if coherence is not None:
        coherence = coherence[lut_azm_i, :][:, lut_rng_j]
        stack_lut["coherence"] = np.abs(coherence) * lut_nodata_mask
        stack_lut["interferogram"] = np.angle(coherence) * lut_nodata_mask

    # The spatial coregistration shifts [m].
    stack_lut["azimuthCoregistrationShifts"] = (
        compute_spatial_azimuth_shifts(
            coreg_azimuth_shifts,
            stack_data_specs.azimuth_sampling_step,
            ground_speed=stack_data_specs.target_ground_speeds[image_index],
            coreg_primary_azimuth_shifts=primary_coreg_azimuth_shifts,
        )
        * lut_nodata_mask
    )
    stack_lut["azimuthOrbitCoregistrationShifts"] = (
        compute_spatial_azimuth_shifts(
            coreg_azimuth_geo_shifts,
            stack_data_specs.azimuth_sampling_step,
            ground_speed=stack_data_specs.target_ground_speeds[image_index],
            coreg_primary_azimuth_shifts=primary_coreg_azimuth_shifts,
        )
        * lut_nodata_mask
    )
    stack_lut["rangeCoregistrationShifts"] = (
        compute_spatial_range_shifts(
            coreg_range_shifts,
            stack_data_specs.range_sampling_step,
            coreg_primary_range_shifts=primary_coreg_range_shifts,
        )
        * lut_nodata_mask
    )
    stack_lut["rangeOrbitCoregistrationShifts"] = (
        compute_spatial_range_shifts(
            coreg_range_geo_shifts,
            stack_data_specs.range_sampling_step,
            coreg_primary_range_shifts=primary_coreg_range_shifts,
        )
        * lut_nodata_mask
    )

    if "coregistrationShiftsQuality" in stack_lut:
        stack_lut["coregistrationShiftsQuality"] *= lut_nodata_mask

    # The vertical wavenumbers (Kz).
    kz = vertical_wavenumbers[image_index]
    if kz is None:
        raise RuntimeError("Vertical wavenumbers cannot be None")

    # The DSI data (phi_flat).
    flattening_phases = flattening_phases[image_index]
    if flattening_phases is None:
        raise RuntimeError("Flattening phases cannot be None")

    # We guarantee consistency between L1c and L2, we ensure that we compute
    # the flattning phases screens exactly as computed in the SKP (if enabled).
    # Populate DSI's and Kz's, by default we use 64-bit for those.
    dsi_dtype = EstimationDType.from_32bit_flag(use_32bit_flag=aux_pps.skp_phase_calibration.use_32bit_flag).float_dtype

    azimuth_window_size = max(2 * np.diff(lut_azm_i[0:2]) - 1, 1)
    if azimuth_window_size >= 3:
        F_azm, _ = build_sparse_uniform_filter_matrix(
            input_size=full_primary_axes[0].size,
            axis=0,
            subsampling_step=lut_azm_i,
            uniform_filter_window_size=azimuth_window_size,
            border_type=ConvolutionBorderType.ISOLATED,
            dtype=dsi_dtype,
        )
        kz = F_azm @ kz
        flattening_phases = F_azm @ flattening_phases.astype(dsi_dtype)
    else:
        kz = kz[lut_azm_i, :]
        flattening_phases = flattening_phases[lut_azm_i, :].astype(dsi_dtype)

    range_window_size = max(2 * np.diff(lut_rng_j[0:2]) - 1, 1)
    if range_window_size >= 3:
        F_rng, _ = build_sparse_uniform_filter_matrix(
            input_size=full_primary_axes[1].size,
            axis=1,
            subsampling_step=lut_rng_j,
            uniform_filter_window_size=range_window_size,
            border_type=ConvolutionBorderType.ISOLATED,
            dtype=dsi_dtype,
        )
        kz = kz @ F_rng
        flattening_phases = flattening_phases.astype(dsi_dtype) @ F_rng
    else:
        kz = kz[:, lut_rng_j]
        flattening_phases = flattening_phases[:, lut_rng_j].astype(dsi_dtype)

    stack_lut["waveNumbers"] = kz * lut_nodata_mask
    stack_lut["flatteningPhaseScreen"] = flattening_phases * lut_nodata_mask

    # Populate the ionosphere phase screens.
    if (
        MSC_NAME in calibration_products
        and calibration_products[MSC_NAME]["is_solution_usable"]
        and calibration_products[MSC_NAME]["ionosphere_estimation_outputs"][image_index] is not None
    ):
        iono_output = calibration_products[MSC_NAME]["ionosphere_estimation_outputs"][image_index]
        stack_lut["ionospherePhaseScreens"] = iono_output.iono_layer_phase_screens

    # The SKP-calibration products.
    if SKP_NAME in calibration_products and calibration_products[SKP_NAME]["is_solution_usable"]:
        skp_products = calibration_products[SKP_NAME]
        stack_lut["skpCalibrationPhaseScreen"] = (
            skp_products["skp_calibration_phase_screen"][image_index] * lut_nodata_mask
        )
        stack_lut["skpCalibrationPhaseScreenQuality"] = (
            skp_products["skp_calibration_phase_screen_quality"] * lut_nodata_mask
        )

        dsi_mismatch = np.max(np.abs(skp_products["skp_flattening_phase_screen"][image_index] - flattening_phases))
        if dsi_mismatch > np.finfo(np.float32).eps:
            raise RuntimeError(
                f"DSI used by {SKP_NAME} are not consistent with flatteningPhaseScreen. Mismatch [rad]: {dsi_mismatch}"
            )

    return (
        stack_lut,
        np.nanmean(coreg_azimuth_shifts[0]) * stack_data_specs.azimuth_sampling_step,  # [s].
    )


def _fill_stack_insar_parameters(
    *,
    job_order: StackJobOrder,
    spatial_ordering: tuple[int, ...],
    calibration_products: dict,
    stack_data_specs: StackDataSpecs,
    calib_reference_image_index: int,
    image_index: int,
    incidence_angle: float,
) -> BIOMASSStackInSARParameters:
    """
    Populate a BIOMASSStackInSARParameters object.

    Parameters
    ----------
    job_order: StackJobOrder
        The STA_P JobOrder object.

    spatial_ordering: tuple[int, ...]
        The index permutation that increasingly orders the stack
        images wrt the spatial (normal) baselines.

    calibration_products: int
        The calibration products.

    stack_data_specs: StackDataSpecs
        Specifications regarding the stack data.

    calib_reference_image_index: int
        The calibration reference image index.

    image_index: int
        The current image index.

    Return
    ------
    BIOMASSStackInSARParameters
        The stack InSAR parameters of the job.

    IonosphereAxes | None
        The ionosphere estimation annotation object.

    """
    # Some optional by-products.
    azimuth_central_frequency = (
        stack_data_specs.azimuth_central_frequency if stack_data_specs.azimuth_central_frequency is not None else 0.0
    )
    azimuth_common_bandwidth = (
        stack_data_specs.azimuth_bandwidths[0] if stack_data_specs.azimuth_central_frequency is not None else 0.0
    )

    # The products of the AZF, if enabled.
    if AZF_NAME in calibration_products and calibration_products[AZF_NAME]["is_solution_usable"]:
        azf_products = calibration_products[AZF_NAME]
        azimuth_common_bandwidth = azf_products["azimuth_common_bandwidth"]
        azimuth_central_frequency = azf_products["azimuth_central_frequency"]

    # The products of the IOB, if enabled.
    slow_ionosphere_azimuth_phase_screen = 0.0
    slow_ionosphere_range_phase_screen = 0.0
    slow_ionosphere_quality = 0.0
    slow_ionosphere_removal_interferometric_pairs = []
    if IOB_NAME in calibration_products and calibration_products[IOB_NAME]["is_solution_usable"]:
        iob_products = calibration_products[IOB_NAME]
        slow_ionosphere_azimuth_phase_screen = iob_products["slow_ionosphere_azimuth_phase_screens"][image_index]
        slow_ionosphere_range_phase_screen = iob_products["slow_ionosphere_range_phase_screens"][image_index]
        slow_ionosphere_quality = iob_products["slow_ionosphere_qualities"][image_index]
        slow_ionosphere_removal_interferometric_pairs = iob_products["interferometric_pairs"]

    # The products of the MSC module.
    low_resolution_flag = False
    azimuth_phase_slope = 0.0
    range_phase_slope = 0.0
    azimuth_shift = 0.0
    azimuth_frequency_centroid = 0.0
    iono_layer_altitudes = []
    coherence_check_value = 0.0
    iono_correction_flag = False
    if MSC_NAME in calibration_products and calibration_products[MSC_NAME]["is_solution_usable"]:
        msc_products = calibration_products[MSC_NAME]
        low_resolution_flag = msc_products["low_resolution_switches"][image_index]
        azimuth_phase_slope = msc_products["azimuth_phase_slopes"][image_index]
        range_phase_slope = msc_products["range_phase_slopes"][image_index]
        iono_correction_flag = msc_products["ionosphere_estimation_outputs"][image_index] is not None
        if iono_correction_flag:
            coherence_check_value = msc_products["coherence_checks"][image_index].improvement
            azimuth_shift = msc_products["azimuth_shift_outputs"][image_index].azimuth_shift
            azimuth_frequency_centroid = msc_products["azimuth_shift_outputs"][image_index].azimuth_frequency_centroid
            iono_layer_altitudes = los_range_ground_to_altitude(
                msc_products["ionosphere_estimation_outputs"][image_index].ionosphere_layer_ranges,
                incidence_angle=incidence_angle,
            )

    # The SKP-calibration products.
    skp_calibration_phase_screen_mean = 0.0
    skp_calibration_phase_screen_std = 0.0
    skp_calibration_phase_screen_var = 0.0
    skp_calibration_phase_screen_mad = 0.0
    if SKP_NAME in calibration_products and calibration_products[SKP_NAME]["is_solution_usable"]:
        skp_calibration_phases_statistics_current_frame = calibration_products[SKP_NAME][
            "skp_calibration_phases_statistics"
        ][image_index]
        skp_calibration_phase_screen_mean = skp_calibration_phases_statistics_current_frame.avg_phase
        skp_calibration_phase_screen_std = skp_calibration_phases_statistics_current_frame.std_phase
        skp_calibration_phase_screen_var = skp_calibration_phases_statistics_current_frame.var_phase
        skp_calibration_phase_screen_mad = skp_calibration_phases_statistics_current_frame.mad_phase

    # The reference image name.
    calib_reference_image_name = job_order.input_stack[calib_reference_image_index].name

    return BIOMASSStackInSARParameters(
        calibration_primary_image=str(calib_reference_image_name),
        azimuth_common_bandwidth=azimuth_common_bandwidth,
        azimuth_central_frequency=azimuth_central_frequency,
        slow_ionosphere_range_phase_screen=slow_ionosphere_range_phase_screen,
        slow_ionosphere_azimuth_phase_screen=slow_ionosphere_azimuth_phase_screen,
        slow_ionosphere_quality=slow_ionosphere_quality,
        slow_ionosphere_removal_interferometric_pairs=slow_ionosphere_removal_interferometric_pairs,
        multi_squint_calibration_low_resolution_flag=low_resolution_flag,
        multi_squint_calibration_azimuth_phase_slope=azimuth_phase_slope,
        multi_squint_calibration_range_phase_slope=range_phase_slope,
        multi_squint_calibration_azimuth_shift=azimuth_shift,
        multi_squint_calibration_azimuth_frequency_centroid=azimuth_frequency_centroid,
        multi_squint_calibration_ionospheric_layer_altitudes=iono_layer_altitudes,
        multi_squint_calibration_coherence_check_value=coherence_check_value,
        multi_squint_calibration_ionosphere_correction_flag=iono_correction_flag,
        baseline_ordering_index=int(spatial_ordering[image_index]),
        skp_calibration_phase_screen_mean=skp_calibration_phase_screen_mean,
        skp_calibration_phase_screen_std=skp_calibration_phase_screen_std,
        skp_calibration_phase_screen_var=skp_calibration_phase_screen_var,
        skp_calibration_phase_screen_mad=skp_calibration_phase_screen_mad,
    )


def _fill_stack_quality(
    *,
    aux_pps: AuxiliaryStaprocessingParameters,
    calibration_products: dict,
    stack_polarizations: tuple[EPolarization, ...],
    rfi_indices: dict[EPolarization, float],
    faraday_decorrelation: float,
    invalid_residual_shifts_ratio: float,
    overall_product_quality_index: np.uint32,
    coherences: list[npt.NDArray[float]],
    invalid_l1a_samples_ratio: float = 0.0,
) -> BIOMASSStackQuality:
    """Fill the BIOMASSStackQuality structure"""
    # The SKP validity.
    invalid_skp_calibration_phase_screen_ratio = 0.0
    skp_decomposition_index = 0
    if SKP_NAME in calibration_products:
        skp_product = calibration_products[SKP_NAME]
        skp_decomposition_index = skp_product["skp_decomposition_index"]
        if skp_product["is_solution_usable"]:
            invalid_skp_calibration_phase_screen_ratio = skp_product["invalid_skp_calibration_phase_screen_ratio"]

    # NOTE: We need to force a conversion to int since np.uint32 is not
    # supported by XSD.
    return BIOMASSStackQuality(
        overall_product_quality_index=int(overall_product_quality_index),
        sta_quality_parameters_list=[
            BIOMASSStackQualityParameters(
                invalid_l1a_data_samples=invalid_l1a_samples_ratio,
                rfi_decorrelation=rfi_indices[polarization],
                faraday_decorrelation=faraday_decorrelation,
                faraday_decorrelation_threshold=aux_pps.primary_image_selection.faraday_decorrelation_threshold,
                rfi_decorrelation_threshold=aux_pps.primary_image_selection.rfi_decorrelation_threshold,
                invalid_residual_shifts_ratio=invalid_residual_shifts_ratio,
                residual_shifts_quality_threshold=aux_pps.coregistration.residual_shift_quality_threshold,
                invalid_skp_calibration_phase_screen_ratio=invalid_skp_calibration_phase_screen_ratio,
                skp_calibration_phase_screen_quality_threshold=aux_pps.skp_phase_calibration.skp_calibration_phase_screen_quality_threshold,
                skp_decomposition_index=skp_decomposition_index,
                polarization=translate_polarization(polarization),
                coherence_mean=np.nanmean(coherence),
                coherence_std=np.nanstd(coherence),
                coherence_min=np.nanmin(coherence),
                coherence_max=np.nanmax(coherence),
                coherence_median=np.nanmedian(coherence),
                coherence_histogram=(
                    BIOMASSStackQualityParameters.CoherenceHistogram.from_coherence_abs(coherence)
                    if coherence.size > 1
                    else BIOMASSStackQualityParameters.CoherenceHistogram(
                        bins=np.array([], dtype=np.float64),
                        pdf_values=np.array([], dtype=np.float64),
                    )
                ),
            )
            for polarization, coherence in zip(stack_polarizations, coherences)
        ],
    )


def _fill_stack_processing_parameters(
    *,
    aux_pps: AuxiliaryStaprocessingParameters,
    primary_image_info: common.PrimaryImageSelectionInformationType,
    processed_polarizations: tuple[EPolarization, ...],
    cross_pol_method: common.PolarisationCombinationMethodType,
    l1a_product: BIOMASSL1Product,
    actualized_coregistration_parameters: dict,
) -> BIOMASSStackProcessingParameters:
    """Populate a BIOMASSStackProcessingParameters object."""

    def _map_selection_info_to_selection_method(
        info: common.PrimaryImageSelectionInformationType,
    ) -> PrimaryImageSelectionMethodType:
        if info == common.PrimaryImageSelectionInformationType.GEOMETRY:
            return PrimaryImageSelectionMethodType.GEOMETRY

        if info in (
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
        ):
            return PrimaryImageSelectionMethodType.GEOMETRY_AND_QUALITY

        if info == common.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE:
            return PrimaryImageSelectionMethodType.TEMPORAL_BASELINE

        raise RuntimeError(f"Unknown primary image selection information type: {info}")

    return BIOMASSStackProcessingParameters(
        processor_version=VERSION,
        product_generation_time=PreciseDateTime.now(),
        polarizations_used=len(processed_polarizations),
        polarization_combination_method=translate_common.translate_polarisation_combination_method_to_model(
            cross_pol_method
        ),
        primary_image_selection_method=_map_selection_info_to_selection_method(primary_image_info),
        coregistration_method=actualized_coregistration_parameters["coregistration_method"],
        height_model=HeightModelType(
            value=l1a_product.height_model.value,
            version=l1a_product.height_model.version,
        ),
        rfi_degradation_estimation_flag=aux_pps.rfi_degradation_estimation.rfi_degradation_estimation_flag,
        azimuth_spectral_filtering_flag=aux_pps.azimuth_spectral_filtering.azimuth_spectral_filtering_flag,
        calibration_primary_image_flag=aux_pps.slow_ionosphere_removal.primary_image_flag,
        polarization_used_for_slow_ionosphere_removal=translate_polarization(
            aux_pps.slow_ionosphere_removal.polarization_used
        ),
        polarization_used_for_multi_squint_calibration=translate_polarization(
            aux_pps.multi_squint_calibration.polarization_used,
        ),
        slow_ionosphere_removal_flag=aux_pps.slow_ionosphere_removal.slow_ionosphere_removal_flag,
        multi_squint_calibration_flag=aux_pps.multi_squint_calibration.multi_squint_calibration_flag,
        multi_squint_calibration_force_ionosphere_correction_flag=aux_pps.multi_squint_calibration.force_multi_squint_iono_correction_flag,
        multi_squint_calibration_disable_ionosphere_correction_flag=aux_pps.multi_squint_calibration.disable_multi_squint_iono_correction_flag,
        multi_squint_calibration_coherence_improvement_threshold=aux_pps.multi_squint_calibration.multi_squint_coherence_improvement_threshold,
        multi_squint_calibration_coherence_azimuth_window_size=aux_pps.multi_squint_calibration.coherence_azimuth_window_size,
        multi_squint_calibration_coherence_range_window_size=aux_pps.multi_squint_calibration.coherence_range_window_size,
        multi_squint_calibration_multi_squint_coherence_subband_resolution=aux_pps.multi_squint_calibration.multi_squint_coherence_subband_resolution,
        multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size=aux_pps.multi_squint_calibration.multi_squint_coherence_high_res_azimuth_window_size,
        multi_squint_calibration_multi_squint_high_res_coherence_range_window_size=aux_pps.multi_squint_calibration.multi_squint_coherence_high_res_range_window_size,
        multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size=aux_pps.multi_squint_calibration.multi_squint_coherence_low_res_azimuth_window_size,
        multi_squint_calibration_multi_squint_low_res_coherence_range_window_size=aux_pps.multi_squint_calibration.multi_squint_coherence_low_res_range_window_size,
        skp_phase_calibration_flag=aux_pps.skp_phase_calibration.skp_phase_estimation_flag,
        skp_phase_correction_flag=skp_phase_correction_flag(aux_pps.skp_phase_calibration),
        skp_phase_correction_flattening_only_flag=skp_phase_correction_flattening_only_flag(
            aux_pps.skp_phase_calibration
        ),
        skp_estimation_window_size=aux_pps.skp_phase_calibration.estimation_window_size,
        skp_calibration_phase_postprocessing=aux_pps.skp_phase_calibration.calibration_phase_postprocessing,
        skp_postprocessing_filter_window_size=aux_pps.skp_phase_calibration.postprocessing_filter_window_size,
        skp_goldstein_filter_window_size=aux_pps.skp_phase_calibration.goldstein_filter_window_size,
        slow_ionosphere_removal_multi_baseline_threshold=(
            aux_pps.slow_ionosphere_removal.multi_baseline_critical_baseline_threshold
        ),
        slow_ionosphere_removal_use_32bit_flag=aux_pps.slow_ionosphere_removal.use_32bit_flag,
        azimuth_spectral_filtering_use_32bit_flag=aux_pps.azimuth_spectral_filtering.use_32bit_flag,
        multi_squint_estimation_use_32bit_flag=aux_pps.multi_squint_calibration.use_32bit_estimation_flag,
        multi_squint_correction_use_32bit_flag=aux_pps.multi_squint_calibration.use_32bit_correction_flag,
        skp_phase_calibration_use_32bit_flag=aux_pps.skp_phase_calibration.use_32bit_flag,
    )


def _fill_ionosphere_axes(
    *,
    calibration_products: dict,
    stack_specs: StackDataSpecs,
    image_index: int,
    incidence_angle: float,
) -> IonosphereAxes | None:
    """Fill the ionosphere estimation annotation."""
    if (
        MSC_NAME in calibration_products
        and calibration_products[MSC_NAME]["is_solution_usable"]
        and calibration_products[MSC_NAME]["ionosphere_estimation_outputs"][image_index] is not None
    ):
        iono_output = calibration_products[MSC_NAME]["ionosphere_estimation_outputs"][image_index]
        iono_slant_range_axis = calibration_products[MSC_NAME]["ionosphere_slant_range_space_axes"][image_index]
        ionosphere_azimuth_axis, ionosphere_slant_range_axis = compute_ionosphere_axes(
            ionosphere_azimuth_space_axis=iono_output.azimuth_iono_space_axis,
            ionosphere_slant_range_space_axis=iono_slant_range_axis,
            ionosphere_los_range_ground=np.nanmean(iono_output.ionosphere_layer_ranges),
            satellite_ground_speed=stack_specs.satellite_ground_speeds[image_index],
            incidence_angle=incidence_angle,
            earth_radius=stack_specs.earth_radii[image_index],
        )
        return IonosphereAxes(
            ionosphere_azimuth_axis=ionosphere_azimuth_axis,
            ionosphere_slant_range_axis=ionosphere_slant_range_axis,
        )
    return None


def _fill_stack_coregistration_parameters(
    *,
    aux_pps: AuxiliaryStaprocessingParameters,
    primary_image_info: common.PrimaryImageSelectionInformationType,
    primary_image: str,
    secondary_image: str,
    azimuth_coreg_shifts: npt.NDArray[float],
    range_coreg_shifts: npt.NDArray[float],
    normal_baseline: float,
    l1a_product: BIOMASSL1Product,
) -> BIOMASSStackCoregistrationParameters:
    """Populate a BIOMASSStackProcessingParameters object."""

    average_azimuth_coreg_shifts = aux_pps.l1c_product_export.no_pixel_value
    if not np.all(np.isnan(azimuth_coreg_shifts)):
        average_azimuth_coreg_shifts = np.nanmean(azimuth_coreg_shifts)

    average_range_coreg_shifts = aux_pps.l1c_product_export.no_pixel_value
    if not np.all(np.isnan(range_coreg_shifts)):
        average_range_coreg_shifts = np.nanmean(range_coreg_shifts)

    return BIOMASSStackCoregistrationParameters(
        datum=translate_common.translate_datum_to_model(l1a_product.datum),
        primary_image=primary_image,
        secondary_image=secondary_image,
        primary_image_selection_information=translate_common.translate_primary_image_selection_information_to_model(
            primary_image_info
        ),
        average_azimuth_coregistration_shift=average_azimuth_coreg_shifts,
        average_range_coregistration_shift=average_range_coreg_shifts,
        normal_baseline=normal_baseline,
        polarization_used=translate_polarization(aux_pps.coregistration.polarization_used),
    )


def _export_to_l1c_format(
    *,
    input_product: BIOMASSL1Product,
    product_primary: SARProduct,
    output_folder: Path,
    path_primary_l1a: Path,
    path_secondary_l1a: Path,
    l1c_export_conf: L1cProductExportConf,
    configuration: BIOMASSStackProductConfiguration,
    coherence: np.ndarray | None,
    lut_dict: dict,
    lut_axes_primary: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]],
    ionosphere_axes: IonosphereAxes | None,
    stack_processing_parameters: BIOMASSStackProcessingParameters,
    stack_coregistration_parameters: BIOMASSStackCoregistrationParameters,
    stack_in_sarparameters: BIOMASSStackInSARParameters,
    stack_quality: BIOMASSStackQuality,
    stack_nodata_mask: npt.NDArray[float],
    stack_footprint: list[list[float]],
    stack_id: StackUniqueID,
    input_stack: tuple[str, ...],
    file_name_aux_pps: str | None = None,
    file_name_fnf: str | None = None,
    num_threads: int = 1,
    add_monitoring_product: bool = True,
):
    """Export data to L1c product."""
    # The Quick-look exporting configurations.
    ql_conf = QuickLookConf(
        l1c_colour_coding_method=l1c_export_conf.l1c_ql_colour_coding_method,
        azimuth_decimation_factor=l1c_export_conf.ql_azimuth_decimation_factor,
        azimuth_averaging_factor=l1c_export_conf.ql_azimuth_averaging_factor,
        range_decimation_factor=l1c_export_conf.ql_range_decimation_factor,
        range_averaging_factor=l1c_export_conf.ql_range_averaging_factor,
        absolute_scaling_factor=l1c_export_conf.ql_absolute_scaling_factor,
    )

    # Export standard product.
    output_standard = BIOMASSStackProduct(
        product=input_product,
        product_primary=product_primary,
        is_monitoring=False,
        configuration=configuration,
        stack_processing_parameters=stack_processing_parameters,
        stack_coregistration_parameters=stack_coregistration_parameters,
        stack_in_sarparameters=stack_in_sarparameters,
        stack_quality=stack_quality,
        mission_phase_id=input_product.mission_phase_id,
        datatake_id=input_product.datatake_id,
        orbit_number=input_product.orbit_number,
        global_coverage_id=input_product.global_coverage_id,
        repeat_cycle_id=input_product.repeat_cycle_id,
        major_cycle_id=input_product.major_cycle_id,
        track_number=input_product.track_number,
        platform_heading=input_product.platform_heading,
        number_of_lines=product_primary.number_of_lines,
        number_of_samples=product_primary.number_of_samples,
        rg_time_interval=product_primary.rg_time_interval,
        az_time_interval=product_primary.az_time_interval,
        stack_footprint=stack_footprint,
    )

    bps_logger.info("Writing L1c S-product %s", output_standard.name)
    BIOMASSStackProductWriter(
        product=output_standard,
        output_dir=output_folder,
        source_product1_path=path_primary_l1a,
        source_product2_path=path_secondary_l1a,
        file_name_aux_pps=file_name_aux_pps,
        file_name_fnf=file_name_fnf,
        source_product_names=input_stack,
        product_lut=lut_dict,
        lut_axes_primary=lut_axes_primary,
        ionosphere_axes=ionosphere_axes,
        coherence=coherence,
        ql_conf=ql_conf,
        ql_export_coherence_flag=l1c_export_conf.ql_export_coherence_flag,
        ql_export_interferogram_flag=l1c_export_conf.ql_export_interferogram_flag,
        stack_nodata_mask=stack_nodata_mask,
        processor_name=BPS_STACK_PROCESSOR_NAME,
        processor_version=VERSION,
        stack_id=stack_id,
        num_threads=num_threads,
    ).write()

    # If required, export monitoring product too. A monitoring product is an
    # L1c product that differs from the standard product by not having the
    # images/frames.
    if add_monitoring_product:
        output_monitoring = BIOMASSStackProduct(
            product=input_product,
            product_primary=product_primary,
            is_monitoring=True,
            configuration=configuration,
            stack_processing_parameters=stack_processing_parameters,
            stack_coregistration_parameters=stack_coregistration_parameters,
            stack_in_sarparameters=stack_in_sarparameters,
            stack_quality=stack_quality,
            mission_phase_id=input_product.mission_phase_id,
            datatake_id=input_product.datatake_id,
            orbit_number=input_product.orbit_number,
            global_coverage_id=input_product.global_coverage_id,
            repeat_cycle_id=input_product.repeat_cycle_id,
            major_cycle_id=input_product.major_cycle_id,
            track_number=input_product.track_number,
            platform_heading=input_product.platform_heading,
            number_of_lines=product_primary.number_of_lines,
            number_of_samples=product_primary.number_of_samples,
            rg_time_interval=product_primary.rg_time_interval,
            az_time_interval=product_primary.az_time_interval,
            stack_footprint=stack_footprint,
        )

        bps_logger.info("Writing L1c M-product %s", output_monitoring.name)
        BIOMASSStackProductWriter(
            product=output_monitoring,
            output_dir=output_folder,
            source_product1_path=path_primary_l1a,
            source_product2_path=path_secondary_l1a,
            file_name_aux_pps=file_name_aux_pps,
            file_name_fnf=file_name_fnf,
            source_product_names=input_stack,
            coherence=coherence,
            product_lut=lut_dict,
            lut_axes_primary=lut_axes_primary,
            ionosphere_axes=ionosphere_axes,
            ql_conf=ql_conf,
            ql_export_coherence_flag=l1c_export_conf.ql_export_coherence_flag,
            ql_export_interferogram_flag=l1c_export_conf.ql_export_interferogram_flag,
            stack_nodata_mask=stack_nodata_mask,
            processor_name=BPS_STACK_PROCESSOR_NAME,
            processor_version=VERSION,
            stack_id=stack_id,
        ).write()


def _compute_overall_products_quality_indices(
    *,
    stack_pre_proc_exec_products: dict,
    stack_coreg_exec_products: dict,
    lut_shift_exec_products: dict,
    stack_cal_proc_exec_products: dict,
    coreg_primary_image_index: int,
    calib_reference_image_index: int,
) -> npt.NDArray[np.uint32]:
    """Pack the overall quality input bitset."""
    # Some shortcuts.
    num_images = len(stack_cal_proc_exec_products["calibrated_stack_images"])
    cal_products = stack_cal_proc_exec_products["calibration_products"]

    # Input and preprocessor bitsets.
    input_quality_bitset = stack_pre_proc_exec_products["input_quality_bitset"]
    preproc_quality_bitset = stack_pre_proc_exec_products["preproc_quality_bitset"]

    # Coregistration bitsets (bit #3 and #4 are unassigned).
    coreg_quality_bitset = np.zeros((num_images, 4), dtype=np.uint8)
    coreg_quality_bitset[:, 0] = stack_coreg_exec_products["quality_bitset"]
    coreg_quality_bitset[:, 1] = lut_shift_exec_products["quality_bitset"]

    # Calibration bitsets.
    cal_quality_bitset = np.zeros((num_images, 20), dtype=np.uint8)
    if _has_calibration_product(cal_products, AZF_NAME):
        cal_quality_bitset[:, 0:2] = cal_products[AZF_NAME]["quality_bitset"]
    if _has_calibration_product(cal_products, IOB_NAME):
        cal_quality_bitset[:, 2:6] = cal_products[IOB_NAME]["quality_bitset"]
    if _has_calibration_product(cal_products, SKP_NAME):
        cal_quality_bitset[:, 14:20] = cal_products[SKP_NAME]["quality_bitset"]

    # Stack bitsets.
    stack_quality_bitset = np.zeros((num_images, 3), dtype=np.uint8)
    stack_quality_bitset[:, 0] = input_quality_bitset[coreg_primary_image_index]
    stack_quality_bitset[:, 1] = input_quality_bitset[calib_reference_image_index]
    for i in range(num_images):
        stack_quality_bitset[i, 2] = np.any(input_quality_bitset[np.arange(num_images) != i])

    # Pack the all quality bitsets.
    quality_bitsets = np.hstack(
        [
            input_quality_bitset,
            preproc_quality_bitset,
            coreg_quality_bitset,
            cal_quality_bitset,
            stack_quality_bitset,
        ],
    )

    return np.array(
        [StackQualityIndex.from_bitset(bitset).encode() for bitset in quality_bitsets],
        dtype=np.uint32,
    )


def _reshape_raster_info(
    raster_info: RasterInfo,
    *,
    lines: int | None = None,
    samples: int | None = None,
) -> RasterInfo:
    """Return a duplicate a RasterInfo but with a new shape."""
    return RasterInfo(
        lines=lines if lines is not None else raster_info.lines,
        samples=samples if samples is not None else raster_info.samples,
        celltype=raster_info.cell_type,
        filename=raster_info.file_name,
        header_offset_bytes=raster_info.header_offset_bytes,
        row_prefix_bytes=raster_info.row_prefix_bytes,
        byteorder=raster_info.byte_order,
        invalid_value=raster_info.invalid_value,
        format_type=raster_info.format_type,
    )


def _has_calibration_product(
    calibration_products: dict,
    cal_module_name: Literal[AZF_NAME, MSC_NAME, SKP_NAME],
) -> bool:
    """Check if the solution is valid for selected module name."""
    return cal_module_name in calibration_products


def _slant_range_space_to_slant_range_time_polycoeffs(
    coeffs_x_sr: npt.NDArray[float],
    t0: float,
    *,
    reverse_order: bool,
) -> list[float]:
    r"""
    Convert the coefficients of a polynomial expressed in spatial ground-range
    to the coefficients of a polynomial in temporal relative slant-range.

    The method applies the following relation: Set t{sr} and x{gr} to
    respectively be the temporal slant-range and the spatial ground-range. Then
    x{sr} = t{sr} * c/2, with a the incidence angle. It follows that, setting
    respectively x'{sr} and t'{sr} to be the relative spatial and temporal slant
    range axes:

          a{0} + a{1} * x{sr} =
        = a{0} + a{1} * (x{sr,0} + x'{sr})
        = a{0} + a{1} * (c/2) * (t{sr,0} + t'{sr})
        = (a{0} + a{1} * c/2 * t{sr,0}) + (a{1} * c/2) * t'{sr}

    Hence,

        b{0} = a{0} + a{1} * c/2 * t{sr,0}
        b{1} = a{1} * c/2

    Parameters
    ----------
    coeffs_x_sr: npt.NDArray[float]
        The coefficients in spatial ground-range.

    t0: float [s]
        The slant-range start time.

    reverse_order: bool
        If true, revert the order of the coefficients.

    Return
    ------
    list[float]
        The list of coefficients.

    """
    # If a polynomial is identically 0, np.polynomial.Polynomial.convert() converts
    # it into a deg=0 polynomial.
    if np.all(coeffs_x_sr == 0):
        coeffs_x_sr = np.zeros((2,))
    if coeffs_x_sr.size != 2:
        raise ValueError(f"Excepted a polynomial of degree 1, got {coeffs_x_sr=}")

    if reverse_order:
        coeffs_x_sr = coeffs_x_sr[::-1]

    return [
        float(coeffs_x_sr[0] + coeffs_x_sr[1] * sp.constants.speed_of_light / 2 * t0),
        float(coeffs_x_sr[1] * sp.constants.speed_of_light / 2),
    ]


def _compute_coherences(
    *,
    calibrated_stack: tuple[tuple[npt.NDArray[complex], ...], ...],
    flattening_phases: tuple[npt.NDArray[float], ...] | None,
    coreg_primary_image_index: int,
    image_index: int,
    num_threads: int,
) -> list[npt.NDArray[float] | None]:
    """Compute the coherence of a secondary image."""
    if image_index == coreg_primary_image_index:
        return [None] * len(calibrated_stack[image_index])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # The primary frame and DSI.
        image_p = calibrated_stack[coreg_primary_image_index]
        dsi_p = flattening_phases[coreg_primary_image_index]

        # The secondary frame and DSI.
        image_s = calibrated_stack[image_index]
        dsi_s = flattening_phases[image_index]

        # The coherence main runner.
        def compute_coherence_fn(pol: int) -> npt.NDArray[complex]:
            return compute_coherence(
                image_p=image_p[pol] * np.exp(1j * dsi_p, dtype=np.complex64),
                image_s=image_s[pol] * np.exp(1j * dsi_s, dtype=np.complex64),
            )

        return list(executor.map(compute_coherence_fn, range(len(image_p))))


def _abs_or_zero(values: npt.NDArray | None) -> npt.NDArray[float]:
    """Return the abs of `values` if not None, else `default_value`"""
    return np.abs(values) if values is not None else np.zeros((1,))
