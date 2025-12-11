# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Stack Pre-Processor Execution Manager
-----------------------------------------
"""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.io.metadata import EPolarization, ESideLooking, StateVectors
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import MissionPhaseID, bps_logger
from bps.common.io import common
from bps.common.roi_utils import raise_if_roi_is_invalid
from bps.common.toi_utils import TimeOfInterest
from bps.stack_cal_processor.core.utils import get_time_axis
from bps.stack_pre_processor.core.geometry import compute_ecef_dem
from bps.stack_pre_processor.core.preprocessing import (
    baseline_ordering,
    compute_coreg_primary_image_index,
    compute_critical_baseline,
    compute_faraday_decorrelation_index,
    compute_rfi_degradation_indices,
    compute_spatial_baselines,
    compute_temporal_baselines,
    cross_pol_merging,
    prepare_stack_data,
)
from bps.stack_pre_processor.core.utils import (
    StackPreProcessorRuntimeError,
    common_polarizations,
    compute_processing_footprint,
    compute_processing_roi,
)
from bps.stack_pre_processor.interface import write_pre_processor_dem_product
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.internal.intermediates import (
    StackPreProcessorOutputProducts,
)
from bps.transcoder.io import common_annotation_l1
from bps.transcoder.sarproduct.biomass_l1product import BIOMASSL1Product
from bps.transcoder.sarproduct.biomass_l1product_reader import BIOMASSL1ProductReader
from bps.transcoder.sarproduct.l1.quality_index import L1QualityIndex

# Limits in seconds, for what swath length is considered to be nominal.
SWATH_DURATION_MIN = 20.5  # [s].
SWATH_DURATION_MAX = 21.5  # [s].

# Length of the compressed creation time string.
CREATION_STAMP_LENGTH = 6

# These ones are required all the times.
GEOMETRY_LUT_NAMES = [
    "latitude",
    "longitude",
    "height",
    "elevationAngle",
    "incidenceAngle",
    "terrainSlope",
]

# File that flags that the pre-processor was successfully executed.
STOP_AND_RESUME_PATH = "PRE_PROCESSING_COMPLETE.txt"


class InvalidL1ProductLUTError(RuntimeError):
    """Handle missing or invalid LUTs in an L1a product."""

    def __init__(self, l1a_product_name: str, missing_luts: dict[str, str]):
        super().__init__(
            "LUT(s) in L1a product {} are missing or invalid: {}".format(
                l1a_product_name,
                missing_luts,
            )
        )


class StackPreProcessorExecutionManager:
    """
    Manage the execution of the Stack Pre-Processor.

    Parameters
    ----------
    job_order: StackJobOrder
        The STA_P JobOrder object.

    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS object.

    """

    def __init__(
        self,
        *,
        job_order: StackJobOrder,
        aux_pps: AuxiliaryStaprocessingParameters,
    ):
        """Instantiate the object."""
        self.job_order = job_order
        self.aux_pps = aux_pps

    def run(
        self,
        *,
        stack_pre_proc_output_products: tuple[StackPreProcessorOutputProducts, ...],
        breakpoint_dir: Path,
        num_worker_threads: int = 1,
    ) -> dict:
        """
        Run the pre-processor multithreaded.

        Parameters
        ----------
        stack_pre_proc_output_products: tuple[StackPreProcessorOutputProducts, ...]
            The pre-processor output products associated to the input stack.

        breakpoint_dir: Path
            Path to the breakpoint directory.

        num_worker_threads: int = 1
            Number of threads assigned to the pre-processor.

        Raises
        ------
        StackPreProcessorRuntimeError, InvalidRegionOfInterestError, InvalidL1ProductLUTError

        Return
        ------
        dict
            Dictionary containing the pre-processor output.

        """
        # Setup the printing options.
        np.set_printoptions(formatter={"float": "{:0.3f}".format})

        # Validate the user provided inputs.
        _raise_if_job_order_input_is_invalid(self.job_order.input_stack)

        if self.aux_pps.general.allow_duplicate_images_flag:
            _warn_if_l1a_stack_has_duplicates(self.job_order.input_stack)
        else:
            _raise_if_l1a_stack_has_duplicates(self.job_order.input_stack)

        # The stack creation timestamp.
        stack_creation_stamp = PreciseDateTime.now()

        # The length of the input stacks.
        num_images = len(self.job_order.input_stack)

        # The bitsets.
        input_quality_bitset = np.zeros((num_images, 1), dtype=np.uint8)
        preproc_quality_bitset = np.zeros((num_images, 4), dtype=np.uint8)

        # Setup the readers.
        l1a_product_readers = tuple(
            BIOMASSL1ProductReader(l1a_product_path, nodata_fill_value=0.0)
            for l1a_product_path in self.job_order.input_stack
        )

        skip_coreg_input_setup = False
        if (breakpoint_dir / STOP_AND_RESUME_PATH).is_file():
            bps_logger.info("Coregistration inputs already available.")
            skip_coreg_input_setup = True

        # Read the data.
        bps_logger.info(
            "Reading the L1a annotations%s",
            "" if skip_coreg_input_setup else " and measurements",
        )
        l1a_product_data = _read_l1a_products(
            l1a_product_readers=l1a_product_readers,
            annotation_only=skip_coreg_input_setup,
            num_worker_threads=num_worker_threads,
        )
        _raise_if_l1a_stack_is_invalid(l1a_product_data, skip_check_data=skip_coreg_input_setup)

        # Update the quality bitset.
        (
            l1a_nonnominal_pol_product_indices,
            l1a_nonnominal_swath_product_indices,
            l1a_inconsistent_iono_indices,
            l1a_lowquality_product_indices,
        ) = _warn_if_l1a_stack_is_nonnominal(l1a_product_data)

        input_quality_bitset[l1a_lowquality_product_indices, 0] = 1

        preproc_quality_bitset[l1a_nonnominal_pol_product_indices, 1] = 1
        preproc_quality_bitset[l1a_nonnominal_swath_product_indices, 2] = 1
        preproc_quality_bitset[l1a_inconsistent_iono_indices, 3] = 1

        # Possibly disable rfi_degradation_index if not enabled in L1_P.
        bps_logger.info("Reading the L1a input stack LUTs")
        l1a_product_luts = _read_l1a_product_luts(
            l1a_product_readers=l1a_product_readers,
        )
        bps_logger.info("Reading the L1a input stack focalization parameters")
        l1a_product_defoc_params = _read_l1a_product_focalization_params(
            l1a_product_readers=l1a_product_readers,
        )
        bps_logger.info("Reading the L1a input stack ionosphere corrections")
        l1a_product_iono_corrections = _read_l1a_product_iono_corrections(
            l1a_product_readers=l1a_product_readers,
        )

        # Compute usable polarizations and prepare the stack data.
        usable_polarizations = common_polarizations(
            tuple(tuple(map(EPolarization, d.polarization_list)) for d in l1a_product_data),
        )
        if len(usable_polarizations) == 0:
            raise StackPreProcessorRuntimeError(
                "No common polarization available in the stack {:s}".format(
                    {p.name: p.polarization_list for p in l1a_product_data}
                )
            )

        bps_logger.info(
            "Usable polarizations: %s",
            [p.value for p in usable_polarizations],
        )

        # Validate the LUTs.
        for lut_data, l1a_product in zip(l1a_product_luts, l1a_product_data):
            _raise_if_invalid_luts(
                aux_pps=self.aux_pps,
                usable_polarizations=usable_polarizations,
                lut_data=lut_data[0],
                l1a_product=l1a_product,
            )

        bps_logger.info("Preparing L1a stack data")
        for l1a_data in l1a_product_data:
            prepare_stack_data(
                common_polarizations=usable_polarizations,
                data_list=l1a_data.data_list,
                polarization_list=l1a_data.polarization_list,
                metadata_list=[
                    l1a_data.raster_info_list,
                    l1a_data.burst_info_list,
                    l1a_data.swath_info_list,
                    l1a_data.sampling_constants_list,
                    l1a_data.acquisition_timeline_list,
                    l1a_data.data_statistics_list,
                    l1a_data.dc_vector_list,
                    l1a_data.dr_vector_list,
                    l1a_data.slant_to_ground_list,
                    l1a_data.ground_to_slant_list,
                    l1a_data.pulse_list,
                ],
            )

        # Execute the X-pol merging.
        if self.aux_pps.general.polarization_combination_method is None:
            bps_logger.warning("No cross-pol merging. The output L1c products will not be usable by L2")
        else:
            bps_logger.info(
                "Cross-pol merging: %s",
                self.aux_pps.general.polarization_combination_method.value,
            )
        for l1a_data, lut_data in zip(l1a_product_data, l1a_product_luts):
            l1a_data.channels = cross_pol_merging(
                data_list=l1a_data.data_list,
                swath_info_list=l1a_data.swath_info_list,
                polarization_list=l1a_data.polarization_list,
                metadata_list=[
                    l1a_data.raster_info_list,
                    l1a_data.burst_info_list,
                    l1a_data.sampling_constants_list,
                    l1a_data.acquisition_timeline_list,
                    l1a_data.data_statistics_list,
                    l1a_data.dc_vector_list,
                    l1a_data.dr_vector_list,
                    l1a_data.slant_to_ground_list,
                    l1a_data.ground_to_slant_list,
                    l1a_data.pulse_list,
                ],
                lut_data=lut_data[0],
                lut_axes=(lut_data[1], lut_data[2]),
                xpol_merging_method=self.aux_pps.general.polarization_combination_method,
            )
        if any(d.polarization_list != l1a_product_data[0].polarization_list for d in l1a_product_data):
            raise StackPreProcessorRuntimeError("Stack not packed correctly: Polarization misaligned")
        stack_polarizations = tuple(EPolarization(p) for p in l1a_product_data[0].polarization_list)
        bps_logger.info(
            "Stack polarizations: %s",
            [p.value for p in stack_polarizations],
        )
        if self.aux_pps.coregistration.polarization_used not in stack_polarizations:
            raise StackPreProcessorRuntimeError(
                "Selected coreg reference polarization {:s} no longer available".format(
                    self.aux_pps.coregistration.polarization_used.value
                )
            )

        # Compute the RFI and Faraday decorrelation indices. Setting all RFI
        # indices to 1.0 is equivalent to ignoring them.
        rfi_indices = tuple({p: 1.0 for p in stack_polarizations} for _ in l1a_product_luts)

        products_without_rfis = []
        if self.aux_pps.rfi_degradation_estimation.rfi_degradation_estimation_flag:
            bps_logger.info("Possibly compute RFI degradation indices")
            products_without_rfis = [
                l1a_product_index
                for l1a_product_index, l1a_product_lut in enumerate(l1a_product_luts)
                if not _has_rfi_masks(l1a_product_lut[0])
            ]
            if len(products_without_rfis):
                bps_logger.warning(
                    "Missing RFI masks from input product(s) %s",
                    [self.job_order.input_stack[i].name for i in products_without_rfis],
                )

            rfi_indices = tuple(
                compute_rfi_degradation_indices(lut_data[0], stack_polarizations) for lut_data in l1a_product_luts
            )
            for input_image, rfi in zip(self.job_order.input_stack, rfi_indices):
                if input_image.name not in products_without_rfis:
                    bps_logger.info(
                        "RFI indices %s: %s",
                        input_image.name,
                        {p.value: round(float(f), 3) for p, f in rfi.items()},
                    )

        bps_logger.info("Possibly compute Faraday Rotation indices")
        products_without_frs = [
            l1a_product_index
            for l1a_product_index, l1a_product_lut in enumerate(l1a_product_luts)
            if not _has_faraday_rotation(l1a_product_lut[0])
        ]
        if len(products_without_frs) > 0:
            bps_logger.warning(
                "Missing Faraday rotation mask from input product(s) %s",
                [self.job_order.input_stack[i].name for i in products_without_frs],
            )
        faraday_indices = tuple(compute_faraday_decorrelation_index(lut_data[0]) for lut_data in l1a_product_luts)
        for input_image, fri in zip(self.job_order.input_stack, faraday_indices):
            if input_image.name not in products_without_frs:
                bps_logger.info(
                    "Faraday Rotation Indices %s: %.3f",
                    input_image.name,
                    fri,
                )

        # Compute the stack spatial baselines wrt the median time.
        _, stack_spatial_median_ordering = compute_spatial_baselines(
            stack_state_vectors=tuple(self.__get_svs(d) for d in l1a_product_data),
            reference_state_vectors=self.__get_svs(l1a_product_data[0]),
            reference_azimuth_time=self.__get_azm_time(l1a_product_data[0]),
            reference_range_time=self.__get_rng_time(l1a_product_data[0]),
            satellite_side_looking=self.__get_side_looking(l1a_product_data[0]),
        )
        bps_logger.info("Stack spatial ordering: %s", stack_spatial_median_ordering)

        # Compute the stack temporal baselines wrt to the median time.
        _, stack_temporal_median_ordering = compute_temporal_baselines(
            stack_start_times=tuple(self.__get_start_time(d) for d in l1a_product_data),
            reference_start_time=self.__get_start_time(l1a_product_data[0]),
        )
        bps_logger.info("Stack temporal ordering: %s", stack_temporal_median_ordering)

        # Compute the primary image.
        coreg_primary_image_index, coreg_primary_selection_info = compute_coreg_primary_image_index(
            job_order_input_stack=self.job_order.input_stack,
            job_order_primary_image=self.job_order.processing_parameters.primary_image,
            config=self.aux_pps.primary_image_selection,
            stack_spatial_ordering=stack_spatial_median_ordering,
            stack_temporal_ordering=stack_temporal_median_ordering,
            reference_polarization=self.aux_pps.coregistration.polarization_used,
            rfi_coherence_degradation_indices=rfi_indices,
            faraday_decorrelation_indices=faraday_indices,
        )

        # Extract the processing ROI.
        stack_roi = None
        if self.job_order.processor_configuration.azimuth_interval is not None:
            azimuth_interval = self.job_order.processor_configuration.azimuth_interval
            primary_raster_info = l1a_product_data[coreg_primary_image_index].raster_info_list[0]

            stack_roi = compute_processing_roi(
                raster_info=primary_raster_info,
                toi=TimeOfInterest(
                    time_begin=azimuth_interval[0],
                    time_end=azimuth_interval[1],
                ),
            )

            # If STA_P needs to process a subregion of the data. Check that all
            # is consistent and report to the user.
            if stack_roi is not None:
                raise_if_roi_is_invalid(primary_raster_info, stack_roi)
                bps_logger.info(
                    "Processing stack in TOI %s (azimuth ROI: %s)",
                    azimuth_interval,
                    (stack_roi[0], stack_roi[0] + stack_roi[2]),
                )

        # Update the quality index.
        if coreg_primary_selection_info is common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS:
            preproc_quality_bitset[products_without_rfis, 0] = 1
            preproc_quality_bitset[products_without_frs, 0] = 1

        if coreg_primary_selection_info is common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION:
            preproc_quality_bitset[products_without_rfis, 0] = 1

        if coreg_primary_selection_info is common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION:
            preproc_quality_bitset[products_without_frs, 0] = 1

        # Compute the stack spatial and temporal baselines and orderings wrt
        # the selected coregistration primary image. This may be the same as
        # the ordering from median if the primary image selection method is
        # 'Geometry' or 'Temporal baseline'.
        coreg_primary_l1a_product_data = l1a_product_data[coreg_primary_image_index]

        # Just reporting the user.
        bps_logger.info(
            "Stack's critical baseline [m]: %.3f",
            _compute_critical_baseline(
                coreg_primary_l1a_product_data,
                l1a_product_luts[coreg_primary_image_index][0],
            ),
        )

        stack_spatial_baselines, _ = compute_spatial_baselines(
            stack_state_vectors=tuple(self.__get_svs(d) for d in l1a_product_data),
            reference_state_vectors=self.__get_svs(coreg_primary_l1a_product_data),
            reference_azimuth_time=self.__get_azm_time(coreg_primary_l1a_product_data),
            reference_range_time=self.__get_rng_time(coreg_primary_l1a_product_data),
            satellite_side_looking=self.__get_side_looking(coreg_primary_l1a_product_data),
        )
        bps_logger.info("Stack spatial baselines [m]: %s:", np.array(stack_spatial_baselines))

        stack_temporal_baselines, _ = compute_temporal_baselines(
            stack_start_times=tuple(self.__get_start_time(d) for d in l1a_product_data),
            reference_start_time=self.__get_start_time(coreg_primary_l1a_product_data),
        )
        bps_logger.info(
            "Stack temporal baselines [D, h:mm:ss]: %s",
            [str(timedelta(seconds=secs)) for secs in stack_temporal_baselines],
        )

        # Export the DEM and intermediate products, if not already available.
        if not skip_coreg_input_setup:
            bps_logger.info("Upsampling DEM for coreg primary product")
            dem_product = compute_ecef_dem(
                lut_data=l1a_product_luts[coreg_primary_image_index][0],
                lut_axes=l1a_product_luts[coreg_primary_image_index][1:],
                l1a_data_axes=(
                    get_time_axis(
                        l1a_product_data[coreg_primary_image_index].raster_info_list[0],
                        axis=0,
                        absolute=True,
                    )[0],
                    get_time_axis(
                        l1a_product_data[coreg_primary_image_index].raster_info_list[0],
                        axis=1,
                        absolute=True,
                    )[0],
                ),
            )

            # Export intermediate products for the coreg primary. We have to
            # export the DEM as well.
            bps_logger.info("Exporting intermediate products of the coreg primary frame")
            _export_intermediate_product(
                stack_pre_proc_output_products[coreg_primary_image_index],
                l1a_product_data=l1a_product_data[coreg_primary_image_index],
                dem_product_data=dem_product,
            )

            # Exporting the intermediate products (no DEM) for the secondary
            # stack input products.
            bps_logger.info("Exporting intermediate products for the secondary frames")
            for image_index, (l1a_product, stack_pre_proc_output_product) in enumerate(
                zip(l1a_product_data, stack_pre_proc_output_products)
            ):
                if image_index != coreg_primary_image_index:
                    _export_intermediate_product(
                        stack_pre_proc_output_product,
                        l1a_product_data=l1a_product,
                        dem_product_data=None,
                    )

        # Compute the stack footprint.
        stack_footprint = compute_processing_footprint(
            primary_footprint=coreg_primary_l1a_product_data.footprint,
            primary_raster_info=coreg_primary_l1a_product_data.raster_info_list[0],
            stack_roi=stack_roi,
        )
        bps_logger.debug(
            "L1a primary footprint: %s, stack footprint: %s",
            coreg_primary_l1a_product_data.footprint,
            stack_footprint,
        )

        # Report that the execution of the pre-processor was successful.
        _write_stop_and_resume_file(breakpoint_dir / STOP_AND_RESUME_PATH)

        # Resert nunpy print options.
        np.set_printoptions()

        return {
            "l1a_product_data": tuple(l1a_product_data),
            "l1a_product_luts": tuple(lut[0] for lut in l1a_product_luts),
            "l1a_product_luts_azm_axis": tuple(lut[1] for lut in l1a_product_luts),
            "l1a_product_luts_rng_axis": tuple(lut[2] for lut in l1a_product_luts),
            "l1a_product_focwindow_params": l1a_product_defoc_params,
            "l1a_product_iono_corrections": l1a_product_iono_corrections,
            "coreg_primary_image_index": coreg_primary_image_index,
            "coreg_primary_selection_info": coreg_primary_selection_info,
            "faraday_rotations": faraday_indices,
            "rfi_indices": rfi_indices,
            "stack_creation_stamp": stack_creation_stamp,
            "stack_roi": stack_roi,
            "stack_footprint": stack_footprint,
            "stack_spatial_baselines": stack_spatial_baselines,
            "stack_spatial_ordering": baseline_ordering(stack_spatial_baselines),
            "stack_temporal_baselines": stack_temporal_baselines,
            "stack_temporal_ordering": baseline_ordering(stack_temporal_baselines),
            "stack_polarizations": stack_polarizations,
            "input_quality_bitset": input_quality_bitset,
            "preproc_quality_bitset": preproc_quality_bitset,
        }

    def __get_svs(self, l1a_product_data: BIOMASSL1Product) -> StateVectors:
        """Retrieve the state vector for coreg reference polarization."""
        return l1a_product_data.general_sar_orbit[0]

    def __get_side_looking(self, l1a_product_data: BIOMASSL1Product) -> ESideLooking:
        """Retrieve the side looking of a L1 product."""
        return l1a_product_data.dataset_info[0].side_looking

    def __get_azm_time(self, l1a_product_data: BIOMASSL1Product) -> PreciseDateTime:
        """Retrieve the mid azimuth time for selected polarization."""
        return get_time_axis(
            l1a_product_data.raster_info_list[
                l1a_product_data.polarization_list.index(self.aux_pps.coregistration.polarization_used.value)
            ],
            axis=0,
            absolute=True,
        )[1]

    def __get_rng_time(self, l1a_product_data: BIOMASSL1Product) -> float:
        """Retrieve the mid range time for selected polarization."""
        return get_time_axis(
            l1a_product_data.raster_info_list[
                l1a_product_data.polarization_list.index(self.aux_pps.coregistration.polarization_used.value)
            ],
            axis=1,
            absolute=True,
        )[1]

    def __get_start_time(self, l1a_product_data: BIOMASSL1Product) -> PreciseDateTime:
        """Retrieve the start time for selected polarization."""
        return l1a_product_data.raster_info_list[
            l1a_product_data.polarization_list.index(self.aux_pps.coregistration.polarization_used.value)
        ].lines_start


def _raise_if_job_order_input_is_invalid(stack_input_paths: list[Path]):
    """Raise an error if the input paths are invalid."""
    # The stack must have at least 2 inputs.
    if len(stack_input_paths) < 2:
        raise StackPreProcessorRuntimeError("Stack must contain at least 2 images")

    # Check if there are nonexistsing products.
    nonexisting_l1a_products = [l1a.name for l1a in stack_input_paths if not l1a.exists()]
    if len(nonexisting_l1a_products) > 0:
        raise StackPreProcessorRuntimeError(f"Found nonexisting input products: {nonexisting_l1a_products}")


def _raise_if_l1a_stack_has_duplicates(stack_input_paths: list[Path]):
    """Raise if l1a stack has duplicates."""
    counter = Counter([_strip_creation_stamp(p.name) for p in stack_input_paths])
    duplicates = [p.name for p in stack_input_paths if counter[_strip_creation_stamp(p.name)] > 1]
    if len(duplicates) > 0:
        raise StackPreProcessorRuntimeError(f"Job order contains duplicated products: {duplicates}")


def _warn_if_l1a_stack_has_duplicates(stack_input_paths: list[Path]):
    """Warn if l1a stack has duplicates."""
    counter = Counter([_strip_creation_stamp(p.name) for p in stack_input_paths])
    duplicates = [p.name for p in stack_input_paths if counter[_strip_creation_stamp(p.name)] > 1]
    if len(duplicates) > 0:
        bps_logger.warning("Job order contains duplicated products: %s", duplicates)


def _raise_if_l1a_stack_is_invalid(
    l1a_product_data: tuple[BIOMASSL1Product, ...],
    skip_check_data: bool,
):
    """Raise an error if L1a input stack cannot be processed."""
    # Check if there are monitoring products.
    l1a_m_products = [l1a.name for l1a in l1a_product_data if l1a.is_monitoring]
    if len(l1a_m_products) > 0:
        raise StackPreProcessorRuntimeError(f"Found monitoring (M) products: {l1a_m_products}")

    # Check for products that has no data.
    l1a_empty_products = [l1a.name for l1a in l1a_product_data if len(l1a.data_list) == 0]
    if len(l1a_empty_products) > 0 and not skip_check_data:
        raise StackPreProcessorRuntimeError(f"Found product(s) with no data: {l1a_empty_products}")

    # Check for potential frames that contain the wrong number of raster info.
    l1a_wrong_num_raster_products = [
        l1a.name for l1a in l1a_product_data if len(l1a.raster_info_list) != len(l1a.data_list)
    ]
    if len(l1a_wrong_num_raster_products) and not skip_check_data:
        raise StackPreProcessorRuntimeError(
            f"Found product(s) with wrong number of rasters: {l1a_wrong_num_raster_products}"
        )

    # Check frames that has empty images.
    l1a_nodata_products = {l1a.name for l1a in l1a_product_data if any(d.size == 0 for d in l1a.data_list)}
    if len(l1a_nodata_products) > 0 and not skip_check_data:
        raise StackPreProcessorRuntimeError(f"Found product(s) with empty data: {l1a_nodata_products}")

    # Check for potential mismatching in data shape and raster shape.
    l1a_wrong_raster_shape_products = [
        l1a.name
        for l1a in l1a_product_data
        if any(d.shape[0] != r.lines for d, r in zip(l1a.data_list, l1a.raster_info_list))
        or any(d.shape[1] != r.samples for d, r in zip(l1a.data_list, l1a.raster_info_list))
    ]
    if len(l1a_wrong_raster_shape_products) > 0 and not skip_check_data:
        raise StackPreProcessorRuntimeError(
            f"Found product(s) with wrong raster shape: {l1a_wrong_raster_shape_products}"
        )

    # Check for potential internal inconsistencies.
    l1a_bad_time_annot_products = [l1a.name for l1a in l1a_product_data if _inconsistent_annotation(l1a)]
    if len(l1a_bad_time_annot_products) > 0 and not skip_check_data:
        raise StackPreProcessorRuntimeError(
            f"Found product(s) with with bad annotations: {l1a_bad_time_annot_products}"
        )


def _warn_if_l1a_stack_is_nonnominal(
    l1a_product_data: tuple[BIOMASSL1Product, ...],
) -> tuple[tuple[int, ...], ...]:
    """Check if and warn the user if the input stack is nonnominal."""
    # Warn the user there's an inconsistency between the the mission type and
    # the number of input frames. Note that this is only an heads-up for the
    # user, this won't affect the quality index.
    # NOTE: black formats this very oddly, we just skip formatting here.
    # fmt: off
    if not all(
        d.mission_phase_id == l1a_product_data[0].mission_phase_id
        for d in l1a_product_data
    ):
        bps_logger.warning("Found L1a products with inconsistent phase id")
    elif (
        l1a_product_data[0].mission_phase_id == MissionPhaseID.TOMOGRAPHIC.name
        and len(l1a_product_data) not in {7, 8}
    ):
        bps_logger.warning(
            "Got %d L1a products but 7 or 8 are expected for a TOMOGRAPHIC mission",
            len(l1a_product_data),
        )
    elif (
        l1a_product_data[0].mission_phase_id == MissionPhaseID.INTERFEROMETRIC.name
        and len(l1a_product_data) != 3
    ):
        bps_logger.warning(
            "Got %d L1a products but 3 are expected for an INTERFEROMETRIC",
            len(l1a_product_data),
        )
    # fmt: on

    # Check whether products have a non-nominal set of polarizations.
    l1a_nonnominal_pol_products = {
        i: len(p.polarization_list) for i, p in enumerate(l1a_product_data) if len(p.polarization_list) != 4
    }
    if len(l1a_nonnominal_pol_products):
        bps_logger.warning(
            "Found L1a products with non-nominal polarizations: %s",
            {l1a_product_data[i].name: n for i, n in l1a_nonnominal_pol_products.items()},
        )

    # Check swath duration of each frame and log if nonnominal.
    l1a_nonnominal_swath_products = {
        i: round(p.stop_time - p.start_time, 2)
        for i, p in enumerate(l1a_product_data)
        if not SWATH_DURATION_MIN < p.stop_time - p.start_time < SWATH_DURATION_MAX
    }
    if len(l1a_nonnominal_swath_products) > 0:
        bps_logger.warning(
            "Found L1a products with non-nominal swath duration [s]: %s",
            {l1a_product_data[i].name: t for i, t in l1a_nonnominal_swath_products.items()},
        )

    # Check the quality index for bad frames.
    l1a_quality_indices = {
        i: L1QualityIndex.decode(p.overall_product_quality_index) for i, p in enumerate(l1a_product_data)
    }

    l1a_lowquality_products = tuple(i for i, q in l1a_quality_indices.items() if any(asdict(q).values()))
    if len(l1a_lowquality_products) > 0:
        bps_logger.warning(
            "Found L1a products with nonzero quality index: %s",
            [l1a_product_data[i].name for i in l1a_lowquality_products],
        )
    l1a_inconsistent_iono_products = tuple(
        i for i, q in l1a_quality_indices.items() if q.inconsistent_phasescreen_and_rangeshifts_luts
    )
    if len(l1a_inconsistent_iono_products) > 0:
        bps_logger.warning(
            "Found L1a products with inconsistent phase-screen and range-shift LUTs: %s",
            [l1a_product_data[i].name for i in l1a_inconsistent_iono_products],
        )

    return (
        tuple(l1a_nonnominal_pol_products),
        tuple(l1a_nonnominal_swath_products),
        l1a_inconsistent_iono_products,
        l1a_lowquality_products,
    )


def _export_intermediate_product(
    stack_pre_proc_output_products: StackPreProcessorOutputProducts,
    *,
    l1a_product_data: BIOMASSL1Product | None,
    dem_product_data: tuple[npt.NDArray[float], ...] | None = None,
):
    """Export the intermediate raw data product and optionally the DEM."""
    stack_pre_proc_output_products.rmtree(ignore_errors=True)
    stack_pre_proc_output_products.mkdir(exist_ok=True, parents=True)

    # Export the raw data for coregistration.
    bps_logger.info(
        "Writing raw data product to %s",
        stack_pre_proc_output_products.raw_data_product.name,
    )

    # Export in product folder
    try:
        l1a_product_data.write(
            product_path=stack_pre_proc_output_products.raw_data_product,
            use_eff_dc_vectors=True,
        )
        # pylint: disable-next=broad-exception-caught
    except Exception as err:
        bps_logger.error(
            "Cannot write export coreg inputs to %s",
            stack_pre_proc_output_products.raw_data_product,
        )
        raise StackPreProcessorRuntimeError(err) from err

    # NOTE: We release the data and its memory since this is no longer
    # needed in the BPS.
    l1a_product_data.data_list = []

    # Optionally, export the DEM for coregistration.
    if dem_product_data is not None:
        bps_logger.info(
            "Writing DEM product to %s",
            stack_pre_proc_output_products.xyz_product.name,
        )
        try:
            write_pre_processor_dem_product(
                dem_product_xyz=dem_product_data,
                num_data_channels=l1a_product_data.channels,
                raster_info_list=l1a_product_data.raster_info_list,
                output_path=stack_pre_proc_output_products.xyz_product,
            )
            # pylint: disable-next=broad-exception-caught
        except Exception as err:
            bps_logger.error(
                "Cannot write DEM product to %s",
                stack_pre_proc_output_products.xyz_product,
            )
            raise StackPreProcessorRuntimeError(err) from err


def _read_l1a_products(
    l1a_product_readers: tuple[BIOMASSL1ProductReader, ...],
    *,
    annotation_only: bool,
    num_worker_threads: int,
) -> tuple[BIOMASSL1Product, ...]:
    """Read the L1a measurement data."""
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        return tuple(
            executor.map(
                lambda reader: _read_l1a_product_core(reader, annotation_only),
                l1a_product_readers,
            )
        )


def _read_l1a_product_luts(
    l1a_product_readers: tuple[BIOMASSL1ProductReader, ...],
) -> tuple[
    tuple[
        dict[str, npt.NDArray[float]],
        dict[str, npt.NDArray[PreciseDateTime]],
        dict[str, npt.NDArray[float]],
    ],
    ...,
]:
    """Read the L1a product LUTs."""
    l1a_lut_data = []
    for l1a_product_reader in l1a_product_readers:
        lut_data, rng_axis, azm_axis = l1a_product_reader.read_lut_annotation()
        # Drop the LUTs that are neither used by the Stack nor written in the
        # L1c products.
        for lut_item in (lut_data, rng_axis, azm_axis):
            lut_item.pop("tec", None)
            lut_item.pop("azimuthShifts", None)
            lut_item.pop("autofocusPhaseScreen", None)
        # We typically use azm first, and then rng.
        l1a_lut_data.append((lut_data, azm_axis, rng_axis))

    return tuple(l1a_lut_data)


def _read_l1a_product_focalization_params(
    l1a_product_readers: tuple[BIOMASSL1ProductReader, ...],
) -> tuple[dict[str, float], ...]:
    """Read the focalization parameters."""
    return tuple(l1a_product_reader.read_processing_parameters() for l1a_product_reader in l1a_product_readers)


def _read_l1a_product_iono_corrections(
    l1a_product_readers: tuple[BIOMASSL1ProductReader, ...],
) -> tuple[common_annotation_l1.IonosphereCorrection, ...]:
    """Read the L1a ionosphere correction parameters."""
    return tuple(l1a_product_reader.read_ionosphere_correction() for l1a_product_reader in l1a_product_readers)


def _compute_required_luts(
    aux_pps: AuxiliaryStaprocessingParameters,
    usable_polarizations: tuple[EPolarization, ...],
    l1a_product: BIOMASSL1Product,
) -> tuple[str, ...]:
    """Compute the required LUTs depending on AUX-PPS and available polarizations."""
    # The required polarizations, given the current polarization combination
    # method. This prevents interrupting the stack if, say, denoisingHV is
    # missing but the polarization combination method is V/H (so we will
    # dismiss the polarization H/V).
    xpol_method = aux_pps.general.polarization_combination_method
    if xpol_method is None or xpol_method is common.PolarisationCombinationMethodType.AVERAGE:
        required_nominal_polarizations = (EPolarization(p) for p in ["H/H", "H/V", "V/H", "V/V"])
    if xpol_method is common.PolarisationCombinationMethodType.HV:
        required_nominal_polarizations = set(EPolarization(p) for p in ["H/H", "H/V", "V/V"])
    if xpol_method is common.PolarisationCombinationMethodType.VH:
        required_nominal_polarizations = set(EPolarization(p) for p in ["H/H", "V/H", "V/V"])

    required_polarizations = set(usable_polarizations).intersection(required_nominal_polarizations)

    # Geometrical and denoising LUTs are always needed.
    required_luts = set(GEOMETRY_LUT_NAMES)
    for pol in required_polarizations:
        required_luts.add("denoising{}".format(pol.value.replace("/", "")))

    # If we need to perform RFI estiamtion, LUTs are needed for all the usable
    # polarizations. No check on RFI and FR LUTs for the primary image
    # selection are performed to be robust to missing info.
    if (
        aux_pps.rfi_degradation_estimation.rfi_degradation_estimation_flag
        and l1a_product.processing_parameters.rfi_detection_flag
    ):
        for pol in required_polarizations:
            # Only for frequency-based RFI mitigation we use the rfiFreqMask.
            # For all other methods, we use the time-based mask.
            if "time" in l1a_product.processing_parameters.rfi_mitigation_method.value.lower():
                required_luts.add("rfiTimeMask{}".format(pol.value.replace("/", "")))
            else:
                required_luts.add("rfiFreqMask{}".format(pol.value.replace("/", "")))

    return tuple(required_luts)


def _raise_if_invalid_luts(
    *,
    aux_pps: AuxiliaryStaprocessingParameters,
    usable_polarizations: tuple[EPolarization, ...],
    lut_data: dict,
    l1a_product: BIOMASSL1Product,
):
    """Check if all needed LUTs exist or raise an error otherwise."""
    # We don't always need all the LUTs.
    required_luts = _compute_required_luts(aux_pps, usable_polarizations, l1a_product)

    invalid_luts = {
        "missing": [lut for lut in required_luts if lut not in lut_data],
        "is_none": [lut for lut in required_luts if lut in lut_data and lut_data[lut] is None],
        "empty": [lut for lut in required_luts if lut in lut_data and lut_data[lut].size == 0],
    }
    if any(len(m) > 0 for m in invalid_luts.values()):
        raise InvalidL1ProductLUTError(l1a_product.name, invalid_luts)


def _read_l1a_product_core(
    l1a_product_reader: BIOMASSL1ProductReader,
    annotation_only: bool,
) -> BIOMASSL1Product:
    """Read L1a data and update with the correct product name."""
    l1a_product_name = Path(l1a_product_reader.product_path).name
    bps_logger.info("Reading L1a product %s", l1a_product_name)

    # Just set the product name as the actual product name.
    l1a_product_data = l1a_product_reader.read(annotation_only=annotation_only)
    l1a_product_data.name = l1a_product_name

    # Null out all the NaN and possibly Inf values.
    for data in l1a_product_data.data_list:
        data[_invalid_data(data)] = 0

    return l1a_product_data


def _compute_critical_baseline(
    l1a_product: BIOMASSL1Product,
    l1a_lut_data: dict,
) -> float:
    """Compute the critical baseline of a L1a data."""
    _, rng_time = get_time_axis(l1a_product.raster_info_list[0], axis=1)
    incidence_angles = np.deg2rad(l1a_lut_data["incidenceAngle"])
    return compute_critical_baseline(
        absolute_distance=sp.constants.speed_of_light * rng_time / 2,
        incidence_angle=sp.stats.circmean(incidence_angles),
        central_frequency=l1a_product.dataset_info[0].fc_hz,
        range_bandwidth=l1a_product.sampling_constants_list[0].brg_hz,
    )


def _has_rfi_masks(l1a_luts: dict[str, npt.NDArray[float]]) -> bool:
    """Check if LUTs contain RFIs."""
    rfi_regex = re.compile("rfi((Time)|(Freq))Mask")
    return any(re.match(rfi_regex, lut_name) is not None for lut_name in l1a_luts)


def _has_faraday_rotation(l1a_luts: dict[str, npt.NDArray[float]]) -> bool:
    """Check if LUTs contain the Faraday rotation."""
    return "faradayRotation" in l1a_luts


def _invalid_data(data: npt.NDArray[complex]) -> npt.NDArray[bool]:
    """Mask of possibly problematic data."""
    return np.isnan(data) | (np.abs(data) == np.inf)


def _inconsistent_annotation(l1a_product: BIOMASSL1Product) -> bool:
    """Check that the internal time annotations are consistent."""
    eps = np.power(10.0, -np.finfo(np.float64).precision)  # pylint: disable=no-member
    for raster in l1a_product.raster_info_list:
        if not (
            raster.lines != l1a_product.number_of_lines
            and raster.samples != l1a_product.number_of_samples
            and raster.lines_start != l1a_product.start_time
            and np.isclose(raster.lines_step, l1a_product.az_time_interval, atol=eps)
            and np.isclose(raster.samples_step, l1a_product.rg_time_interval, atol=eps)
        ):
            return False
    return True


def _write_stop_and_resume_file(output_path: Path):
    """Write the stop-and-resume file."""
    try:
        with output_path.open(mode="w", encoding="utf-8") as f:
            f.write("succeeded")
        if not output_path.exists():
            bps_logger.warning("Could not export stack's state file. Resuming STA_P will not be possible")
        # pylint: disable-next=broad-exception-caught
    except Exception as err:
        bps_logger.error(
            "Error occurred while writing stack's pre-proc state file %s to working directory",
            output_path,
        )
        raise StackPreProcessorRuntimeError(err) from err


def _strip_creation_stamp(l1a_product_name: str) -> str:
    """Remove the creation stamp suffix from the L1a product name."""
    return l1a_product_name[:-CREATION_STAMP_LENGTH]
