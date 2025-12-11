# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Coreg-Processor Execution Manager
---------------------------------------
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from shutil import copytree

import numpy as np
import numpy.typing as npt
from arepyextras.runner import Environment
from arepytools.io import open_product_folder
from arepytools.io.metadata import RasterInfo
from arepytools.io.productfolder2 import (
    ProductFolder2,
    is_product_folder,
    rename_product_folder,
)
from bps.common import bps_logger
from bps.common.configuration import fill_bps_configuration_file, write_bps_configuration_file
from bps.common.io import common
from bps.common.io.common_types.models import CoregistrationMethodType
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from bps.common.runner_helper import run_application
from bps.common.toi_utils import TimeOfInterest, toi_to_axis_slice
from bps.stack_cal_processor.core.utils import (
    compute_spatial_azimuth_shifts,
    compute_spatial_range_shifts,
    get_time_axis,
    read_productfolder_data,
    read_raster_info,
)
from bps.stack_coreg_processor.core.interpolation import interpolate_points_on_grid
from bps.stack_coreg_processor.core.shifting import coreg_primary_lut_axes, shift_lut
from bps.stack_coreg_processor.interface import (
    StackCoregProcInterfaceFiles,
    load_actualized_coregistration_parameters_raise_if_invalid,
    load_stop_and_resume_file,
    write_coreg_configuration_file,
    write_coreg_input_file,
    write_stop_and_resume_file,
)
from bps.stack_coreg_processor.utils import (
    StackCoregProcessorRuntimeError,
    write_product_folder,
)
from bps.stack_processor import __version__ as VERSION
from bps.stack_processor.execution.utils import setup_coreg_processor_env
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
    log_params,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.internal.intermediates import (
    ACTUALIZED_COREGISTRATION_PATH,
    CoregistrationOutputProducts,
    StackPreProcessorOutputProducts,
)
from bps.stack_processor.interface.internal.utils import (
    fill_stack_coreg_processor_config,
    fill_stack_coreg_processor_input_files,
)
from bps.transcoder.sarproduct.biomass_l1product import BIOMASSL1Product
from bps.transcoder.utils.constants import (
    AVERAGE_GROUND_VELOCITY as AVERAGE_GROUND_SPEED,
)

# The filename contains the configuration of the coregistration module.
BPS_CONF_FILENAME_XML = "BPSConf.xml"

# File that flags that the coregistration was successfully executed.
STOP_AND_RESUME_PATH = "COREGISTRATION_COMPLETE.json"


class StackCoregProcessorExecutionManager:
    """
    Manage the execution of the BPSStackProcessor.

    Parameters
    ----------
    job_order: StackJobOrder
        The STA_P job order.

    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS object.

    breakpoint_dir: Path
        Path to the breakpoint directory

    log_coreg_params: bool = True
        Optionally, log the parameters.

    """

    def __init__(
        self,
        *,
        job_order: StackJobOrder,
        aux_pps: AuxiliaryStaprocessingParameters,
        breakpoint_dir: Path,
        log_coreg_params: bool = True,
    ):
        """Initialize the object."""
        # Set the internal configurations.
        self.job_order = job_order
        self.aux_pps = aux_pps
        self.breakpoint_dir = breakpoint_dir

        # Environment objects to run the BPSStackProcessor subprocess.
        self.sta_p_env, self.sta_p_bin = setup_coreg_processor_env(self.breakpoint_dir)

        # Log the coregistration parameters.
        if log_coreg_params:
            bps_logger.info("Parameters:")
            log_params(asdict(aux_pps.coregistration), indent=1)

    def run_coregistration(
        self,
        *,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts],
        stack_pre_proc_exec_products: dict,
        export_distance_product_primary: bool = True,
        skip_check_products: list[str] | None = None,
        num_worker_threads: int = 1,
    ) -> dict:
        """
        Parameters
        ----------
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts]
            The STA_P preprocessor outputs.

        stack_coreg_proc_output_products: list[CoregistrationOutputProducts]
            The coregistration output products.

        stack_pre_proc_exec_products: dict
            The output of the execution of the pre-processor.

        export_distance_product_primary: bool = True
            Export the distance product (_DAPD) of the coregistration primary.

        skip_check_products: list[str] | None
            Optionally, a list of coregistration output products that are not
            of interest, and so they are not verified upon execution or resume.

        num_worker_threads: int = 1
            Number of threads assigned to the coregistration jobs.

        Raises
        ------
        StackCoregProcessorRuntimeError

        """
        # Check stop and resume file.
        needs_resume = True
        if (self.breakpoint_dir / STOP_AND_RESUME_PATH).is_file():
            succeeded, needs_resume = load_stop_and_resume_file(self.breakpoint_dir / STOP_AND_RESUME_PATH)
            if not needs_resume:
                bps_logger.info("Coregistration products already available. Skipping coregistration.")

        # Run coregistration if there was no stop and resume file or it was
        # made with shift_estimation_only.
        if needs_resume:
            self.__raise_if_invalid_coregistration_input(
                stack_pre_proc_output_products,
                stack_pre_proc_exec_products["coreg_primary_image_index"],
            )

            succeeded = self.__run_coregistration_multithreaded(
                breakpoint_dir=self.breakpoint_dir,
                stack_pre_proc_output_products=stack_pre_proc_output_products,
                stack_coreg_proc_output_products=stack_coreg_proc_output_products,
                l1a_products=stack_pre_proc_exec_products["l1a_product_data"],
                coreg_primary_image_index=stack_pre_proc_exec_products["coreg_primary_image_index"],
                export_distance_product_primary=export_distance_product_primary,
                num_worker_threads=num_worker_threads,
            )
            if not succeeded[stack_pre_proc_exec_products["coreg_primary_image_index"]]:
                raise StackCoregProcessorRuntimeError("Failed to coregister primary vs primary.")
            if sum(succeeded) < 2:
                raise StackCoregProcessorRuntimeError("All coregistrations failed.")
            if not all(succeeded):
                bps_logger.warning(
                    "Coregistration failed for the following products: %s",
                    [p.name for p, ok in zip(self.job_order.input_stack, succeeded) if not ok],
                )
            write_stop_and_resume_file(
                self.breakpoint_dir / STOP_AND_RESUME_PATH,
                succeeded_mask=succeeded,
                needs_resume=False,
            )

        _rearrange_stack(
            job_order=self.job_order,
            stack_pre_proc_output_products=stack_pre_proc_output_products,
            stack_coreg_proc_output_products=stack_coreg_proc_output_products,
            stack_pre_proc_exec_products=stack_pre_proc_exec_products,
            succeeded_mask=succeeded,
        )
        self.__raise_if_invalid_coregistration_outputs(
            stack_coreg_proc_output_products,
            stack_pre_proc_exec_products,
            check_distance_product_primary=export_distance_product_primary,
            check_coreg_product=(
                self.aux_pps.coregistration.coregistration_execution_policy
                is not common.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY
            ),
            skip_check_products=skip_check_products,
        )

        # The actual coregistration methods used.
        actualized_coreg_params = tuple(
            self.__read_actualized_coregistration_parameters(
                coreg_product,
                index == stack_pre_proc_exec_products["coreg_primary_image_index"],
            )
            for index, coreg_product in enumerate(stack_coreg_proc_output_products)
        )

        # The quality bitset. This will be bit #1.
        quality_bitset = np.array(
            [
                params["quality_coregistration"] < self.aux_pps.coregistration.fitting_quality_threshold
                for params in actualized_coreg_params
            ],
            dtype=np.uint8,
        )

        return {
            "succeeded_coregistrations": succeeded,
            "actualized_coregistration_parameters": actualized_coreg_params,
            "quality_bitset": quality_bitset,
        }

    def run_lut_shifting(
        self,
        *,
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts],
        stack_coreg_proc_exec_products: dict,
        stack_pre_proc_exec_products: dict,
    ):
        """
        Shift the LUTs.

        Parameters
        ----------
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts]
            The coregistration output products.

        stack_coreg_proc_output_products: dict
            The output of the execution of the coregistration processor.

        stack_pre_proc_exec_products: dict
            The output of the execution of the pre-processor.

        Raises
        ------
        StackCoregProcessorRuntimeError

        """
        # The coregistration primary.
        l1a_product_data = stack_pre_proc_exec_products["l1a_product_data"]
        coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]
        primary_raster_info = l1a_product_data[coreg_primary_image_index].raster_info_list[0]

        coreg_primary_products = stack_coreg_proc_output_products[coreg_primary_image_index]

        # The geometric LUT grid of the primary (we just pick 'latitude')
        l1a_primary_geo_lut_axes = (
            stack_pre_proc_exec_products["l1a_product_luts_azm_axis"][coreg_primary_image_index]["latitude"],
            stack_pre_proc_exec_products["l1a_product_luts_rng_axis"][coreg_primary_image_index]["latitude"],
        )
        if l1a_primary_geo_lut_axes[0].size <= 1 or l1a_primary_geo_lut_axes[1].size <= 1:
            raise StackCoregProcessorRuntimeError("Duped L1a coreg primary geometric axes")

        # The LUT of the primary, just for some logging later on.
        l1a_primary_luts = stack_pre_proc_exec_products["l1a_product_luts"][coreg_primary_image_index]

        # Compute the LUT primary grid, this is the output grid of the STA_P
        # LUTs. All LUTs will be stitched onto this grid. lut_azm_indices and
        # lut_rng_indices are wrt the full axes.
        *lut_primary_axes, lut_azm_indices, lut_rng_indices = coreg_primary_lut_axes(
            primary_raster_info,
            target_azimuth_step=float(np.diff(l1a_primary_geo_lut_axes[0][:2])),
            target_range_step=float(np.diff(l1a_primary_geo_lut_axes[1][:2])),
            roi=stack_pre_proc_exec_products["stack_roi"],
        )

        # Define the shifts that represent no-shifs. We use "0.0" to make sure
        # they are cast to float64.
        azimuth_no_shifts, range_no_shifts = np.mgrid[
            0.0 : primary_raster_info.lines, 0.0 : primary_raster_info.samples
        ]

        # The decimation factor to be applied to the LUTs.
        lut_azimuth_decimation_factor = np.diff(lut_azm_indices[:2])[0]
        lut_range_decimation_factor = np.diff(lut_rng_indices[:2])[0]

        bps_logger.info(
            "Exporting LUTs @ %.2f m x %.2f m resolution (azm decimation factor: %d, rng decimation factor: %d)",
            compute_spatial_azimuth_shifts(
                lut_azimuth_decimation_factor,
                primary_raster_info.lines_step,
                ground_speed=AVERAGE_GROUND_SPEED,
            ),
            compute_spatial_range_shifts(
                lut_range_decimation_factor,
                primary_raster_info.samples_step,
                incidence_angle=np.mean(np.deg2rad(l1a_primary_luts["incidenceAngle"])),
            ),
            lut_azimuth_decimation_factor,
            lut_range_decimation_factor,
        )

        # The possible bias in the coregistration due to small DEM errors when
        # upsampling.
        any_coregistration_from_geometry = any(
            p["coregistration_method"] is CoregistrationMethodType.GEOMETRY
            for p in stack_coreg_proc_exec_products["actualized_coregistration_parameters"]
        )

        azimuth_coreg_shifts_bias = 0
        range_coreg_shifts_bias = 0
        if self.aux_pps.general.flattening_phase_bias_compensation_flag and any_coregistration_from_geometry:
            azimuth_coreg_shifts_bias = (
                read_productfolder_data(
                    open_product_folder(coreg_primary_products.az_shifts_product),
                )
                - azimuth_no_shifts
            )
            range_coreg_shifts_bias = (
                read_productfolder_data(
                    open_product_folder(coreg_primary_products.rg_shifts_product),
                )
                - range_no_shifts
            )

        invalid_coreg_shifts_ratios = []
        invalid_coreg_shifts_masks = []

        # We shift the LUTs of all products. Note that also the primary product
        # requires being shifted since the LUT grid and the data grid may not
        # be aligned.
        for (
            index,
            (
                l1a_data,
                lut_data,
                lut_azm_axis,
                lut_rng_axis,
                coreg_product,
                coreg_params,
                l1a_path,
            ),
        ) in enumerate(
            zip(
                l1a_product_data,
                stack_pre_proc_exec_products["l1a_product_luts"],
                stack_pre_proc_exec_products["l1a_product_luts_azm_axis"],
                stack_pre_proc_exec_products["l1a_product_luts_rng_axis"],
                stack_coreg_proc_output_products,
                stack_coreg_proc_exec_products["actualized_coregistration_parameters"],
                self.job_order.input_stack,
            ),
        ):
            # NOTE: Some LUTs require special treatment and we shift them
            # individually. We report them here.
            done_luts = set()

            bps_logger.info(
                "Shifting and subsampling LUTs for secondary product %s",
                l1a_path.name,
            )

            # Apply the shifts to the upsampled LUTs.
            bps_logger.debug("Reading coregistration shifts")

            # The bias may just be 0, if not applicable (see above).
            azimuth_coreg_shifts = (
                read_productfolder_data(
                    open_product_folder(coreg_product.az_shifts_product),
                )
                - azimuth_coreg_shifts_bias
            )
            azimuth_geo_coreg_shifts = (
                read_productfolder_data(
                    open_product_folder(coreg_product.az_geo_shifts_product),
                )
                - azimuth_coreg_shifts_bias
            )
            range_coreg_shifts = (
                read_productfolder_data(
                    open_product_folder(coreg_product.rg_shifts_product),
                )
                - range_coreg_shifts_bias
            )
            range_geo_coreg_shifts = (
                read_productfolder_data(
                    open_product_folder(coreg_product.rg_geo_shifts_product),
                )
                - range_coreg_shifts_bias
            )

            # Check the invalid coregistration shifts.
            bps_logger.debug("Retrieving coregistrationShiftsQuality")
            (
                coreg_shifts_quality,
                invalid_coreg_shifts,
                invalid_coreg_shifts_ratio,
            ) = self.__compute_invalid_coreg_shifts(
                coreg_shifts_quality_product=coreg_product.quality_shifts_product,
                primary_raster_info=primary_raster_info,
                lut_azm_indices=lut_azm_indices,
                lut_rng_indices=lut_rng_indices,
                coregistration_method=coreg_params["coregistration_method"],
                is_coreg_primary=(index == coreg_primary_image_index),
                roi=stack_pre_proc_exec_products["stack_roi"],
            )
            invalid_coreg_shifts_ratios.append(invalid_coreg_shifts_ratio)
            invalid_coreg_shifts_masks.append(invalid_coreg_shifts)

            # The IOB may need phaseScreen and rangeShifts for L1 iono
            # compensation. We do not need these LUT in memory. We just dump
            # them. Note that we write this LUT with the full shape of the
            # original L1a data (no TOI). The calibration input manager will
            # take care of loading the portion of the data corresponding to the
            # TOI.
            slow_ionosphere_removal_enabled = (
                self.aux_pps.slow_ionosphere_removal.slow_ionosphere_removal_flag
                and self.aux_pps.slow_ionosphere_removal.compensate_l1_iono_phase_screen_flag
            )

            # We track the expected cached products.
            cached_lut_products = []

            if "phaseScreen" in lut_data:
                phase_screen_lut = lut_data.pop("phaseScreen")
                phase_screen_lut_azm_axis = lut_azm_axis.pop("phaseScreen")
                phase_screen_lut_rng_axis = lut_rng_axis.pop("phaseScreen")

                # Only if the IOB is enabled, we will upsample it and dump
                # it to disk to release some memory, otherwise we simply
                # dump it from memory.
                if slow_ionosphere_removal_enabled:
                    bps_logger.debug("Shifting and caching phaseScreen")
                    write_product_folder(
                        data=shift_lut(
                            lut_data=phase_screen_lut,
                            lut_axes=(
                                phase_screen_lut_azm_axis,
                                phase_screen_lut_rng_axis,
                            ),
                            azimuth_coreg_shifts=azimuth_coreg_shifts,
                            range_coreg_shifts=range_coreg_shifts,
                            primary_raster_info=primary_raster_info,
                            secondary_raster_info=l1a_data.raster_info_list[0],
                            lut_interp_fn=partial(
                                interpolate_points_on_grid,
                                fill_value=0.0,
                            ),
                        ),
                        output_pf_path=coreg_product.l1_iono_phase_screen_product,
                        data_name="phaseScreen",
                    )
                    cached_lut_products.append("l1_iono_phase_screen_product")

                # Report as done.
                done_luts.add("phaseScreen")

            if "rangeShifts" in lut_data:
                phase_screen_lut = lut_data.pop("rangeShifts")
                phase_screen_lut_azm_axis = lut_azm_axis.pop("rangeShifts")
                phase_screen_lut_rng_axis = lut_rng_axis.pop("rangeShifts")

                # Only if the IOB is enabled, we will upsample it and dump
                # it to disk to release some memory, otherwise we simply
                # dump it from memory.
                if slow_ionosphere_removal_enabled:
                    bps_logger.debug("Shifting and caching rangeShifts")
                    write_product_folder(
                        data=shift_lut(
                            lut_data=phase_screen_lut,  # [m].
                            lut_axes=(
                                phase_screen_lut_azm_axis,
                                phase_screen_lut_rng_axis,
                            ),
                            azimuth_coreg_shifts=azimuth_coreg_shifts,
                            range_coreg_shifts=range_coreg_shifts,
                            primary_raster_info=primary_raster_info,
                            secondary_raster_info=l1a_data.raster_info_list[0],
                            lut_interp_fn=partial(
                                interpolate_points_on_grid,
                                fill_value=0.0,
                            ),
                        ),
                        output_pf_path=coreg_product.l1_iono_range_shifts_product,
                        data_name="rangeShifts",
                    )
                    cached_lut_products.append("l1_iono_range_shifts_product")

                # Report as done.
                done_luts.add("rangeShifts")

            # Apply the coregistration shifts to the downsampled LUTs.
            bps_logger.debug("Subsampling coregistration shifts")
            lut_data["azimuthCoregistrationShifts"] = azimuth_coreg_shifts[lut_azm_indices, :][:, lut_rng_indices]
            lut_data["rangeCoregistrationShifts"] = range_coreg_shifts[lut_azm_indices, :][:, lut_rng_indices]

            bps_logger.debug("Subsampling coregistration shifts from orbit")
            # fmt: off
            lut_data["azimuthOrbitCoregistrationShifts"] = (
                azimuth_geo_coreg_shifts[lut_azm_indices, :][:, lut_rng_indices]
            )
            lut_data["rangeOrbitCoregistrationShifts"] = (
                range_geo_coreg_shifts[lut_azm_indices, :][:, lut_rng_indices]
            )
            # fmt: on

            lut_data["coregistrationShiftsQuality"] = coreg_shifts_quality

            for lut_name in list(lut_data.keys()):
                # If we had to upsample LUTs, we took care of those already. If
                # no, we will only shift them downsampled.
                if lut_name in done_luts:
                    continue

                # Both rfiTimeMask* and rfiFreqMask* are no longer used in STA_P.
                if lut_name.startswith("rfi"):
                    lut_data.pop(lut_name, None)
                    lut_azm_axis.pop(lut_name, None)
                    lut_rng_axis.pop(lut_name, None)
                    continue

                # Coregistration shifts are not to be shifted.
                if "coregistration" in lut_name.lower():
                    continue

                if lut_name not in lut_azm_axis or lut_name not in lut_rng_axis:
                    raise StackCoregProcessorRuntimeError(f"{lut_name} has no LUT axes")

                # Shift the LUT.
                bps_logger.debug("Shifting %s", lut_name)
                lut_data[lut_name] = shift_lut(
                    lut_data=lut_data[lut_name],
                    lut_axes=(lut_azm_axis[lut_name], lut_rng_axis[lut_name]),
                    azimuth_coreg_shifts=lut_data["azimuthCoregistrationShifts"],
                    range_coreg_shifts=lut_data["rangeCoregistrationShifts"],
                    primary_raster_info=primary_raster_info,
                    secondary_raster_info=l1a_data.raster_info_list[0],
                    lut_interp_fn=partial(
                        interpolate_points_on_grid,
                        fill_value=np.nan,
                    ),
                )

                self.__raise_if_missing_cached_lut_products(
                    stack_coreg_exec_products=coreg_product,
                    expected_lut_brk_products=cached_lut_products,
                )

        # From now on, all LUTs are defined on the grid of the primary, so no
        # more dictionaries etc. Same axes for all data.
        stack_pre_proc_exec_products.pop("l1a_product_luts_azm_axis", None)
        stack_pre_proc_exec_products.pop("l1a_product_luts_rng_axis", None)

        # The quality bitset for the coregistration. This will bit bit #2.
        quality_bitset = (np.array(invalid_coreg_shifts_ratios) > 0).astype(np.uint8)

        return {
            "lut_data": stack_pre_proc_exec_products.pop("l1a_product_luts"),
            "lut_primary_azm_axis": lut_primary_axes[0],
            "lut_primary_rng_axis": lut_primary_axes[1],
            "lut_primary_azm_indices": lut_azm_indices - lut_azm_indices[0],
            "lut_primary_rng_indices": lut_rng_indices - lut_rng_indices[0],
            "lut_azimuth_decimation_factor": lut_azimuth_decimation_factor,
            "lut_range_decimation_factor": lut_range_decimation_factor,
            "invalid_residual_shifts_ratios": tuple(invalid_coreg_shifts_ratios),
            "invalid_residual_shifts_masks": tuple(invalid_coreg_shifts_masks),
            "quality_bitset": quality_bitset,
        }

    def __raise_if_invalid_coregistration_input(
        self,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
        coreg_primary_image_index: int,
    ):
        """
        Check whether the coregistration inputs (from preprocessor) are valid
        product folders and raise if they are not.

        Parameters
        ----------
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts]
            The STA_P preprocessor outputs.

        coreg_primary_image_index: int
            The index of the coregistration primary image.

        Raises
        ------
        StackCoregProcessorRuntimeError

        """
        for image_index, pre_proc_products in enumerate(stack_pre_proc_output_products):
            # The DEM product is needed only for the primary image index.
            required_products = ["raw_data_product"]
            if image_index == coreg_primary_image_index:
                required_products.append("xyz_product")
            if not pre_proc_products.are_product_folders(products=required_products):
                raise StackCoregProcessorRuntimeError(
                    "Coregistration breakpoints in working directory are missing or broken"
                )

    def __raise_if_invalid_coregistration_outputs(
        self,
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts],
        stack_pre_proc_exec_products: dict,
        *,
        check_coreg_product: bool,
        check_distance_product_primary: bool,
        skip_check_products: list[str] | None,
    ):
        """
        Check whether the coregistration outputs are valid product folders and
        raise if they are not.

        Parameters
        ----------
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts]
            The STA_P coregistration output products.

        stack_pre_proc_exec_products: dict
            Execution products of the stack's pre-processor.

        check_coreg_product: bool
            As to whether the _Cor product should be checked or not (e.g. if
            execution policy is "Shift Estimation Only", the _Cor products are
            not expected).

        check_distance_product_primary: bool
            If true, check that the _DAPD product exists for the primary.

        Raises
        ------
        StackCoregProcessorRuntimeError

        """
        coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]
        coreg_primary_raster_info = stack_pre_proc_exec_products["l1a_product_data"][
            coreg_primary_image_index
        ].raster_info_list[0]

        coreg_product = [] if not check_coreg_product else ["coreg_product"]
        for image_index, prod in enumerate(stack_coreg_proc_output_products):
            required_products = [
                *coreg_product,
                "synth_product",
                "az_shifts_product",
                "rg_shifts_product",
                "kz_product",
            ]
            if skip_check_products is not None:
                required_products = list(set(required_products) - set(skip_check_products))

            # For the coregistration primary, we expect the vertical
            # wavenumbers (if the export was enabled).
            if image_index == coreg_primary_image_index and check_distance_product_primary:
                required_products.append("distance_product")

            # Check that all required products exist.
            if not prod.are_product_folders(products=required_products):
                raise StackCoregProcessorRuntimeError(
                    "Coregistration breakpoints in working directory are missing or broken"
                )

            # Check that all the required rasters have the expected shape.
            for required_prod in required_products:
                if not _raster_info_equal(
                    coreg_primary_raster_info,
                    read_raster_info(open_product_folder(prod.__dict__[required_prod])),
                ):
                    raise StackCoregProcessorRuntimeError(
                        f"Raster {required_prod} for frame={image_index} is incompatible with coreg primary"
                    )

    def __raise_if_missing_cached_lut_products(
        self,
        stack_coreg_exec_products: CoregistrationOutputProducts,
        expected_lut_brk_products: list[str],
    ):
        """
        Check whether the expected breakpoint files are present in the
        working directory.

        Parameters
        ----------
        stack_coreg_exec_product: CoregistrationOutputProducts
            The STA_P coregistration product (1 image).

        expected_lut_brk_products: list[str]
            The expected LUTs that are stored as breakpoint files.

        Raises
        ------
        StackCoregProcessorRuntimeError

        """
        if len(expected_lut_brk_products) == 0:
            return

        # Check that we have all expected products cached in the LUTs.
        if not stack_coreg_exec_products.are_product_folders(products=expected_lut_brk_products):
            raise StackCoregProcessorRuntimeError("LUT shifting breakpoints are missing from the working dir")

    def __read_actualized_coregistration_parameters(
        self,
        stack_coreg_output_product: CoregistrationOutputProducts,
        is_coreg_primary_image_index: bool,
    ) -> dict:
        """
        Read the actualized coregistration parameters.

        Parameters
        ----------
        stack_coreg_output_product: CoregistrationOutputProduct
            The filesystem structure of the coregistration output.

        is_coreg_primary_image_index: bool
            As to whether it is the the primary coregistration frame.

        Raises
        ------
        StackCoregProcessorRuntimeError

        Return
        ------
        dict
            The actualized coregistration parameters (i.e. the content of the
            actualizedCoregistrationParameters.xml file).

        """
        # The coregistration method selected by the user via AUX-PPS.
        aux_pps_coreg_method = self.aux_pps.coregistration.coregistration_method

        # Under 'Geometry'. There's no actualize coregistration parameters.
        if aux_pps_coreg_method is common.CoregistrationMethodType.GEOMETRY or is_coreg_primary_image_index:
            return {
                "coregistration_method": CoregistrationMethodType.GEOMETRY,
                "quality_coregistration_azimuth": 1.0,
                "quality_coregistration_range": 1.0,
                "quality_coregistration": 1.0,
                "valid_blocks_ratio": 1.0,
                "common_band_average_bandwidth_ratio": 1.0,
            }

        if not stack_coreg_output_product.actualized_coregistration_parameters_file.exists():
            raise StackCoregProcessorRuntimeError(f"File {ACTUALIZED_COREGISTRATION_PATH} is missing from coreg cache")
        return load_actualized_coregistration_parameters_raise_if_invalid(
            stack_coreg_output_product.actualized_coregistration_parameters_file
        )

    def __compute_invalid_coreg_shifts(
        self,
        *,
        coreg_shifts_quality_product: Path,
        primary_raster_info: RasterInfo,
        lut_azm_indices: npt.NDArray[int],
        lut_rng_indices: npt.NDArray[int],
        coregistration_method: common.CoregistrationMethodType,
        is_coreg_primary: bool,
        roi: RegionOfInterest | None = None,
    ) -> tuple[npt.NDArray[float], npt.NDArray[bool], float]:
        """
        Compute the invalid coreg shifts.

        Parameters
        ----------
        coreg_shifts_quality_product: Path
            The product folder path with the invalid coreg shifts.

        primary_raster_info: RasterInfo
            Raster info of the primary image.

        lut_azm_indices: npt.NDArray[int]
            The subsampling azimuth indices of the LUT grid wrt the full axis.

        lut_rng_indices: npt.NDArray[int]
            The subsampling range indices of the LUT grid wrt the full axis.

        coregistration_method: common.CoregistrationMethodType
            The coregistration method actually used for coregistring
            the frames (e.g. Geometry after set Automatic).

        is_coreg_primary: bool
            As to whether the current processed image is the the
            coregistration primary.

        roi: Optional[RegionOfInterest] = None
            Optionally, a ROI associated to the data.

        Raises
        ------
        StackCoregProcessorRuntimeError

        Return
        ------
        npt.NDArray[float] [samples]
            The coreg shifts qualities subsampled on the LUTs. A matrix full
            of NaN's if the coreg shifts qualities are not available.

        npt.NDArray[bool]
            A boolean mask that is True on the invalid shifts and False
            on the valid shifts.

        float [%]
            The percentage of invalid shifts (normalized between 0 and 1).

        """
        # Prepare the mask with the invalid coregitration shifts.
        invalid_coreg_shifts = np.empty((0,), dtype=np.bool_)

        # The coregistration shifts quality values.
        #
        # NOTE: We populate with NaN's since if no quality shifts are available
        # due to selected coregistration method (aka Geometry), we will report
        # as noDataValue.
        coreg_shifts_quality = np.full((lut_azm_indices.size, lut_rng_indices.size), np.nan)

        if coregistration_method is CoregistrationMethodType.GEOMETRY_AND_DATA:
            if not is_product_folder(coreg_shifts_quality_product):
                raise StackCoregProcessorRuntimeError(
                    f"Intermediate {coreg_shifts_quality_product} is not a product folder"
                )

            csa_pf = open_product_folder(coreg_shifts_quality_product)
            csa_raster_info = read_raster_info(csa_pf)

            if len(ProductFolder2.get_channels_list(csa_pf)) == 0:
                raise StackCoregProcessorRuntimeError(f"Intermediate {coreg_shifts_quality_product} is empty")

            # If we had selected a TOI in the original data, we need to map it
            # onto the corresponding ROI on the CSA product, since the CSA
            # product has his own raster grid.
            csa_roi = None
            if roi is not None:
                csa_roi = _get_csa_roi(
                    csa_raster_info,
                    data_raster_info=primary_raster_info,
                    data_roi=roi,
                )

            # Clip to 0-1 to avoid small errors at the borders due to tiny
            # errors in CSA's subsampled axes.
            coreg_shifts_quality = np.clip(
                interpolate_points_on_grid(
                    grid_values=read_productfolder_data(csa_pf, roi=csa_roi),
                    axes_in=_get_relative_time_axes(csa_raster_info, csa_roi),
                    query_points=_get_relative_time_axes(primary_raster_info, roi),
                    query_points_on_grid=True,
                ),
                *[0, 1],
            )

            # The mask that represents the invalid shifts.
            invalid_coreg_shifts = coreg_shifts_quality < self.aux_pps.coregistration.residual_shift_quality_threshold
            if not np.any(invalid_coreg_shifts):
                bps_logger.debug("No invalid coreg shifts detected")
            else:
                bps_logger.warning(
                    "Detected %.1f%s of invalid coreg shifts. They will be ignored.",
                    100 * np.sum(invalid_coreg_shifts) / max(invalid_coreg_shifts.size, 1),
                    "%",
                )

            # From now on we only care about subsampling on the LUTs.
            coreg_shifts_quality = coreg_shifts_quality[lut_azm_indices - lut_azm_indices[0], :][
                :, lut_rng_indices - lut_rng_indices[0]
            ]
        elif not is_coreg_primary:
            bps_logger.info("No coreg shifts quality available due to selected coreg method")

        return (
            coreg_shifts_quality,
            invalid_coreg_shifts,
            np.sum(invalid_coreg_shifts) / max(1, invalid_coreg_shifts.size),
        )

    def __run_coregistration_multithreaded(
        self,
        *,
        breakpoint_dir: Path,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts],
        l1a_products: tuple[BIOMASSL1Product, ...],
        coreg_primary_image_index: int,
        export_distance_product_primary: bool = True,
        num_worker_threads: int = 1,
    ) -> tuple[bool, ...]:
        """
        Coregister images in multiple threads.

        Parameters
        ----------
        breakpoint_dir: Path
            The working directory.

        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts]
            The STA_P preprocessor outputs.

        stack_coreg_proc_output_products: list[CoregistrationOutputProducts]
            The coregistration output products.

        l1a_products: tuple[BIOMASSL1Product, ...]
            The stack input L1a products.

        coreg_primary_image_index: int
            The image index of the coregistration primary.

        export_distance_product_primary: bool = True
            Export the distance product (_DAPD) of the coregistration primary.

        num_worker_threads: int = 1
            Number of threads assigned to the coregistration jobs.

        Raises
        ------
        StackCoregProcessorRuntimeError

        Return
        ------
        tuple[bool, ...]
            Flags that report which run was successful or not.

        """
        # We can only coregister data on same DEM.
        if any(d.height_model != l1a_products[0].height_model for d in l1a_products):
            raise StackCoregProcessorRuntimeError("Can only coregister data with same DEM/Height Model")

        # Write the BPS configuration file.
        write_bps_configuration_file(
            fill_bps_configuration_file(
                self.job_order.processor_configuration,
                task_name="STA_P",
                processor_name="STA_P",
                processor_version=bps_logger.get_version_in_logger_format(VERSION),
                node_name=bps_logger.get_default_logger_node(),
            ),
            breakpoint_dir / BPS_CONF_FILENAME_XML,
        )

        # Write processor and coregistration configuration files.
        stack_coreg_proc_interface_files = StackCoregProcInterfaceFiles.from_base_dir(breakpoint_dir)

        # Configuration file for the secondaries.
        write_coreg_configuration_file(
            fill_stack_coreg_processor_config(
                aux_pps=self.aux_pps,
                coregistration_method=self.aux_pps.coregistration.coregistration_method,
                export_distance_product=False,  # Needed only for the primary.
            ),
            stack_coreg_proc_interface_files.coreg_config_file,
        )
        # Configuration file for the primary.
        write_coreg_configuration_file(
            fill_stack_coreg_processor_config(
                aux_pps=self.aux_pps,
                coregistration_method=common.CoregistrationMethodType.GEOMETRY,
                execution_policy=common.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY,
                export_distance_product=export_distance_product_primary,
            ),
            stack_coreg_proc_interface_files.coreg_primary_config_file,
        )

        # Clean up the output directories.
        if (
            self.aux_pps.coregistration.coregistration_execution_policy
            is not common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ):
            for coreg_output_products in stack_coreg_proc_output_products:
                coreg_output_products.rmtree(ignore_errors=True)

        # Prepare the coregistration input files.
        stack_coreg_input_paths = []
        for index, stack_coreg_input_file in enumerate(
            fill_stack_coreg_processor_input_files(
                stack_pre_proc_output_products=stack_pre_proc_output_products,
                coreg_output_products=stack_coreg_proc_output_products,
                coreg_proc_interface_files=stack_coreg_proc_interface_files,
                bps_configuration_file=breakpoint_dir / BPS_CONF_FILENAME_XML,
                bps_log_file=bps_logger.get_log_file().absolute(),
                coreg_primary_image_index=coreg_primary_image_index,
                warping_only=self.aux_pps.coregistration.coregistration_execution_policy
                is common.CoregistrationExecutionPolicyType.WARPING_ONLY,
            )
        ):
            # The path of the BPSStackProcessor input file .xml
            stack_coreg_input_paths.append(stack_coreg_proc_interface_files.input_file(index))
            write_coreg_input_file(
                stack_coreg_input_file,
                stack_coreg_input_paths[-1],
            )

        # We simply copy over the coregistration primary from the preprocessing
        # brk directory to the coregistration brk directory. We will do this only
        # under nominal/warping-only configuration.
        if (
            self.aux_pps.coregistration.coregistration_execution_policy
            is not common.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY
        ):
            try:
                _copy_coreg_primary_product(
                    stack_pre_proc_output_products[coreg_primary_image_index],
                    stack_coreg_proc_output_products[coreg_primary_image_index],
                )
                # pylint: disable-next=broad-exception-caught
            except Exception as err:
                raise StackCoregProcessorRuntimeError(err) from err

        # If we are in Warping Only, we do not estimate the shifts of primary
        # vs primary.
        if (
            self.aux_pps.coregistration.coregistration_execution_policy
            is common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ):
            stack_coreg_input_paths[coreg_primary_image_index] = None

        return _execute_bps_stack_processor_multithreaded(
            input_paths=stack_coreg_input_paths,
            env=self.sta_p_env,
            executable=self.sta_p_bin,
            num_worker_threads=num_worker_threads,
        )


def _execute_bps_stack_processor_multithreaded(
    input_paths: list[Path | None],
    env: Environment,
    executable: str,
    num_worker_threads: int,
):
    """
    Execute the BPSStackProcessor in multiple threads.

    Parameters
    ----------
    input_paths: list[Optional[Path]],
        Paths to the coregistration input xml files.

    env: Environment
        The processor environmental variables.

    executable: str
        Path to the BPSStackProcessorExecutable.

    num_worker_threads: int
        Number of threads assigned to the job.

    Return
    ------
    tuple[bool]
        Flags that report which run was successful or not.

    """
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Spawns the STA_P coregistrator executable
        # (BPSStackProcessor). Returns true if the execution is successful or
        # the input path is None, false if the execution of the coregistation
        # failed.
        def run_bps_stack_processor(env, bps_sta_p_exec, input_path):
            if input_path is not None:
                try:
                    run_application(env, bps_sta_p_exec, input_path, 1)
                except RuntimeError:
                    return False
            return True

        # Run in multiple threads.
        return tuple(
            executor.map(
                lambda path: run_bps_stack_processor(env, executable, path),
                input_paths,
            )
        )


def _rearrange_stack(
    *,
    job_order: StackJobOrder,
    stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
    stack_coreg_proc_output_products: list[CoregistrationOutputProducts],
    stack_pre_proc_exec_products: dict,
    succeeded_mask: tuple[bool],
):
    """Get rid of products associated to failed coregistrations."""
    # If we succeeded on all coregitration tasks, we just skip.
    if all(succeeded_mask):
        return

    # Update the job-order object.
    job_order.input_stack = _select_elements(job_order.input_stack, keep=succeeded_mask)

    # Dump the pre-processor products.
    stack_pre_proc_output_products[:] = _select_elements(stack_pre_proc_output_products, keep=succeeded_mask)

    # Dump the coreg-processor products.
    stack_coreg_proc_output_products[:] = _select_elements(stack_coreg_proc_output_products, keep=succeeded_mask)

    # Find the new index for the coregitration primary.
    # NOTE: At this stage, the coreg primary index should be amongst the
    # succeeded executions.
    old_coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]
    new_coreg_primary_image_index = sum(succeeded_mask[0:old_coreg_primary_image_index])
    if old_coreg_primary_image_index != new_coreg_primary_image_index:
        bps_logger.info("New coreg primary image index: %d", new_coreg_primary_image_index)
        stack_pre_proc_exec_products["coreg_primary_image_index"] = new_coreg_primary_image_index

    # Dump the pre-processor execution products.
    stack_pre_proc_exec_products["l1a_product_data"] = _select_elements(
        stack_pre_proc_exec_products["l1a_product_data"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["l1a_product_luts"] = _select_elements(
        stack_pre_proc_exec_products["l1a_product_luts"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["l1a_product_luts_azm_axis"] = _select_elements(
        stack_pre_proc_exec_products["l1a_product_luts_azm_axis"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["l1a_product_luts_rng_axis"] = _select_elements(
        stack_pre_proc_exec_products["l1a_product_luts_rng_axis"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["l1a_product_focwindow_params"] = _select_elements(
        stack_pre_proc_exec_products["l1a_product_focwindow_params"],
        keep=succeeded_mask,
    )
    stack_pre_proc_exec_products["l1a_product_iono_corrections"] = _select_elements(
        stack_pre_proc_exec_products["l1a_product_iono_corrections"],
        keep=succeeded_mask,
    )
    stack_pre_proc_exec_products["faraday_rotations"] = _select_elements(
        stack_pre_proc_exec_products["faraday_rotations"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["rfi_indices"] = _select_elements(
        stack_pre_proc_exec_products["rfi_indices"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["stack_spatial_baselines"] = _select_elements(
        stack_pre_proc_exec_products["stack_spatial_baselines"], keep=succeeded_mask
    )
    stack_pre_proc_exec_products["stack_temporal_baselines"] = _select_elements(
        stack_pre_proc_exec_products["stack_temporal_baselines"], keep=succeeded_mask
    )

    # As for the baseline ordering, the indices always refer to the "full"
    # input stack, thus we simply drop the failing images (coregistration
    # primary will never be dropped since "failures" affecting the primary have
    # had caused the stack to exit with an error earlier.
    stack_pre_proc_exec_products["stack_spatial_ordering"] = _select_elements(
        stack_pre_proc_exec_products["stack_spatial_ordering"],
        keep=succeeded_mask,
    )
    stack_pre_proc_exec_products["stack_temporal_ordering"] = _select_elements(
        stack_pre_proc_exec_products["stack_temporal_baselines"],
        keep=succeeded_mask,
    )

    # Update the quality bitsets (weird formatting from black, so we disable).
    # fmt: off
    stack_pre_proc_exec_products["input_quality_bitset"] = (
        stack_pre_proc_exec_products["input_quality_bitset"][succeeded_mask, :]
    )
    stack_pre_proc_exec_products["preproc_quality_bitset"] = (
        stack_pre_proc_exec_products["preproc_quality_bitset"][succeeded_mask, :]
    )
    # fmt: on

    bps_logger.info(
        "Coregistration failed for some frames. New coregistation stack %s",
        [p.name for p in job_order.input_stack],
    )


def _copy_coreg_primary_product(
    stack_pre_proc_coreg_primary_output_product: StackPreProcessorOutputProducts,
    stack_coreg_primary_proc_output_products: CoregistrationOutputProducts,
):
    """Copy the product folder to the breakpoint folder and rename it."""
    # The temporary product folder.
    tmp_coreg_product = (
        stack_coreg_primary_proc_output_products.coreg_product.parent
        / stack_pre_proc_coreg_primary_output_product.raw_data_product.name
    )

    # Copy the content.
    stack_coreg_primary_proc_output_products.coreg_product.parent.mkdir(parents=True, exist_ok=True)
    copytree(
        stack_pre_proc_coreg_primary_output_product.raw_data_product,
        tmp_coreg_product,
    )

    # Rename the product.
    rename_product_folder(
        current_folder=tmp_coreg_product,
        new_folder=stack_coreg_primary_proc_output_products.coreg_product,
    )


def _get_relative_time_axes(
    raster_info: RasterInfo,
    roi: RegionOfInterest | None = None,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Get the relative time axes."""
    return (
        get_time_axis(raster_info, axis=0, roi=roi, absolute=False)[0],
        get_time_axis(raster_info, axis=1, roi=roi, absolute=False)[0],
    )


def _get_csa_roi(
    csa_raster_info: RasterInfo,
    *,
    data_raster_info: RasterInfo,
    data_roi: RegionOfInterest | None = None,
) -> RegionOfInterest | None:
    """Map the data ROI to a ROI onto the CSA product."""
    if data_roi is None:
        return None

    data_azm_axis, _ = get_time_axis(data_raster_info, axis=0, roi=data_roi, absolute=True)
    csa_azimuth_begin, csa_azimuth_end = toi_to_axis_slice(
        TimeOfInterest(time_begin=data_azm_axis[0], time_end=data_azm_axis[-1]),
        time_axis=get_time_axis(csa_raster_info, axis=0, absolute=True)[0],
    )

    # Compute and validate the ROI.
    csa_roi = (
        csa_azimuth_begin,
        0,
        csa_azimuth_end - csa_azimuth_begin + 1,
        csa_raster_info.samples,
    )
    raise_if_roi_is_invalid(csa_raster_info, csa_roi)

    return csa_roi


def _select_elements(elements: Iterable, keep: Iterable) -> tuple:
    """Select only elements specified by keep mask."""
    return tuple(el for el, ok in zip(elements, keep) if ok)


def _raster_info_equal(raster_p: RasterInfo, raster_s: RasterInfo) -> bool:
    """Check that 2 raster info are the same."""
    eps = np.power(10.0, -np.finfo(np.float64).precision)  # pylint: disable=no-member
    return (
        raster_p.lines == raster_s.lines
        and raster_p.samples == raster_s.samples
        and np.isclose(raster_p.lines_start - raster_s.lines_start, 0, atol=eps)
        and np.isclose(raster_p.samples_start, raster_s.samples_start, atol=eps)
        and np.isclose(raster_p.lines_step, raster_s.lines_step, atol=eps)
        and np.isclose(raster_s.samples_step, raster_s.samples_step, atol=eps)
    )
