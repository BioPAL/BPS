# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Calibration Execution Manager
-----------------------------------
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from bps.common import bps_logger
from bps.common.fnf_utils import read_fnf_mask
from bps.stack_cal_processor.configuration import (
    AZF_NAME,
    CAL_NAME,
    IOB_NAME,
    PPR_NAME,
    SKP_NAME,
    fill_stack_data_specs,
)
from bps.stack_cal_processor.core.azf.azimuthfilter import azimuth_spectral_filtering
from bps.stack_cal_processor.core.iob.backgroundiono import remove_background_ionosphere
from bps.stack_cal_processor.core.ppr.phaseremoval import remove_phase_plane
from bps.stack_cal_processor.core.skp.skpcalibration import skp_calibration
from bps.stack_cal_processor.core.skp.skpquality import SkpFnFQualityMask
from bps.stack_cal_processor.input_manager import (
    StackCalProcessorInputManager,
    StackCalProcessorInputProducts,
    select_calibration_reference_image,
)
from bps.stack_processor.interface.external.aux_pps import AuxiliaryStaprocessingParameters
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.utils import (
    fill_stack_cal_conf_from_aux_pps,
    parse_user_provided_calib_reference_image_index,
)
from bps.stack_processor.interface.internal.intermediates import (
    StackPreProcessorOutputProducts,
)


class StackCalExecutionManager:
    """
    Manage the execution of the calibration pipeline.

    Parameters
    ----------
    job_order: StackJobOrder
        The job-order used by the stack.

    aux_pps: AuxiliaryStaprocessingParameters
        User configuration stored in the AUX PPS.

    breakpoint_dir: Path
        Path to the breakpoint directory.

    fnf_mask_path: Path | None
        Optionally, a path to an FNF mask. Defaulted to None.

    Raises
    ------
    FileNotFoundError when the FNF path does not exist.

    """

    def __init__(
        self,
        *,
        job_order: StackJobOrder,
        aux_pps: AuxiliaryStaprocessingParameters,
        breakpoint_dir: Path,
        fnf_mask_path: Path | None = None,
    ):
        """Instantiate the object."""
        # Check that FNF path points to an existing file.
        if fnf_mask_path is not None and not fnf_mask_path.exists():
            raise FileNotFoundError(f"FNF: {fnf_mask_path}")

        # Store the inputs and external resources.
        self.job_order = job_order
        self.aux_pps = aux_pps
        self.breakpoint_dir = breakpoint_dir
        self.fnf_mask_path = fnf_mask_path

    def run(
        self,
        *,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
        stack_pre_proc_exec_products: dict,
        stack_coreg_proc_output_products: dict,
        stack_coreg_proc_exec_products: dict,
        lut_shift_exec_products: dict,
        num_worker_threads: int,
    ) -> dict:
        """
        Execute the calibration stack.

        Parameters
        ----------
        stack_pre_proc_output_products: dict
            The output products (intermediates) of the pre-processor.

        stack_pre_proc_exec_products: dict
            The output of the execution of the pre-processor.

        stack_coreg_proc_output_products: dict
            The output products (intermediates)  of the coreg processor.

        stack_coreg_proc_exec_products: dict
            The stack coregistration execution products.

        lut_shift_exec_products: dict
            The output products of the execution of LUT shifting.

        num_worker_threads: int
            Number of threads assigned to the calibration processor.

        Raises
        ------
        ValueError
            In case the number of threads is not positive.

        AzfRuntimeError
            In case the AZF crashes.

        InSarCalibrationRuntimeError
            In case the InSAR calibration fails.

        SkpRuntimeError
            In case the SKP crashes.

        """
        if num_worker_threads <= 0:
            raise ValueError("Number of threads must be positive")

        # Store the coregistration primary imgae index.
        coreg_primary_image_index = stack_pre_proc_exec_products["coreg_primary_image_index"]

        # Select the calibration primary image index.
        calib_reference_image_index = select_calibration_reference_image(
            polarization=self.aux_pps.slow_ionosphere_removal.polarization_used,
            reference=parse_user_provided_calib_reference_image_index(
                job_order=self.job_order,
                aux_pps=self.aux_pps,
                coreg_primary_image_index=coreg_primary_image_index,
            ),
            rfi_indices=stack_pre_proc_exec_products["rfi_indices"],
            faraday_decorrelation_indices=stack_pre_proc_exec_products["faraday_rotations"],
            coreg_primary_image_index=coreg_primary_image_index,
            input_stack_paths=self.job_order.input_stack,
        )

        # The input products of the calibration module.
        stack_cal_input_products = tuple(
            StackCalProcessorInputProducts(
                l1a_product_name=l1a_product_path.name,
                coreg_product=coreg_products.coreg_product,
                synth_geometry_product=coreg_products.synth_product,
                l1_iono_phase_screen_product=coreg_products.l1_iono_phase_screen_product,
                l1_iono_range_shifts_product=coreg_products.l1_iono_range_shifts_product,
                vertical_wavenumber_product=coreg_products.kz_product,
                azimuth_shifts_product=coreg_products.az_shifts_product,
                azimuth_geo_shifts_product=coreg_products.az_geo_shifts_product,
                range_shifts_product=coreg_products.rg_shifts_product,
                dist_product=coreg_products.distance_product,
            )
            for l1a_product_path, coreg_products, preproc_products in zip(
                self.job_order.input_stack,
                stack_coreg_proc_output_products,
                stack_pre_proc_output_products,
            )
        )

        # The configuration of the stack calibration.
        stack_cal_conf = fill_stack_cal_conf_from_aux_pps(
            aux_pps=self.aux_pps,
            polarizations=stack_pre_proc_exec_products["stack_polarizations"],
            skp_lut_azimuth_decimation_factor=lut_shift_exec_products["lut_azimuth_decimation_factor"],
            skp_lut_range_decimation_factor=lut_shift_exec_products["lut_range_decimation_factor"],
        )

        # Preparing the execution manager and kick-off the execution.
        input_manager = StackCalProcessorInputManager(
            stack_cal_input_products,
            stack_pre_proc_exec_products["coreg_primary_image_index"],
            stack_coreg_proc_exec_products["actualized_coregistration_parameters"],
            stack_pre_proc_exec_products["stack_polarizations"],
            roi=stack_pre_proc_exec_products["stack_roi"],
        )

        # Just few checks that the configuration are consistent.
        enabled_modules = {
            AZF_NAME: self.aux_pps.azimuth_spectral_filtering.azimuth_spectral_filtering_flag,
            IOB_NAME: self.aux_pps.slow_ionosphere_removal.slow_ionosphere_removal_flag,
            CAL_NAME: self.aux_pps.in_sar_calibration.in_sar_calibration_flag,
            SKP_NAME: self.aux_pps.skp_phase_calibration.skp_phase_estimation_flag,
        }
        if not any(enabled for _, enabled in enabled_modules.items()):
            bps_logger.warning("All calibration modules are disabled")
        bps_logger.info("Calibration stack's configuration properly loaded")

        # Reading the stack specicifations.
        bps_logger.info("Reading the stack specs")
        stack_data_specs = fill_stack_data_specs(
            coreg_products=input_manager.get_coreg_products(),
            coreg_primary_image_index=coreg_primary_image_index,
            window_compression_parameters=stack_pre_proc_exec_products["l1a_product_focwindow_params"],
            roi=stack_pre_proc_exec_products["stack_roi"],
        )

        # Kick off the calibration stack.
        bps_logger.info(
            "Kicking-off the calibration stack. Running [%s]",
            ", ".join(m for m, on in enabled_modules.items() if on) if any(enabled_modules.values()) > 0 else "nothing",
        )

        # Loading the data that are common to all modules.
        bps_logger.info("Loading the stack images")
        stack_images = input_manager.read_coreg_images()

        bps_logger.info("Loading the synthetic geometric phases (DSI)")
        synth_phases = input_manager.read_synth_geometry_images(
            bias_compensation=self.aux_pps.general.flattening_phase_bias_compensation_flag
        )

        bps_logger.info("Loading the vertical wavenumbers (Kz)")
        vertical_wavenumbers = input_manager.read_vertical_wavenumber_images()

        # Store all by-products of calibration.
        calibration_products = {}

        # Run the Azimuth Spectral Filtering (AzF).
        if enabled_modules[AZF_NAME]:
            calibration_products[AZF_NAME] = azimuth_spectral_filtering(
                stack=stack_images,
                synth_phases=synth_phases,
                doppler_centroids=input_manager.doppler_centroids(),
                conf=stack_cal_conf.azf_conf,
                stack_specs=stack_data_specs,
                coreg_primary_image_index=coreg_primary_image_index,
                max_num_threads=num_worker_threads,
                update_stack_specs=True,  # Shift the baz and set azimuth fc.
            )

        # Run the Slow Ionophere Removal (IoB).
        if enabled_modules[IOB_NAME]:
            calibration_products[IOB_NAME] = remove_background_ionosphere(
                stack=stack_images,
                synth_phases=synth_phases,
                vertical_wavenumbers=vertical_wavenumbers,
                range_coreg_shifts=input_manager.read_range_coreg_shifts(
                    bias_compensation=self.aux_pps.general.flattening_phase_bias_compensation_flag
                ),
                l1_iono_phases=input_manager.read_l1_iono_phase_screens_luts(),
                l1_iono_shifts=input_manager.read_l1_iono_range_shifts_luts(),
                conf=stack_cal_conf.iob_conf,
                stack_specs=stack_data_specs,
                calib_reference_image_index=calib_reference_image_index,
                max_num_threads=num_worker_threads,
            )

        # The InSAR phase calibration. As of now, this only runs the phase
        # plane removal.
        if enabled_modules[CAL_NAME]:
            calibration_products[CAL_NAME] = {
                PPR_NAME: remove_phase_plane(
                    stack=stack_images,
                    synth_phases=synth_phases,
                    conf=stack_cal_conf.ppr_conf,
                    stack_specs=stack_data_specs,
                    coreg_primary_image_index=coreg_primary_image_index,
                    max_num_threads=num_worker_threads,
                )
            }

        # Run the SKP calibration.
        if enabled_modules[SKP_NAME]:
            calibration_products[SKP_NAME] = skp_calibration(
                stack=stack_images,
                synth_phases=synth_phases,
                vertical_wavenumbers=vertical_wavenumbers,
                conf=stack_cal_conf.skp_conf,
                stack_specs=stack_data_specs,
                coreg_primary_image_index=coreg_primary_image_index,
                skp_fnf_mask=_read_skp_fnf_mask(
                    self.fnf_mask_path,
                    stack_pre_proc_exec_products,
                    lut_shift_exec_products,
                ),
                max_num_threads=num_worker_threads,
            )

        return {
            "stack_nodata_mask": input_manager.compute_nodata_mask(),
            "vertical_wavenumbers": vertical_wavenumbers,
            "flattening_phases": synth_phases,
            "stack_data_specs": stack_data_specs,
            "calibrated_stack_images": stack_images,
            "calibration_products": calibration_products,
            "calib_reference_image_index": calib_reference_image_index,
        }


def _read_skp_fnf_mask(
    fnf_mask_path: Path,
    stack_pre_proc_exec_products: dict,
    lut_shift_exec_products: dict,
) -> SkpFnFQualityMask | None:
    """Read the FNF mask."""
    if fnf_mask_path is None:
        return None

    # The Lat/Lon LUTs of the coregitration primary.
    coreg_primary_luts = lut_shift_exec_products["lut_data"][stack_pre_proc_exec_products["coreg_primary_image_index"]]
    coreg_primary_lut_lat = np.deg2rad(coreg_primary_luts["latitude"])
    coreg_primary_lut_lon = np.deg2rad(coreg_primary_luts["longitude"])

    return SkpFnFQualityMask(
        fnf_mask=read_fnf_mask(
            fnf_path=fnf_mask_path,
            latlon_roi=(
                np.min(coreg_primary_lut_lat),
                np.max(coreg_primary_lut_lat),
                np.min(coreg_primary_lut_lon),
                np.max(coreg_primary_lut_lon),
            ),
            units="rad",
            print_info=False,
        ),
        latitudes=coreg_primary_lut_lat,
        longitudes=coreg_primary_lut_lon,
        # All SKP-axes are all relative to grid of the primary.
        azimuth_axis=_rel_axis(lut_shift_exec_products["lut_primary_azm_axis"]),
        range_axis=_rel_axis(lut_shift_exec_products["lut_primary_rng_axis"]),
    )


def _rel_axis(array: npt.NDArray) -> npt.NDArray[float]:
    """Make an axis relative."""
    return (array - array[0]).astype(np.float64)
