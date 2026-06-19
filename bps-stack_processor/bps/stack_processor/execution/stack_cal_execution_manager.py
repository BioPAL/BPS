# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Calibration Execution Manager
-----------------------------------
"""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
from arepytools.io import open_product_folder
from arepytools.io.productfolder2 import ProductFolder2
from bps.common import bps_logger
from bps.common.configuration import fill_bps_configuration_file, write_bps_configuration_file
from bps.common.io import common
from bps.common.runner_helper import run_application
from bps.stack_cal_processor.configuration import (
    AZF_NAME,
    IOB_NAME,
    MSC_NAME,
    SKP_NAME,
    fill_stack_data_specs,
)
from bps.stack_cal_processor.core.azf.azimuthfilter import azimuth_spectral_filtering
from bps.stack_cal_processor.core.iob.backgroundiono import remove_background_ionosphere
from bps.stack_cal_processor.core.msc.mscalibration import multi_squint_calibration
from bps.stack_cal_processor.core.msc.utils import SecondaryStackManager
from bps.stack_cal_processor.core.skp.skpcalibration import skp_calibration
from bps.stack_cal_processor.core.skp.skpquality import SkpFnFQualityMask
from bps.stack_cal_processor.input_manager import (
    StackCalProcessorInputManager,
    StackCalProcessorInputProducts,
    select_calibration_reference_image,
)
from bps.stack_coreg_processor.input_file import BPSCoregProcessorInputFile, CoregProcessorInputFile
from bps.stack_coreg_processor.interface import write_coreg_configuration_file, write_coreg_input_file
from bps.stack_processor import __version__ as VERSION
from bps.stack_processor.execution.fnf import read_fnf_mask
from bps.stack_processor.execution.utils import setup_coreg_processor_env
from bps.stack_processor.interface.external.aux_pps import AuxiliaryStaprocessingParameters
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.utils import (
    fill_stack_cal_conf_from_aux_pps,
    parse_user_provided_calib_reference_image_index,
)
from bps.stack_processor.interface.internal.intermediates import (
    CoregistrationOutputProducts,
    StackPreProcessorOutputProducts,
)
from bps.stack_processor.interface.internal.utils import (
    fill_stack_coreg_processor_config,
)

# The stop and resume path for the warping.
STOP_AND_RESUME_PATH = "WARPING_COMPLETE.json"


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
            MSC_NAME: self.aux_pps.multi_squint_calibration.multi_squint_calibration_flag,
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

        # Run the Multi-Squint Calibration (MSC).
        if enabled_modules[MSC_NAME]:
            calibration_products[MSC_NAME] = multi_squint_calibration(
                stack=stack_images,
                secondary_stack_manager=SecondaryStackManager(
                    product_cor_pfs=_StackWarpingManager(
                        job_order=self.job_order,
                        aux_pps=self.aux_pps,
                        breakpoint_dir=self.breakpoint_dir,
                    ).warp_stack_multithreaded(
                        stack_pre_proc_output_products=stack_pre_proc_output_products,
                        stack_coreg_proc_output_products=stack_coreg_proc_output_products,
                        coreg_primary_image_index=stack_pre_proc_exec_products["coreg_primary_image_index"],
                        max_num_threads=num_worker_threads,
                    ),
                    coreg_primary_image_index=coreg_primary_image_index,
                    polarizations=stack_pre_proc_exec_products["stack_polarizations"],
                    roi=stack_pre_proc_exec_products["stack_roi"],
                ),
                synth_phases=synth_phases,
                conf=stack_cal_conf.msc_conf,
                stack_specs=stack_data_specs,
                coreg_primary_image_index=coreg_primary_image_index,
                max_num_threads=num_worker_threads,
            )

        # Run the SKP calibration.
        if enabled_modules[SKP_NAME]:
            calibration_products[SKP_NAME] = skp_calibration(
                stack=stack_images,
                synth_phases=synth_phases,
                vertical_wavenumbers=vertical_wavenumbers,
                stack_polarizations=stack_pre_proc_exec_products["stack_polarizations"],
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


class _StackWarpingManager:
    """
    Handle the warping of a stack given azimuth and range shifts using
    the BPSStackProcessor coregistration binary.

    Parameters
    ----------
    job_order: StackJobOrder
        The job-order used by the stack.

    aux_pps: AuxiliaryStaprocessingParameters
        User configuration stored in the AUX PPS.

    breakpoint_dir: Path
        Path to the stack working directory.

    """

    def __init__(
        self,
        *,
        job_order: StackJobOrder,
        aux_pps: AuxiliaryStaprocessingParameters,
        breakpoint_dir: Path,
    ):
        """Instantiate the object."""
        # Set the internal configuration.
        self.breakpoint_dir = breakpoint_dir
        self.breakpoint_dir.mkdir(parents=True, exist_ok=True)

        self.sta_p_env, self.sta_p_bin = setup_coreg_processor_env(self.breakpoint_dir)

        # Prepare the coregistration configuration. Coregistration method must
        # be set to "Geometry" otherwise the BPSStackProcessor stops.
        self.coreg_configuration_file_path = self.breakpoint_dir / "coregWarpingConfig.xml"
        write_coreg_configuration_file(
            fill_stack_coreg_processor_config(
                aux_pps=aux_pps,
                coregistration_method=common.CoregistrationMethodType.GEOMETRY,
                execution_policy=common.CoregistrationExecutionPolicyType.WARPING_ONLY,
                export_distance_product=False,
            ),
            self.coreg_configuration_file_path,
        )

        # Write the general configuration file.
        self.bps_configuration_file_path = self.breakpoint_dir / "coregConf.xml"
        write_bps_configuration_file(
            fill_bps_configuration_file(
                job_order.processor_configuration,
                task_name="STA_P",
                processor_name="STA_P",
                processor_version=bps_logger.get_version_in_logger_format(VERSION),
                node_name=bps_logger.get_default_logger_node(),
            ),
            self.bps_configuration_file_path,
        )

    def warp_stack_multithreaded(
        self,
        *,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
        stack_coreg_proc_output_products: list[CoregistrationOutputProducts],
        coreg_primary_image_index: int,
        max_num_threads: int = 1,
    ) -> tuple[ProductFolder2 | None, ...]:
        """Warp the secondary images of a stack by applying the coregistration
        shifts (including residuals from data) in range direction, and only
        using geometry in azimuth direction.

        Parameters
        ----------
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts]
            The pre-processor's output products.

        stack_coreg_proc_output_products: CoregistrationOutputProducts
            The stack coreg processor intermediate products (containing the
            coregistration shifts' products).

        coreg_primary_image_index: int
            The index of the coregistration primary image.

        max_num_threads: int = 1
            Number of threads assigned to the job.

        Raises
        ------
        StackWarpingManagerRuntimeError

        Return
        ------
        tuple[ProductFolder2 | None, ...]
            Product folders containing the warped images. None for the coregistration
            primary image.

        """
        # Reserve a list for storing the stack.
        nimages = len(stack_coreg_proc_output_products)

        # If the warping was already executed. Just read the data.
        if self._status_file_path().is_file():
            bps_logger.info("Stack without azimuth residual shifts already available. Loading stack")
            return self._read_warped_stack_from_status_file_multithreaded(num_images=nimages)

        # Run the coregisration process multithreaded.
        with ThreadPoolExecutor(max_workers=max_num_threads) as executor:
            bps_logger.info("Coregistering stack without azimuth residual shifts")

            # Prepare the output.
            self._cleanup_output_pf(stack_pre_proc_output_products)
            warped_stack_pfs = [None] * nimages

            # The warping coref function.
            def warp_image_fn(index_s):
                warped_stack_pfs[index_s] = self.warp_image(
                    stack_pre_proc_primary_output_products=stack_pre_proc_output_products[coreg_primary_image_index],
                    stack_pre_proc_secondary_output_products=stack_pre_proc_output_products[index_s],
                    stack_coreg_proc_secondary_output_products=stack_coreg_proc_output_products[index_s],
                )

            for _ in executor.map(
                warp_image_fn,
                (i for i in range(nimages) if i != coreg_primary_image_index),
            ):
                pass

            self._write_status_file(
                stack_pre_proc_output_products=stack_pre_proc_output_products,
                coreg_primary_image_index=coreg_primary_image_index,
            )

            return tuple(warped_stack_pfs)

    def warp_image(
        self,
        *,
        stack_pre_proc_primary_output_products: StackPreProcessorOutputProducts,
        stack_pre_proc_secondary_output_products: StackPreProcessorOutputProducts,
        stack_coreg_proc_secondary_output_products: CoregistrationOutputProducts,
    ) -> tuple[ProductFolder2, ...]:
        """Execute the image warping of a product."""
        # The output path.
        output_pf_path = self._bps_coreg_output_pf_path(stack_pre_proc_secondary_output_products)
        output_pf_path.parent.mkdir(exist_ok=True)

        # Write the coregistration input files.
        coreg_input_file_path = output_pf_path.parent / "coregInput.xml"
        write_coreg_input_file(
            BPSCoregProcessorInputFile(
                coregistration_input=CoregProcessorInputFile(
                    primary_product=stack_pre_proc_primary_output_products.raw_data_product,
                    secondary_product=stack_pre_proc_secondary_output_products.raw_data_product,
                    ecef_grid_product=stack_pre_proc_primary_output_products.xyz_product,
                    output_path=output_pf_path.parent,
                    coreg_conf_file=self.coreg_configuration_file_path,
                    az_shifts_product=stack_coreg_proc_secondary_output_products.az_geo_shifts_product,
                    rg_shifts_product=stack_coreg_proc_secondary_output_products.rg_shifts_product,
                ),
                bps_configuration_file=self.bps_configuration_file_path,
                bps_log_file=bps_logger.get_log_file().absolute(),
            ),
            coreg_input_file_path,
        )

        # Execute the coregistration.
        bps_logger.debug("Running %s", self.sta_p_bin)
        run_application(self.sta_p_env, self.sta_p_bin, coreg_input_file_path, 1)
        return open_product_folder(output_pf_path)

    def _write_status_file(
        self,
        *,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
        coreg_primary_image_index: int,
    ):
        """Write the status file with the output information."""
        warped_stack_info = {
            f"{i:02d}": str(self._bps_coreg_output_pf_path(stack_pre_proc_output_products[i]))
            for i in range(len(stack_pre_proc_output_products))
            if i != coreg_primary_image_index
        }
        if not all(Path(pf_path).exists() for pf_path in warped_stack_info.values()):
            raise RuntimeError("Invalid status file in stack_cal_processor's intermediate data dir")

        self._status_file_path().write_text(
            json.dumps(warped_stack_info, indent=2),
            encoding="utf-8",
        )

    def _status_file_path(self):
        """The status file generated only when the warping is complete."""
        return Path(self.breakpoint_dir / STOP_AND_RESUME_PATH)

    def _read_warped_stack_from_status_file_multithreaded(
        self,
        *,
        num_images: int,
    ) -> tuple[ProductFolder2 | None, ...]:
        """Initialize the product folders using the info stored in the status file."""
        # Read the status file.
        warped_stack_info = json.loads(
            self._status_file_path().read_text(encoding="utf-8"),
        )

        # Storage for the output.
        warped_stack_pfs = [None] * num_images
        for i in sorted([int(j) for j in warped_stack_info]):
            warped_stack_pfs[i] = open_product_folder(warped_stack_info[f"{i:02d}"])

        return tuple(warped_stack_pfs)

    def _bps_coreg_output_pf_path(
        self,
        stack_pre_proc_secondary_output_products: StackPreProcessorOutputProducts,
    ) -> Path:
        """The BPSStackProcessor bin appends the suffix _Cor to the output PF."""
        output_pf_name = stack_pre_proc_secondary_output_products.raw_data_product.name
        unique_id = stack_pre_proc_secondary_output_products.raw_data_product.parent.name
        output_pf_path = self.breakpoint_dir / unique_id / f"{output_pf_name}_Cor"
        return output_pf_path.resolve()

    def _cleanup_output_pf(
        self,
        stack_pre_proc_output_products: list[StackPreProcessorOutputProducts],
    ):
        """Cleanup the breakpoint directory."""
        # Cleanup the directories that store the output PFs, i.e. STA_P_01/.
        for preproc_product in stack_pre_proc_output_products:
            output_pf_path = self._bps_coreg_output_pf_path(preproc_product)
            if output_pf_path.exists():
                shutil.rmtree(output_pf_path)


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
            fnf_mask_path=fnf_mask_path,
            latitudes=coreg_primary_lut_lat,
            longitudes=coreg_primary_lut_lon,
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
