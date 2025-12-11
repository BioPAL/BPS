# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Standalone Warper that uses the Stack Coregistrator
---------------------------------------------------
"""

from pathlib import Path
from shutil import rmtree

import numpy.typing as npt
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    iter_channels,
    open_product_folder,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.io.metadata import ECellType, RasterInfo
from bps.common import bps_logger
from bps.common.configuration import fill_bps_configuration_file, write_bps_configuration_file
from bps.common.io import common
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from bps.common.runner_helper import run_application
from bps.stack_cal_processor.core.utils import read_productfolder_data_by_channel, read_raster_info
from bps.stack_coreg_processor.input_file import BPSCoregProcessorInputFile, CoregProcessorInputFile
from bps.stack_coreg_processor.interface import write_coreg_configuration_file, write_coreg_input_file
from bps.stack_processor import __version__ as VERSION
from bps.stack_processor.execution.utils import setup_coreg_processor_env
from bps.stack_processor.interface.external.aux_pps import AuxiliaryStaprocessingParameters
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.internal.intermediates import StackPreProcessorOutputProducts
from bps.stack_processor.interface.internal.utils import (
    fill_stack_coreg_processor_config,
)


class StackWarpingManagerRuntimeError(RuntimeError):
    """The warping manager hit an error during execution."""


class StackWarpingManager:
    """
    Handle the warping of a stack image given azimuth and range shifts using
    the BPSStackProcessor coregistration binary.

    Parameters
    ----------
    job_order: StackJobOrder
        The job-order used by the stack.

    aux_pps: AuxiliaryStaprocessingParameters
        User configuration stored in the AUX PPS.

    stack_pre_proc_primary_output_product: StackPreProcessorOutputProducts
        The pre-processor's output products associated to the
        coregistratin primary image.

    breakpoint_dir: Path
        Path to the stack working directory.

    """

    def __init__(
        self,
        *,
        job_order: StackJobOrder,
        aux_pps: AuxiliaryStaprocessingParameters,
        stack_pre_proc_primary_output_products: StackPreProcessorOutputProducts,
        breakpoint_dir: Path,
        roi: RegionOfInterest | None = None,
    ):
        """Instantiate the object."""
        # Set the internal configuration.
        self.aux_pps = aux_pps
        self.job_order = job_order
        self.breakpoint_dir = breakpoint_dir
        self.roi = roi
        self.stack_pre_proc_primary_output_products = stack_pre_proc_primary_output_products
        self.sta_p_env, self.sta_p_bin = setup_coreg_processor_env(self.breakpoint_dir)

    def __call__(
        self,
        *,
        mpol_image: tuple[npt.NDArray[complex], ...],
        azimuth_shifts: npt.NDArray[float],
        range_shifts: npt.NDArray[float],
    ):
        """
        Execute the image warping in place.

        Parameters
        ----------
        mpol_image: tuple[npt.NDArray[complex], ...]
            The Npol [Nazm x Nrng] multi-polarimetric image.

        azimuth_shifts: npt.NDArray[float]  [px]
            The [Nazm x Nrng] azimuth shifts map in pixels.

        range_shifts: npt.NDArray[float]  [px]
            The [Nazm x Nrng] range shifts map in pixels.

        Raises
        ------
        StackWarpingManagerRuntimeError

        """
        # Initialize the working directory.
        workspace_dir = self.breakpoint_dir / "_stack_coreg_warping_manager_tmp"
        rmtree(workspace_dir, ignore_errors=True)
        workspace_dir.mkdir(exist_ok=True)

        # Write the image.
        input_image_pf_path = workspace_dir / "input_image_pf"
        _write_image_product_folder(
            primary_pf_path=self.stack_pre_proc_primary_output_products.raw_data_product,
            channels_data=mpol_image,
            output_pf_path=input_image_pf_path,
            roi=self.roi,
        )
        bps_logger.debug("Cached the input image in %s", input_image_pf_path)

        # Write the coregistration products.
        azimuth_shifts_pf_path = workspace_dir / "az_shifts_pf"
        _write_shifts_product_folder(
            primary_pf_path=self.stack_pre_proc_primary_output_products.raw_data_product,
            shifts_data=azimuth_shifts,
            output_pf_path=azimuth_shifts_pf_path,
            roi=self.roi,
        )
        bps_logger.debug("Cached the azimuth shifts in %s", azimuth_shifts_pf_path)

        range_shifts_pf_path = workspace_dir / "rg_shifts_pf"
        _write_shifts_product_folder(
            primary_pf_path=self.stack_pre_proc_primary_output_products.raw_data_product,
            shifts_data=range_shifts,
            output_pf_path=range_shifts_pf_path,
            roi=self.roi,
        )
        bps_logger.debug("Cached the range shifts in %s", range_shifts_pf_path)

        # Write the general configuration file.
        bps_configuration_file_path = workspace_dir / "coregConf.xml"
        write_bps_configuration_file(
            fill_bps_configuration_file(
                self.job_order.processor_configuration,
                task_name="STA_P",
                processor_name="STA_P",
                processor_version=bps_logger.get_version_in_logger_format(VERSION),
                node_name=bps_logger.get_default_logger_node(),
            ),
            bps_configuration_file_path,
        )

        # Prepare the coregistration configuration.
        coreg_configuration_file_path = workspace_dir / "coregWarpingConfig.xml"
        write_coreg_configuration_file(
            fill_stack_coreg_processor_config(
                aux_pps=self.aux_pps,
                coregistration_method=common.CoregistrationMethodType.GEOMETRY,
                execution_policy=common.CoregistrationExecutionPolicyType.WARPING_ONLY,
                export_distance_product=False,
            ),
            coreg_configuration_file_path,
        )

        # Write the coregistration input file.
        coreg_input_file_path = workspace_dir / "coregInput.xml"
        write_coreg_input_file(
            BPSCoregProcessorInputFile(
                coregistration_input=CoregProcessorInputFile(
                    primary_product=self.stack_pre_proc_primary_output_products.raw_data_product,
                    secondary_product=input_image_pf_path,
                    ecef_grid_product=self.stack_pre_proc_primary_output_products.xyz_product,
                    output_path=workspace_dir,
                    coreg_conf_file=coreg_configuration_file_path,
                    az_shifts_product=azimuth_shifts_pf_path,
                    rg_shifts_product=range_shifts_pf_path,
                ),
                bps_configuration_file=bps_configuration_file_path,
                bps_log_file=bps_logger.get_log_file().absolute(),
            ),
            coreg_input_file_path,
        )

        # Execute the coregistration.
        try:
            bps_logger.debug("Running %s", self.sta_p_bin)
            run_application(self.sta_p_env, self.sta_p_bin, coreg_input_file_path, 1)
        except RuntimeError as err:
            raise StackWarpingManagerRuntimeError("Failed to apply shifts") from err

        # Read the data and return.
        warped_image_pf_path = _bps_coreg_output_pf_path(input_image_pf_path)
        bps_logger.debug("Reading warped data from %s", warped_image_pf_path)

        warped_image_pf = open_product_folder(warped_image_pf_path)
        for pol, image in enumerate(mpol_image):
            image[...] = read_productfolder_data_by_channel(warped_image_pf, channel=pol, roi=self.roi)

        # Clean up the working dir.
        bps_logger.debug("Cleaning up the working directory %s", workspace_dir)
        rmtree(workspace_dir)


def _write_shifts_product_folder(
    primary_pf_path: Path,
    shifts_data: npt.NDArray[float],
    output_pf_path: Path,
    roi: RegionOfInterest | None,
):
    """Export a product folder with the coregistration shifts."""
    primary_pf = open_product_folder(primary_pf_path)
    output_pf = create_product_folder(output_pf_path)

    primary_raster_info = read_raster_info(primary_pf)
    if roi is not None:
        raise_if_roi_is_invalid(primary_raster_info, roi)

    new_raster_info = RasterInfo(
        lines=primary_raster_info.lines,
        samples=primary_raster_info.samples,
        celltype=ECellType.float64,
        filename=output_pf.get_channel_data(0).name,
    )
    new_raster_info.set_lines_axis(
        primary_raster_info.lines_start,
        primary_raster_info.lines_start_unit,
        primary_raster_info.lines_step,
        primary_raster_info.lines_step_unit,
    )
    new_raster_info.set_samples_axis(
        primary_raster_info.samples_start,
        primary_raster_info.samples_start_unit,
        primary_raster_info.samples_step,
        primary_raster_info.samples_step_unit,
    )

    metadata = create_new_metadata(num_metadata_channels=1)
    metadata.insert_element(new_raster_info)

    write_metadata(metadata_obj=metadata, metadata_file=output_pf.get_channel_metadata(0))
    write_raster_with_raster_info(
        raster_file=output_pf.get_channel_data(0),
        data=shifts_data,
        raster_info=new_raster_info,
    )


def _write_image_product_folder(
    primary_pf_path: Path,
    channels_data: tuple[npt.NDArray[complex], ...],
    output_pf_path: Path,
    roi: RegionOfInterest | None,
):
    """Export a product with the input image."""
    primary_pf = open_product_folder(primary_pf_path)
    output_pf = create_product_folder(output_pf_path)

    for i, channel in iter_channels(primary_pf):
        raster_info = channel.get_raster_info()
        if roi is not None:
            raise_if_roi_is_invalid(raster_info, roi)

        new_raster_info = RasterInfo(
            lines=raster_info.lines,
            samples=raster_info.samples,
            celltype=raster_info.cell_type,
            filename=output_pf.get_channel_data(i).name,
        )
        new_raster_info.set_lines_axis(
            raster_info.lines_start,
            raster_info.lines_start_unit,
            raster_info.lines_step,
            raster_info.lines_step_unit,
        )
        new_raster_info.set_samples_axis(
            raster_info.samples_start,
            raster_info.samples_start_unit,
            raster_info.samples_step,
            raster_info.samples_step_unit,
        )

        metadata = create_new_metadata(num_metadata_channels=1)
        metadata.insert_element(new_raster_info)

        # These has to be written otherwise the coregistrator
        # complains. However, only the raster info matters in warping only.
        metadata.insert_element(channel.get_swath_info())
        metadata.insert_element(channel.get_dataset_info())
        metadata.insert_element(channel.get_sampling_constants())
        metadata.insert_element(channel.get_doppler_centroid())

        write_metadata(metadata_obj=metadata, metadata_file=output_pf.get_channel_metadata(i))
        write_raster_with_raster_info(
            raster_file=output_pf.get_channel_data(i),
            data=channels_data[i],
            raster_info=new_raster_info,
        )


def _bps_coreg_output_pf_path(input_pf_path: Path) -> Path:
    """The BPSStackProcessor bin append the suffix _Cor to the output PF."""
    input_pf_name = input_pf_path.name
    return input_pf_path.parent / f"{input_pf_name}_Cor"
