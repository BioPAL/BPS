# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Commands
--------
"""

from __future__ import annotations

import ast
import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from bps.common import bps_logger, retrieve_aux_product_data_single_content
from bps.common.fnf_utils import read_fnf_mask, retrieve_fnf_content
from bps.common.l2_joborder_tags import (
    L2A_OUTPUT_PRODUCT_FD,
    L2A_OUTPUT_PRODUCT_FH,
    L2A_OUTPUT_PRODUCT_GN,
    L2A_OUTPUT_PRODUCT_TFH,
)
from bps.common.translate_job_order import get_bps_logger_level
from bps.l2a_processor import BPS_L2A_PROCESSOR_NAME
from bps.l2a_processor.basins import BASINS
from bps.l2a_processor.core.aux_pp2_2a import GeneralConf
from bps.l2a_processor.core.parsing import parse_aux_pp2_2a, parse_l2a_job_order
from bps.l2a_processor.fd.fd_commands import FD
from bps.l2a_processor.fh.fh_commands import FH
from bps.l2a_processor.gn.gn_commands import GN
from bps.l2a_processor.io.aux_pp2_2a_models.models import CalibrationScreenType
from bps.l2a_processor.l2a_common_functionalities import (
    calibration,
    fast_read_baseline_ordering_index,
    forest_coverage_check,
    int_subsetting,
    refine_dgg_search_tiles,
    scs_axis_generation,
)
from bps.l2a_processor.processor_interface.xsd_validation import validate_aux_pp2_2a
from bps.l2a_processor.tomo_fh.tomo_fh_commands import TOMO_FH
from bps.transcoder.sarproduct.biomass_l2aproduct_reader import BIOMASSL2aProductReader
from bps.transcoder.sarproduct.biomass_l2aproduct_writer import FLOAT_NODATA_VALUE
from bps.transcoder.sarproduct.biomass_stackproduct_reader import (
    BIOMASSStackProductReader,
)
from bps.transcoder.sarproduct.mph import read_coverage_and_footprint
from bps.transcoder.sarproduct.sta.quality_index import StackQualityIndex
from bps.transcoder.utils.dgg_utils import dgg_search_tiles
from osgeo import ogr

warnings.filterwarnings("ignore", message="invalid value encountered in")
warnings.filterwarnings("ignore", message="divide by zero encountered")

TROPIC_OF_CANCER_LATITUDE = 23.45
TROPIC_OF_CAPRICORN_LATITUDE = -23.45


def run_l2a_processing(
    job_order_file: Path,
    working_dir: Path,
):
    """Performs processing as described in job order.

    Parameters
    ----------
    job_order_file : Path
        job_order xml file
    working_dir : Path
        Working directory: the directory where the orchestrator will write the internal input files.
    """

    assert job_order_file.exists()
    assert working_dir.exists()
    assert job_order_file.is_absolute()
    assert working_dir.is_absolute()

    processing_start_time = datetime.now()

    # Input parsing: joborder
    job_order = parse_l2a_job_order(job_order_file.read_text())

    # Relative paths in the joborder are intended as relative to the directory where the JobOrder is
    job_order_dir = job_order_file.parent
    if not job_order.output_directory.is_absolute():
        job_order.output_directory = job_order_dir.joinpath(job_order.output_directory)

    # Update logging level
    log_level = get_bps_logger_level(
        job_order.processor_configuration.stdout_log_level,
        job_order.processor_configuration.stderr_log_level,
    )
    bps_logger.update_logger(loglevel=log_level)

    # Input parsing: aux_pp2_2a
    bps_logger.info("AUX PP2 2A configuration: %s", job_order.aux_pp2_2a_path)
    aux_pp2_2a_file = retrieve_aux_product_data_single_content(job_order.aux_pp2_2a_path)
    validate_aux_pp2_2a(aux_pp2_2a_file)
    aux_pp2_2a = parse_aux_pp2_2a(aux_pp2_2a_file.read_text())

    # Read input L2a product
    if job_order.input_l2a_fd_product:
        bps_logger.info(
            "L2a FD additional product is specified in Job Order, %s :",
            job_order.input_l2a_fd_product,
        )
        l2a_fd_product = BIOMASSL2aProductReader(job_order.input_l2a_fd_product).read()
    else:
        bps_logger.info("Optional L2a product not specified in Job Order")
        l2a_fd_product = None

    # Perform INT subsetting in case of TOM phase
    if (
        aux_pp2_2a.general.subsetting_rule.value == GeneralConf.SubsettingRules.MAINTAIN_ALL.value
        and (aux_pp2_2a.fd.enable_product_flag or aux_pp2_2a.fh.enable_product_flag)
        and len(job_order.input_stack_acquisitions) > 3
    ):
        raise ValueError(
            f"AUX PP2 2A configuration subsettingRule {GeneralConf.SubsettingRules.MAINTAIN_ALL.value} cannot be used if L2a FD or L2a FH are enabled. It works only for L2a GN and TOMO FH."
        )

    # getting input mission phase ID from primary image main annotation
    mission_phase_id = fast_read_mission_phase_id(job_order.input_stack_acquisitions)

    (
        selected_baselines_indices,
        input_stack_mph_files_sub,
    ) = int_subsetting(
        list(job_order.input_stack_mph_files),
        job_order.input_stack_acquisitions,
        aux_pp2_2a.general.subsetting_rule.value,
        mission_phase_id,
    )

    # Read input L1c stack products
    stack_products_list = []
    stack_lut_list = []
    lut_axes_primary_list = []
    acquisition_paths_selected = []
    primary_image_index = None  # it will be filled during product reading
    bps_logger.info("Reading input stack acquisitions")
    counter = 0
    input_baselines_indices = []
    for acquisition_path in job_order.input_stack_acquisitions:
        input_baselines_indices.append(fast_read_baseline_ordering_index(acquisition_path))

    baselines_sorted_as_read = []  # used for debugging
    for acquisition_path, current_baseline_ordering_index in zip(
        job_order.input_stack_acquisitions, input_baselines_indices
    ):
        if current_baseline_ordering_index in selected_baselines_indices:
            bps_logger.debug(f"Reading acquisition with baseline indice #{current_baseline_ordering_index}")
            baselines_sorted_as_read.append(current_baseline_ordering_index)

            # read only the products selected by int subsetting routine
            stack_product_obj = BIOMASSStackProductReader(acquisition_path, nodata_fill_value=np.nan)

            stack_products_list.append(stack_product_obj.read())
            acquisition_paths_selected.append(acquisition_path)

            decoded_quality = StackQualityIndex.decode(stack_product_obj.stack_quality.overall_product_quality_index)
            decoded_quality_string = [k.replace("_", " ") for k, v in decoded_quality.__dict__.items() if v is True]
            if len(decoded_quality_string):
                bps_logger.warning(
                    f"Input L1c acquisition staQuality overallProductQualityIndex={stack_product_obj.stack_quality.overall_product_quality_index}: {decoded_quality_string[0]}"
                )
            if stack_product_obj.stack_processing_parameters.skp_phase_correction_flag:
                bps_logger.debug(f"SKP phase correction has been applied to input L1c acquisition {acquisition_path}")
                if counter == 0 and aux_pp2_2a.general.apply_calibration_screen != CalibrationScreenType.NONE:
                    if aux_pp2_2a.general.apply_calibration_screen == CalibrationScreenType.GEOMETRY:
                        raise ValueError(
                            f"SKP phase correction has been applied to input L1c acquisitions, but L2 AUX PP specifies apply_calibration_screen = {CalibrationScreenType.GEOMETRY.value}"
                        )

                    bps_logger.warning(
                        "Disabling apply calibration screen flag: SKP phase correction has been already applied to input L1c acquisitions"
                    )
                    aux_pp2_2a.general.apply_calibration_screen = CalibrationScreenType.NONE

            else:
                bps_logger.debug(
                    f"SKP phase correction has NOT been applied to input L1c acquisition {acquisition_path}"
                )
            if (
                counter
                == 0  # this is to absure entering here, in case no reference is present (TBD for int subsetting)
                or stack_product_obj.stack_coregistration_parameters.primary_image
                == stack_product_obj.stack_coregistration_parameters.secondary_image
            ):
                if (
                    stack_product_obj.stack_coregistration_parameters.primary_image
                    == stack_product_obj.stack_coregistration_parameters.secondary_image
                ):
                    # There is not a specific flag telling which one is the primary image
                    # this is the way to check for it.
                    # Primary Image Index is a zero-based index of the primary image:
                    # the index is respect the order as found in the job order (NOT baselines-sorted)
                    primary_image_index = counter

            # manually read the LUTs from the product
            (
                lut,
                lut_axes_primary,
                _,
            ) = stack_product_obj.read_lut_annotation()
            # Convert no data values to nan, for the processing
            for lut_key, data in lut.items():
                if lut_key in [
                    "sigmaNought",
                    "denoisingHH",
                    "denoisingXX",
                    "denoisingVV",
                    "height",
                    "latitude",
                    "longitude",
                    "incidenceAngle",
                    "terrainSlope",
                    "waveNumbers",
                    "skpCalibrationPhaseScreen",
                    "flatteningPhaseScreen",
                ]:
                    nan_mask = data == FLOAT_NODATA_VALUE
                    lut[lut_key][nan_mask] = np.nan

            stack_lut_list.append(lut)
            lut_axes_primary_list.append(lut_axes_primary)
            counter += 1
    del lut

    for stack_product_to_check in stack_products_list:
        if not stack_product_to_check.polarization_list == ["H/H", "X/X", "V/V"]:
            error_msg = f"Wrong polarizations found in input product: {stack_product_to_check.polarization_list}; L2A works with ['H/H', 'X/X', 'V/V'] instead."
            raise ValueError(error_msg)

    # Phase screen to be used:
    skp_phases_screen = None
    geometry_phases_screen = None
    if aux_pp2_2a.general.apply_calibration_screen == CalibrationScreenType.SKP:
        skp_phases_screen = [lut["skpCalibrationPhaseScreen"].astype(np.float64) for lut in stack_lut_list]
        geometry_phases_screen = [lut["flatteningPhaseScreen"].astype(np.float64) for lut in stack_lut_list]
        assert skp_phases_screen is not None
        assert geometry_phases_screen is not None
        assert skp_phases_screen[0].shape == stack_lut_list[0]["waveNumbers"].shape
        assert geometry_phases_screen[0].shape == stack_lut_list[0]["waveNumbers"].shape

    if aux_pp2_2a.general.apply_calibration_screen == CalibrationScreenType.GEOMETRY:
        geometry_phases_screen = [lut["flatteningPhaseScreen"].astype(np.float64) for lut in stack_lut_list]
        assert geometry_phases_screen is not None
        assert geometry_phases_screen[0].shape == stack_lut_list[0]["waveNumbers"].shape

    if primary_image_index is None:
        # Enter here if no reference image in input stack,
        # or if reference image has been discarted during INT selection
        primary_image_index, lut_wavenumbers_reset = reset_primary_image(
            [lut["waveNumbers"] for lut in stack_lut_list],
            selected_baselines_indices,
        )

        # re fill the stack_lut_list with recomputed wavenumbers
        for original_luts, lut_wavenumber_reset in zip(stack_lut_list, lut_wavenumbers_reset):
            original_luts["waveNumbers"] = lut_wavenumber_reset

    # Sort read inputs basing on wavenumbers
    wavenumbers_sorting_indices = np.argsort([np.mean(lut["waveNumbers"]) for lut in stack_lut_list])
    stack_products_list = [stack_products_list[idx] for idx in wavenumbers_sorting_indices]
    acquisition_paths_selected = [acquisition_paths_selected[idx] for idx in wavenumbers_sorting_indices]
    stack_lut_list = [stack_lut_list[idx] for idx in wavenumbers_sorting_indices]
    primary_image_index = list(wavenumbers_sorting_indices).index(primary_image_index)
    input_stack_mph_files_sub = [input_stack_mph_files_sub[idx] for idx in wavenumbers_sorting_indices]
    lut_axes_primary_list = [lut_axes_primary_list[idx] for idx in wavenumbers_sorting_indices]
    if skp_phases_screen is not None:
        skp_phases_screen = [skp_phases_screen[idx] for idx in wavenumbers_sorting_indices]
    if geometry_phases_screen is not None:
        geometry_phases_screen = [geometry_phases_screen[idx] for idx in wavenumbers_sorting_indices]

    # Fast MPH parsing to get input L1c stack cumulative footprint
    latlon_coverage, footprints_list = read_coverage_and_footprint(input_stack_mph_files_sub)

    # Get BASINS IDs
    basin_id_list = get_basins(latlon_coverage, stack_products_list[primary_image_index].footprint)

    # Read input FNF, basing on footprint
    bps_logger.info(f"Reading input FNF from {retrieve_fnf_content(job_order.fnf_directory)}")
    fnf_mask_obj = read_fnf_mask(
        retrieve_fnf_content(job_order.fnf_directory),
        latlon_coverage,
        units="deg",
    )

    # Check forest coverage percentage inside L1c stack footprint
    # Use deepcopy to not modify fnf
    do_computation, forest_coverage_percentage = forest_coverage_check(
        deepcopy(fnf_mask_obj),
        aux_pp2_2a.general.forest_coverage_threshold,
        [footprint.to_list() for footprint in footprints_list],
    )

    if do_computation:
        # Common functionalities
        # get scs axes from sar image main annotation xml of primary image
        scs_axes_dict = {}
        (
            scs_axes_dict["scs_axis_sr_s"],
            scs_axes_dict["scs_axis_az_s"],
            scs_axes_dict["scs_axis_az_mjd"],
        ) = scs_axis_generation(stack_products_list[primary_image_index])

        assert stack_products_list[0].data_list[0].shape == (
            len(scs_axes_dict["scs_axis_az_s"]),
            len(scs_axes_dict["scs_axis_sr_s"]),
        )

        stack_lut_axes_dict = create_lut_axis(lut_axes_primary_list[primary_image_index], scs_axes_dict)

        if aux_pp2_2a.general.apply_calibration_screen != CalibrationScreenType.NONE:
            assert geometry_phases_screen[0].shape == (
                len(stack_lut_axes_dict["axis_primary_az_s"]),
                len(stack_lut_axes_dict["axis_primary_sr_s"]),
            )
            if aux_pp2_2a.general.apply_calibration_screen == CalibrationScreenType.SKP:
                assert skp_phases_screen[0].shape == (
                    len(stack_lut_axes_dict["axis_primary_az_s"]),
                    len(stack_lut_axes_dict["axis_primary_sr_s"]),
                )

        # Calibration
        # optupt will be a list with with P polarizations
        #   each containing a list with M acquisitions
        scs_pol_list_calibrated = calibration(
            [product.data_list for product in stack_products_list],
            scs_axes_dict["scs_axis_sr_s"],
            scs_axes_dict["scs_axis_az_s"],
            skp_phases_screen,
            geometry_phases_screen,
            stack_lut_axes_dict["axis_primary_sr_s"],
            stack_lut_axes_dict["axis_primary_az_s"],
            aux_pp2_2a.general.apply_calibration_screen.value,
        )

        # Call Forest Disturbance processor
        if L2A_OUTPUT_PRODUCT_FD in job_order.output_products:
            if not aux_pp2_2a.fd.enable_product_flag:
                bps_logger.warning(
                    "FD processing is enabled in Job Order Task, but disabled in AUX PP2 2A: AUX PP2 2A flag will be ignored."
                )

            fd_obj = FD(
                job_order,
                aux_pp2_2a,
                working_dir,
                stack_products_list,
                scs_axes_dict,
                scs_pol_list_calibrated,
                stack_lut_list,
                stack_lut_axes_dict,
                primary_image_index,
                acquisition_paths_selected,
                mission_phase_id,
                fnf_mask_obj,
                latlon_coverage,
                forest_coverage_percentage,
                basin_id_list,
                l2a_fd_product,  # type: ignore
            )
            fd_obj.run_l2a_fd_processing()

        # Call Forest Height processor
        if L2A_OUTPUT_PRODUCT_FH in job_order.output_products:
            if not aux_pp2_2a.fh.enable_product_flag:
                bps_logger.warning(
                    "FH processing is enabled in Job Order Task, but disabled in AUX PP2 2A: AUX PP2 2A flag will be ignored."
                )

            bps_logger.info("\n")
            fh_obj = FH(
                job_order,
                aux_pp2_2a,
                working_dir,
                stack_products_list,
                scs_axes_dict,
                scs_pol_list_calibrated,
                stack_lut_list,
                stack_lut_axes_dict,
                primary_image_index,
                acquisition_paths_selected,
                mission_phase_id,
                fnf_mask_obj,
                latlon_coverage,
                forest_coverage_percentage,
                basin_id_list,
            )
            fh_obj.run_l2a_fh_processing()

        # Call Ground Notching processor (L2a GN)
        if L2A_OUTPUT_PRODUCT_GN in job_order.output_products:
            if not aux_pp2_2a.agb.enable_product_flag:
                bps_logger.warning(
                    "GN processing is enabled in Job Order Task, but disabled in AUX PP2 2A: AUX PP2 2A flag will be ignored."
                )

            bps_logger.info("\n")
            gn_obj = GN(
                job_order,
                aux_pp2_2a,
                working_dir,
                stack_products_list,
                scs_axes_dict,
                scs_pol_list_calibrated,
                stack_lut_list,
                stack_lut_axes_dict,
                primary_image_index,
                acquisition_paths_selected,
                mission_phase_id,
                fnf_mask_obj,
                latlon_coverage,
                forest_coverage_percentage,
                basin_id_list,
            )
            gn_obj.run_l2a_gn_processing()

        # Call Tomo Forest Height processor
        if L2A_OUTPUT_PRODUCT_TFH in job_order.output_products:
            if not aux_pp2_2a.tfh.enable_product_flag:
                bps_logger.warning(
                    "TOMO FH processing is enabled in Job Order Task, but disabled in AUX PP2 2A: AUX PP2 2A flag will be ignored."
                )

            bps_logger.info("\n")
            tomo_fh_obj = TOMO_FH(
                job_order,
                aux_pp2_2a,
                working_dir,
                stack_products_list,
                scs_axes_dict,
                scs_pol_list_calibrated,
                stack_lut_list,
                stack_lut_axes_dict,
                primary_image_index,
                acquisition_paths_selected,
                mission_phase_id,
                fnf_mask_obj,
                latlon_coverage,
                forest_coverage_percentage,
                basin_id_list,
            )
            tomo_fh_obj.run_l2a_tomo_fh_processing()

    processing_stop_time = datetime.now()
    elapsed_time = processing_stop_time - processing_start_time
    bps_logger.info(
        "%s total processing time: %.3f s",
        BPS_L2A_PROCESSOR_NAME,
        elapsed_time.total_seconds(),
    )


def _rebuild_subsampled_axis(regular_axis: np.ndarray, subsampled_axis: np.ndarray) -> np.ndarray:
    """From L1 PFD:
    all LUTs except the fast-ionosphere-LUTs will be defined on a grid obtained by subsampling
    the primary data grid using range and azimuth steps that best approximate the target resolution for each axis

    This function is used to reconstruct LUT axis using from SCS data regular axis sampling
    """

    subsampled_axis_sampling = (subsampled_axis[-1] - subsampled_axis[0]) / (subsampled_axis.size - 1)
    regular_axis_sampling = (regular_axis[-1] - regular_axis[0]) / (regular_axis.size - 1)
    subsampling_factor = np.rint(subsampled_axis_sampling / regular_axis_sampling).astype(int)

    return regular_axis[0::subsampling_factor]


def create_lut_axis(
    lut_axes_primary: tuple[np.ndarray, np.ndarray],
    scs_axes_dict: dict[str, np.ndarray],
) -> dict:
    """
    Create the LUT axes, as BPS L2a expects,
    axes are also reconstructed using scs data regular axis sampling
    See also _rebuild_subsampled_axis(), scs_axis_generation()


    Parameters
    ----------
    lut_axes_primary: Tuple[np.ndarray, np.ndarray]
        First tuple entry contains azimuth primary LUT axis, in absolute Mjd format
        Second tuple entry contains slant range primary LUT axis, in absolute seconds
        Those axes are valid for all the LUTs used in L2a processor
    scs_axes_dict: Dict[str, np.ndarray]
        Dict containing scs data axis in the keys "scs_axis_az_s" and "scs_axis_rg_s"

    Returns
    -------
    stack_lut_axes_dict: Dict
        containing following keys:
            "axis_primary_az_s"
            "axis_primary_sr_s"
        Respect to the input "lut_axes_primary":
        Azimuth axis_primary_az_s are relative (subtracting the first axis value)
        Slant range axis_primary_sr_s are absolute: the start time is always the axis_primary_sr_s first axis value
    """
    # Azimuth axis should be relative
    # Range axis shoud be absolute

    # Absolute to relative
    lut_axis_az_s = (lut_axes_primary[0] - lut_axes_primary[0][0]).astype(np.float64)

    # Already absolute
    lut_axis_rg_s = (lut_axes_primary[1]).astype(np.float64)

    # Resample the axis basing on regular SCS data axis
    stack_lut_axes_dict = {}
    stack_lut_axes_dict["axis_primary_az_s"] = _rebuild_subsampled_axis(scs_axes_dict["scs_axis_az_s"], lut_axis_az_s)
    stack_lut_axes_dict["axis_primary_sr_s"] = _rebuild_subsampled_axis(scs_axes_dict["scs_axis_sr_s"], lut_axis_rg_s)

    return stack_lut_axes_dict


def reset_primary_image(
    lut_wavenumbers_list: list[np.ndarray],
    selected_baselines_indices_not_sorted: list[int],
) -> tuple[int, list[np.ndarray]]:
    """
    Reset the primary image, if not present in the L1c stack
    or if not present after INT subsetting rule (TOM phase)

    Parameters
    ----------
    lut_wavenumbers_list: List[np.ndarray]
        Vertical wavenumber matrices from the L1c LUT, for each L1c Product
    selected_baselines_indices_not_sorted: List[int]
        zero-based, ordered indices of the selected baseline to operate as in INT phase
        Not sorted, they are ordered as found in the job order

    Returns
    -------
    primary_image_index: int
        zero-based index of the new selected primary image:
        the index is respect the order as found in the job order (NOT baselines-sorted)
    lut_wavenumbers_list: List[np.ndarray]
        Vertical wavenumber matrices from the L1c LUT, for each L1c Product
        rescaled considering the new reference (which has zero vertical wavenumbers)
    """

    bps_logger.info("Reference image not found in read BIOMASS L1c products:")
    bps_logger.info("    resetting reference image to the L1c Product having lower median vertical wavenumber value")
    # get the median value of each vertical wavenumber LUT
    median_vert_wn_values = [np.nanmedian(vw) for vw in lut_wavenumbers_list]
    medians_string = str([f"{median_value:2.2f}" for median_value in median_vert_wn_values])[1:-1].replace("'", "")
    bps_logger.info(f"    median values of each vertical wavenumber from L1c LUTs: {medians_string} [deg/m]")
    # Get the crescent vertical wavenumber order indexing, basing on median value:
    # Wavenumber of first index will be new reference
    ordered_vert_wn_indices = np.argsort(median_vert_wn_values)
    primary_image_index = ordered_vert_wn_indices[0]

    bps_logger.info(
        f"    new reference image is the L1c Product having zero-based baseline oredring index: {selected_baselines_indices_not_sorted[primary_image_index]}"
    )

    # Rescale the verticsal wavenumbers
    bps_logger.info("    rescaling vertical wavenumbers values of each L1c Product, basing on new reference")
    new_primary_original_vert_wavenumber = lut_wavenumbers_list[primary_image_index]
    nan_mask = np.isnan(new_primary_original_vert_wavenumber)
    for idx in range(len(lut_wavenumbers_list)):
        if idx == primary_image_index:
            lut_wavenumbers_list[idx] = np.zeros(new_primary_original_vert_wavenumber.shape, dtype=np.float32)
            lut_wavenumbers_list[idx][nan_mask] = np.nan
        else:
            lut_wavenumbers_list[idx] = lut_wavenumbers_list[idx] - new_primary_original_vert_wavenumber

    median_vert_wn_values = [np.nanmean(vw) for vw in lut_wavenumbers_list]
    means_string = str([f"{mean_value:2.2f}" for mean_value in median_vert_wn_values])[1:-1].replace("'", "")
    bps_logger.info(f"    mean values of the vertical wavenumbers LUTs after reset: {means_string} [deg/m]")
    return primary_image_index, lut_wavenumbers_list


def _sph_geometry_to_lat_lon(geometry_ref):
    "Read coordinates of one feature from the Natural Earth shapefile features (@ naturalearthdata.com, Admin 0 - Countries) (a single continent)"

    geometry_info_string = geometry_ref.ExportToJson()
    idx_coordinates = geometry_info_string.find("coordinates") + len("coordinates")
    list_evaluated = ast.literal_eval(geometry_info_string[idx_coordinates + 3 : -2])
    swath_coordinates = []
    for current_value in list_evaluated:
        swath_coordinates.append(np.squeeze(np.array(current_value)))

    return swath_coordinates


def _basin_check_intersection(
    country_swath_coordinates: list[np.ndarray],
    basin_matrix_lon_idx,
    basin_matrix_lat_idx,
):
    """BASIN check intersection:
    Description
    -----------
    Internal function called by get_basin(), to check if input STA product intercets or not the country coordinartes
    Country and STA coordinates/footprints are lists, so two "For" nested loops are needed.

    Parameters
    ----------
    country_swath_coordinates: list[np.ndarray]
        Coordinates of a single sph feature of the Natural Earth Shapefile (@ naturalearthdata.com, Admin 0 - Countries) (a single continent)
        Each list entry is a 2D array of dimensions [longitudes x latitudes] in degrees
    footprints_list: list[list[float]]
        One footprint for each input STA product:
        each list entry is a footprint: a list containing [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon] in degrees

    Returns
    -------
    is_in_this_country: bool
        True if the STA product is intersecting the continent, False otherwise
    """

    is_in_this_country = False

    # ogr polygon of the input STA footprint
    lat_min_matrix = basin_matrix_lat_idx - 90.0
    lat_max_matrix = lat_min_matrix + 1.0
    lon_min_matrix = basin_matrix_lon_idx - 180.0
    lon_max_matrix = lon_min_matrix + 1.0
    wkt_stack_input = f"POLYGON (({lon_min_matrix} {lat_min_matrix}, {lon_min_matrix} {lat_max_matrix}, {lon_max_matrix} {lat_max_matrix}, {lon_max_matrix} {lat_min_matrix}, {lon_min_matrix} {lat_min_matrix}))"

    poly_stack = ogr.CreateGeometryFromWkt(wkt_stack_input)

    # Cycle over the input Country swath coordinates (a country in the Natural Earth data may be split in several coordinates)
    for swath_coordinate in country_swath_coordinates:
        # ogr polygon of the current tile footprint
        lat_min_tile = np.min(swath_coordinate[:, 1])
        lat_max_tile = np.max(swath_coordinate[:, 1])
        lon_min_tile = np.min(swath_coordinate[:, 0])
        lon_max_tile = np.max(swath_coordinate[:, 0])
        wkt_tile = f"POLYGON (({lon_min_tile} {lat_min_tile}, {lon_min_tile} {lat_max_tile}, {lon_max_tile} {lat_max_tile}, {lon_max_tile} {lat_min_tile}, {lon_min_tile} {lat_min_tile}))"
        poly_tile = ogr.CreateGeometryFromWkt(wkt_tile)

        # Check if the Country intersects the STA footprint
        if not poly_tile.Intersection(poly_stack).ExportToWkt() == "POLYGON EMPTY":
            is_in_this_country = True
            break

    return is_in_this_country, lat_min_matrix, lat_max_matrix


def get_basins(latlon_coverage: list[float], sta_footprint: list[float]) -> list[str]:
    """
    Get the BASIN ID where the input STA products are located

    Parameters
    ----------
    latlon_coverage:list[float]
        Cumulative coverage of all the input STA products, in degrees
        [min_lat, max_lat, min_lon, max_lon]

    sta_footprint:list[float]
         Cumulative footprint of the input STA products, in degrees
          [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon]

    Returns
    -------
    basin_id_list: list[str]:
        List of BASIN IDs
            South America tropical: "B100"
            South America temperate: "B200"
            Africa tropical/temperate: "B300"
            Asia temperate: "B400"
            Asia tropical: "B500"
            Oceania tropical/temperate: "B600"
            Exclusion zone: "B700"
    """

    world_basin_ids = BASINS().convert_to_world_matrix()
    (
        _,
        _,
        _,
        tiles_dict,
    ) = dgg_search_tiles(latlon_coverage, True)
    # Refine the above search, searching the tiles footprints falling in the STA footprint
    tile_id_list = refine_dgg_search_tiles(
        tiles_dict,
        np.array(sta_footprint)[:, 0],
        np.array(sta_footprint)[:, 1],
    )

    # Check basin ID of each tile
    basin_id_list = []
    for tile_id in tile_id_list:
        lat_sign = 1 if tile_id[0] == "N" else -1
        lon_sign = 1 if tile_id[3] == "E" else -1
        world_basins_lat_idx = lat_sign * int(tile_id[1:3]) + 90
        world_basins_lon_idx = lon_sign * int(tile_id[4:7]) + 180
        basin_id = f"B{world_basin_ids[world_basins_lon_idx, world_basins_lat_idx]}".ljust(4, "0")
        if basin_id not in basin_id_list:
            basin_id_list.append(basin_id)

    bps_logger.info(f"Input stack geographic location is in Basins {', '.join(basin_id_list)}")

    return basin_id_list


def _prepare_basin_id_lut():
    """
    Internal function.

    This function is used to construct a LookUp Table of Basin ids for each latitude/longitude of the World, with a resolution of 1 degree

    The LookUp table is then saved to txt, which is used to fill the content of bps.l2a_processor.basins.BASINS basin_ids tuple

    Implementation details
    For South America, Africa and Asia tropical/temperate region, this function bases on Tropic of Cancer and Tropic of Capricorn latitudes:
        Africa between Tropic of Cancer and Capricorn latitudes is tropical, otherwise Excvlusion zone
        Asia for latitudes over Tropic of Cancer is temperate, otherwise tropical
        South America for latitudes over the Tropic of Capricorn is tropical, otherwise temperate

    For the computation, to know in which country/continent the input STA product lies, we base on the
    Natural Earth Country ShapeFile:
        Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.
        1:110m Cultural Vectors
        Admin 0 - Countries

    Return
    ------
    The txt is saved in a single line with 360*180 = 64800 integers, as:
        [latitude1:all longitudes, latitude2: all longitudes, ... ]
    """

    natural_earth_shp_path = Path(
        r"C:\Users\emanuele.giorgi\Downloads\ne_110m_admin_0_countries_lakes\ne_110m_admin_0_countries_lakes.shp"
    )

    lon_indices = np.arange(360)
    lat_indices = np.arange(180)
    basin_ids_lut = np.zeros((360, 180), dtype=np.uint8)

    # Read the input Natural Earth Shapefile to get the coordinates of each country in the world
    sph = ogr.Open(str(natural_earth_shp_path))
    sph_layer = sph.GetLayer()

    # DEBUG Natural Earth Shapefile content:
    # all_continents = []
    # for feature in sph_layer:
    #     attributes = feature.items()
    #     continent = attributes["CONTINENT"]
    #     if not continent in all_continents:
    #         all_continents.append(continent)
    # bps_logger.info(
    #     f"Natural Earth Shapefile contains {len(all_continents)} continents: {all_continents}"
    # )

    # Cycle over feature in the SPH file: each feature is a world country
    counterN = len(sph_layer)
    for layer_idx, country_sph in enumerate(sph_layer):
        print(f"Filling basin_ids_lut, cycling sph layer {layer_idx + 1} of {counterN}")
        attributes = country_sph.items()

        # continent of the country
        continent = attributes["CONTINENT"]
        country = attributes["NAME"]
        print(f"We are in {country}, {continent}")
        if country == "Russia":
            continent = "Asia"

        if continent in [
            "Antarctica",
            "Europe",
            "Seven seas (open ocean)",
            "North America",
        ]:
            continue

        # Lat Lon footprints of the country:
        # each contry in the SPH file can have one or more footprints
        country_swath_coordinates = _sph_geometry_to_lat_lon(country_sph.GetGeometryRef())

        for lon_idx in lon_indices:
            for lat_idx in lat_indices:
                if basin_ids_lut[lon_idx, lat_idx] == 0:
                    is_in_this_country, data_min_latitude, data_max_latitude = _basin_check_intersection(
                        country_swath_coordinates,
                        lon_idx,
                        lat_idx,
                    )
                    if is_in_this_country:
                        # If STA product is in this country, create the basin_id and exit
                        # Exclusion zone: 700
                        # basin_id = 7
                        basin_id = None

                        if continent == "Oceania":
                            # Oceania tropical/temperate: 600
                            basin_id = 6

                        if continent == "Africa":
                            # Africa tropical/temperate: 300
                            basin_id = 3

                        if continent == "Asia":
                            if data_min_latitude > TROPIC_OF_CANCER_LATITUDE:
                                # Asia temperate: 400
                                basin_id = 4
                            else:
                                # Asia tropical: 500
                                basin_id = 5

                        if continent == "South America":
                            # South America tropical: 100
                            # South America temperate: 200
                            if data_max_latitude < TROPIC_OF_CAPRICORN_LATITUDE:
                                # South America temperate: 200
                                basin_id = 2
                            else:
                                # South America tropical: 100
                                basin_id = 1
                        if basin_id is not None:
                            basin_ids_lut[lon_idx, lat_idx] = basin_id

    basin_ids_lut[basin_ids_lut == 0] = 7

    with open(r"C:\ARESYS_PROJ\bps-test-plan\basin_ids_lut.txt", "ab") as f:
        for lat_idx in range(basin_ids_lut.shape[1]):
            np.savetxt(f, basin_ids_lut[:, lat_idx], fmt="%i", newline=",")
    np.save(r"C:\ARESYS_PROJ\bps-test-plan\basin_ids_lut.npy", basin_ids_lut)


def fast_read_mission_phase_id(
    input_stack_acquisitions: tuple[Path],
) -> str:
    """read mission phase id from stack primary product

    The function is very fast by avoiding Stack Product Reader and
    by reading directly the xml fields needed

    Parameters
    ----------
    input_stack_acquisitions : tuple[Path]
        Path of each input stack acquisition

    Returns
    -------
    mission_phase_id: str
        String identifying the mission phase, among INT, TOM, or COM
        depending on the number of input_stack_acquisitons_path
    """

    # Read each main annotation coregistered file, until the primary one is found
    # then get the mission phase ID from that one
    # If primary image is not present, use one of the other xml to get the ID
    mission_phase_id = None
    for acq_path in input_stack_acquisitions:
        annot_coregistered_path = acq_path.joinpath(
            "annotation_coregistered", acq_path.name[0:-10].lower() + "_annot.xml"
        )

        root = ET.parse(annot_coregistered_path).getroot()

        xpath_primaryImage_posList = ".//staCoregistrationParameters/primaryImage"
        primaryImage_node = root.find(xpath_primaryImage_posList)
        assert primaryImage_node is not None
        primaryImage_node.text

        xpath_secondaryImage_posList = ".//staCoregistrationParameters/secondaryImage"
        secondaryImage_node = root.find(xpath_secondaryImage_posList)
        assert secondaryImage_node is not None
        secondaryImage_node.text

        xpath_missionPhaseID_posList = ".//acquisitionInformation/missionPhaseID"
        missionPhaseID_node = root.find(xpath_missionPhaseID_posList)
        assert missionPhaseID_node is not None
        if primaryImage_node.text == secondaryImage_node.text:
            mission_phase_id = missionPhaseID_node.text
            break

    if mission_phase_id is None:
        mission_phase_id = missionPhaseID_node.text

    return mission_phase_id
