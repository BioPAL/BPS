# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2a common functionalities
--------------------------
"""

import concurrent.futures
import multiprocessing
from datetime import datetime
from pathlib import Path

import cv2
import numba as nb
import numpy as np
import numpy.typing as npt
from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.ellipsoid import WGS84
from arepytools.geometry.generalsarorbit import GeneralSarOrbit, create_general_sar_orbit
from arepytools.io.metadata import StateVectors
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.fnf_utils import FnFMask
from bps.l2a_processor.core.aux_pp2_2a import GeneralConf
from bps.l2a_processor.io.aux_pp2_2a_models.models import CalibrationScreenType
from bps.transcoder.sarproduct.biomass_l2aproduct_writer import COMPRESSION_SCHEMA_MDS_ZSTD, INT_NODATA_VALUE
from bps.transcoder.sarproduct.biomass_stackproduct import BIOMASSStackProduct
from bps.transcoder.utils.time_conversions import pdt_to_compact_date
from osgeo import gdal, ogr
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay

LIGHTSPEED = 299792458


class InvalidStackAcquisitions(ValueError):
    """Raised when failing to find at least two acquisitions with usable baseline, from input stack"""


def interpolate_fnf(
    fnf: np.ndarray,
    fnf_latitude_axis: np.ndarray,
    fnf_longitude_axis: np.ndarray,
    forest_mask_interpolation_threshold: float,
    l2a_latitude_axis: np.ndarray,
    l2a_longitude_axis: np.ndarray,
) -> dict:
    """Interpolate the input FNF:
        FNF is precisely cut over the current L2a product coverage and interpoled over the DGG grid sampling

    Parameters
    ----------
    fnf: np.ndarray
        input FNF
    fnf_latitude_axis: np.ndarray
        input FNF latitude axis, in degrees
    fnf_longitude_axis: np.ndarray
        input FNF longitude axis, in degrees
    forest_mask_interpolation_threshold: float
        configuration from AUX PP2 2A
        threshold to fix rounding of pixels with decimal values originated from binary FNF interpolation onto L2a grid.
        This creates a safety buffer around forest border.
        Range of values from 0 to 1, default 0.5.
    l2a_latitude_axis: np.ndarray
        interpolation output latitude axis, in degrees: it is the L2A Product DGG latitude axis
    l2a_longitude_axis: np.ndarray
        interpolation output longitude axis, in degrees: it is the L2A Product DGG longitude axis

    Returns
    -------
    lut_fnf_dict: dict
        containing following keys:
        fnf: np.ndarray of dimensions latitude_vec x longitude_vec, interpolated FNF mask array
        latitude_vec: interpolated FNF latitude axis in degrees.
        longitude_vec: interpolated FNF longitude axis in degrees.
    """

    bps_logger.info(
        f"    interpolation of input FNF over DGG, using AUX PP2 2A forest mask interpolation threshold: {forest_mask_interpolation_threshold}"
    )

    # interpolation function: no data values filled with zeros

    fnf_interp = parallel_reinterpolate(
        [fnf],  # function works with lists
        fnf_latitude_axis,  # fnf orientation is our default
        fnf_longitude_axis,  # fnf orientation is our default
        l2a_latitude_axis,
        l2a_longitude_axis,
        fill_value=INT_NODATA_VALUE,
    )[0]  # function works with lists

    # thresholding edges
    fnf_interp[fnf_interp > (forest_mask_interpolation_threshold + 1)] = INT_NODATA_VALUE
    fnf_interp[
        np.logical_and(
            fnf_interp > forest_mask_interpolation_threshold,
            fnf_interp <= (forest_mask_interpolation_threshold + 1),
        )
    ] = 1.0
    fnf_interp[fnf_interp <= forest_mask_interpolation_threshold] = 0.0
    fnf_interp.astype(np.uint8)

    # writing to dict, after casting to uint8
    lut_fnf_dict = {
        "fnf": fnf_interp.astype(np.uint8),
        "latitude_vec": l2a_latitude_axis,
        "longitude_vec": l2a_longitude_axis,
    }
    return lut_fnf_dict


def forest_coverage_check(
    fnf: FnFMask,
    forest_coverage_threshold: float,
    footprints_list: list[list[float]],
) -> tuple[bool, float]:
    """Check the forest coverage to and compare with a threshold, to decide if start or skip computation

    Parameters
    ----------
    fnf: FnFMask,
        forest-non-forest mask object, containing fnf itself and axis definitions
    forest_coverage_threshold: float
        configuration from AUX P2 2A XML file;
        threshold of forest coverage, in percentage, to trigger L2A processing;
        0.0%: considered a special value, used to skip forest coverage check and returns do_computation = True

    footprint: List[float]
        footprint of the data to be processed, in degrees
         [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon]


    Returns
    -------
    do_computation: bool
        flag to trigger or not the L2A processor
        do_computation = True: triggers the processor
        do_computation = False: ends the processor
    forest_coverage_percentage: float
        percentage of forest coverage over the footprint
    """

    start_time = datetime.now()
    if not forest_coverage_threshold:
        # Special value, if threshold is zero, do not consider it and do computation
        bps_logger.info("Found AUX PP2 2A forest coverage threshold = 0: optional step 'forest coverage check' ignored")

    else:
        bps_logger.info("Forest coverage check: compute forest coverage over the footprint, basing on input FNF")

    # compute mask_footprint:
    # this mask has fnf dimensions and true values only inside the footprint
    latitude_n = len(fnf.lat_axis)
    longitude_n = len(fnf.lon_axis)
    n_pixels_forested_area = 0
    n_pixels_total_area = 0
    mask_footprint_compound = np.zeros((latitude_n, longitude_n)).astype(np.uint8)
    for footprint in footprints_list:
        mask_footprint = np.zeros((latitude_n, longitude_n)).astype(np.uint8)

        # Generate the mask
        # dgg_footprint: [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon]
        ne_lat_idx = np.where(np.sort(fnf.lat_axis) <= footprint[0])[0][-1]
        ne_lon_idx = np.where(np.sort(fnf.lon_axis) <= footprint[1])[0][-1]
        se_lat_idx = np.where(np.sort(fnf.lat_axis) <= footprint[2])[0][-1]
        se_lon_idx = np.where(np.sort(fnf.lon_axis) <= footprint[3])[0][-1]
        sw_lat_idx = np.where(np.sort(fnf.lat_axis) <= footprint[4])[0][-1]
        sw_lon_idx = np.where(np.sort(fnf.lon_axis) <= footprint[5])[0][-1]
        nw_lat_idx = np.where(np.sort(fnf.lat_axis) <= footprint[6])[0][-1]
        nw_lon_idx = np.where(np.sort(fnf.lon_axis) <= footprint[7])[0][-1]

        my_roi = [
            (nw_lon_idx, nw_lat_idx),
            (ne_lon_idx, ne_lat_idx),
            (se_lon_idx, se_lat_idx),
            (sw_lon_idx, sw_lat_idx),
        ]  #  1
        cv2.fillPoly(mask_footprint, [np.array(my_roi)], 1)
        mask_footprint_compound = np.logical_or(mask_footprint, mask_footprint_compound)

        # the sum of true values in mask_footprint is the total area in number of pixels
        # n_pixels_total_area += np.sum(mask_footprint)

        # the forested area is the sum of true values in input fnf, only inside the mask_footprint area
        # fnf_dict["fnf"][np.logical_not(mask_footprint)] = np.uint8(0)
        # n_pixels_forested_area += np.sum(fnf_dict["fnf"])

    fnf.mask[np.logical_not(mask_footprint_compound)] = np.uint8(0)
    n_pixels_total_area = np.sum(mask_footprint_compound)
    n_pixels_forested_area = np.sum(fnf.mask)
    fnf.mask[np.logical_not(mask_footprint)] = INT_NODATA_VALUE
    forest_coverage_percentage = float(100.0 * n_pixels_forested_area / n_pixels_total_area)

    bps_logger.info(
        f"   forest coverage of input stack is {forest_coverage_percentage:2.1f} %; threshold from AUX PP2 2A is {forest_coverage_threshold}% :"
    )

    if forest_coverage_threshold:
        # standard case, compare threshold with percentage of forest coverage
        do_computation = forest_coverage_percentage > forest_coverage_threshold

        if do_computation:
            bps_logger.info("       forest coverage is enough for L2A processing")

        else:
            bps_logger.warning(
                f"       forest coverage of input stack is less than {forest_coverage_threshold} %. L2A processing skipped"
            )
    else:
        # special case, threshold is zero, always do computation
        do_computation = True

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Forest coverage check processing time: {elapsed_time:2.3f} s")
    return do_computation, forest_coverage_percentage


def int_subsetting(
    input_stack_mph_files: list[Path],
    input_stack_acquisitons_path: list[Path],
    subsetting_rule: str,
    mission_phase_id: str,
) -> tuple[list[int], list[Path]]:
    """In case of TOM phase (from a minumum of #4 to a maximum of #8 acquisitions),
    select the #3 acquisitions to operate as in INT phase.

    Parameters
    ----------
    input_stack_mph_files: List[Path]
        list of path for each stack acquisition MPH header file
    subsetting_rule: str
    mission_phase_id: str
        String identifying the mission phase, among INT, TOM, COM, used only for logging

    Returns
    -------
    selected_baselines_indices. List[int]
        those are the zero-based indices of the selected baselines
    input_stack_mph_files_sub: List[Path]
        MPH files path of the selected baselines
    """

    start_time = datetime.now()
    number_of_acquisitions = len(input_stack_mph_files)
    # Get from main annotation xml, all the baselines indexes present
    current_baseline_indexes = []
    for acquisition_path in input_stack_acquisitons_path:
        current_baseline_indexes.append(fast_read_baseline_ordering_index(acquisition_path))
    # Lengths <2 or >8 are already checked in job order parser function.
    if number_of_acquisitions > 3:
        bps_logger.info(f"L2a {mission_phase_id} phase processing (with a stack of #{number_of_acquisitions} images)")
        bps_logger.info("Acquisitions sub setting:")
        bps_logger.info(f"    using AUX PP2 2A subsetting rule: {subsetting_rule}")

        # order the found baselines
        current_baseline_indexes = np.sort(current_baseline_indexes)

        # crescent index of each theoretical baseline present
        all_baselines_indexes = np.arange(7) if number_of_acquisitions < 8 else np.arange(8)

        # prepare a boolean array telling if the theoretical baselines are present in the current stack
        input_baseline_is_present = np.zeros(len(all_baselines_indexes), dtype=bool)
        for all_bsl_idx in all_baselines_indexes:
            if all_bsl_idx in current_baseline_indexes:
                input_baseline_is_present[all_bsl_idx] = True

        bps_logger.info(
            f"    available acquisitions (zero-based ordered baseline indices) are: {current_baseline_indexes}"
        )
        assert subsetting_rule in [el.value for el in GeneralConf.SubsettingRules]
        if subsetting_rule == GeneralConf.SubsettingRules.GEOMETRY.value:
            rule_indexes = _int_subsetting_geometry_rule(
                number_of_acquisitions,
                input_baseline_is_present,
            )

            # This is the indices in the input paths ordering, selected by the algorithm
            temp = np.sort(
                [np.where(current_baseline_indexes == current_rule_idx)[0][0] for current_rule_idx in rule_indexes]
            )
            selected_baselines_indices = current_baseline_indexes[temp]

        elif subsetting_rule == GeneralConf.SubsettingRules.MAINTAIN_ALL.value:
            selected_baselines_indices = current_baseline_indexes

        if subsetting_rule == GeneralConf.SubsettingRules.GEOMETRY.value:
            bps_logger.info(
                f"        selected acquisitions (zero-based ordered baseline indices): {selected_baselines_indices}, "
            )
        elif subsetting_rule == GeneralConf.SubsettingRules.MAINTAIN_ALL.value:
            bps_logger.info("        selected all the available acquisitions")

        # select correct MPH files:
        input_stack_mph_files_sub = []
        for input_stack_mph_file in input_stack_mph_files:
            baseline_ordering_index = fast_read_baseline_ordering_index(input_stack_mph_file.parent)
            if baseline_ordering_index in selected_baselines_indices:
                input_stack_mph_files_sub.append(input_stack_mph_file)

        stop_time = datetime.now()
        elapsed_time = (stop_time - start_time).total_seconds()
        bps_logger.info(f"Subsetting processing time: {elapsed_time:2.1f} s")

    else:
        input_stack_mph_files_sub = input_stack_mph_files
        selected_baselines_indices = current_baseline_indexes

        if number_of_acquisitions == 2:
            bps_logger.info(
                f"L2a {mission_phase_id} phase processing (with an incomplete stack of only #{number_of_acquisitions} images)"
            )
        else:
            # 3 images
            bps_logger.info(
                f"L2a {mission_phase_id} phase processing (with a stack of #{number_of_acquisitions} images)"
            )

    return selected_baselines_indices, input_stack_mph_files_sub


def _int_subsetting_geometry_rule(number_of_acquisitions: int, input_baseline_is_present: np.ndarray) -> np.ndarray:
    # Rules, to be evaluated in crescent order:
    rule_1 = np.array([1, 3, 5]).astype(int)
    rule_2 = np.array([0, 2, 4]).astype(int)
    rule_3 = np.array([2, 4, 6]).astype(int)
    rule_4 = np.array([3, 5, 7]).astype(int)  # contingency case rule

    # First try if a valid combination of #3 exists, with original rules:
    if input_baseline_is_present[rule_1].all():
        selected_acquisitions = rule_1
        bps_logger.info("    found three acquisitions from default evaluation order #1")
    elif input_baseline_is_present[rule_2].all():
        selected_acquisitions = rule_2
        bps_logger.info("    found three acquisitions from evaluation order #2")
    elif input_baseline_is_present[rule_3].all():
        selected_acquisitions = rule_3
        bps_logger.info("    found three acquisitions from evaluation order #3")
    elif number_of_acquisitions == 8 and input_baseline_is_present[rule_4].all():
        selected_acquisitions = rule_4
        bps_logger.info("    found three acquisitions from evaluation order #4 (TOM contingency rule)")
    else:
        # Than try if a valid combination of #2 exists, from original rules:
        indices_rule_1 = np.where(input_baseline_is_present[rule_1])[0]
        indices_rule_2 = np.where(input_baseline_is_present[rule_2])[0]
        indices_rule_3 = np.where(input_baseline_is_present[rule_3])[0]
        if number_of_acquisitions == 8:
            indices_rule_4 = np.where(input_baseline_is_present[rule_4])[0]

        if (
            len(indices_rule_1) > 1  # at least two from the selected rule
            and indices_rule_1[1] - indices_rule_1[0] < 3  # max distance double TOM spacing
        ):
            selected_acquisitions = rule_1[indices_rule_1]
            bps_logger.info("    found two acquisitions from default evaluation order #1 (one baseline is missing)")

        elif len(indices_rule_2) > 1 and indices_rule_2[1] - indices_rule_2[0] < 3:
            selected_acquisitions = rule_2[indices_rule_2]
            bps_logger.info("    found two acquisitions from evaluation order #2 (one baseline is missing)")

        elif len(indices_rule_3) > 1 and indices_rule_3[1] - indices_rule_3[0] < 3:
            selected_acquisitions = rule_1[indices_rule_3]
            bps_logger.info("    found two acquisitions from evaluation order #3 (one baseline is missing)")

        elif number_of_acquisitions == 8 and len(indices_rule_4) > 1 and indices_rule_4[1] - indices_rule_4[0] < 3:
            selected_acquisitions = rule_1[indices_rule_4]
            bps_logger.info(
                "    found two acquisitions from evaluation order #4 (TOM contingency rule, with one baseline missing)"
            )

        else:
            # Eventually try any combinatiof of two, out of the rules:
            indices_last_try = np.where(input_baseline_is_present)[0]
            found_rule = False
            if len(indices_last_try) > 1:
                for idx in np.arange(len(indices_last_try) - 1):
                    idx_next = idx + 1
                    if indices_last_try[idx_next] - indices_last_try[idx] < 3:  # max distance double TOM spacing
                        selected_acquisitions = indices_last_try[0:2]
                        bps_logger.info(
                            "    No official evaluation order rule worked; found two acquisitions compliant with baseline spacing"
                        )
                        found_rule = True
                        break
            if not found_rule:
                error_msg = "    Cannot find a minimum of #2 acquisitions with a maximum baseline equivalent to double TOM spacing"
                raise InvalidStackAcquisitions(error_msg)

    return selected_acquisitions


def calibration(
    scs_acq_list: list[list[np.ndarray]],
    scs_axis_sr_s: np.ndarray,
    scs_axis_az_s: np.ndarray,
    skp_gnd_phase_screens_rad: list[np.ndarray],
    geometry_gnd_phase_screens_rad: list[np.ndarray],
    screen_axis_sr_s: np.ndarray,
    screen_axis_az_s: np.ndarray,
    apply_calibration_screen: str,
) -> list[list[np.ndarray]]:
    """Calibration: groundscreen application
    The output list of lists is reshaped also if apply_calibration_screen is "none"
    the reshape ease the further processing:
        see parameter "scs_acq_list" vs returned "scs_pol_list"

    Parameters
    ----------
    scs_acq_list: List[List[np.ndarray]]
        scs data to be calibrated,
            list with M=2 or 3 acquisitions,
            each containing a list with P=3 polarizations (in the order HH, XP, VV),
            each of dimensions [N_az x N_rg]
    scs_axis_sr_s: np.ndarray
        slant range time axis of scs data in scs_acq_list [s]
    scs_axis_az_s: np.ndarray
        azimuth time axis of scs data in scs_acq_list[s]
    gnd_phase_screens_rad: List[np.ndarray]
        ground phase screens [rad]
        M, each [N(az,φ) N(rg,φ) ]
    screen_axis_sr_s: np.ndarray
        slant range time axis of ground phase screen [s]
    screen_axis_az_s: np.ndarray
        azimuth time axis of ground phase screen [s]
    apply_calibration_screen: str
        "none”: no phase screen is applied
        "geometry”: only flattening phase screen is applied (i.e., as computed from acquisition geometry)
        "skp”: complete phase screen is applied (default)


    Returns
    -------
    scs_pol_list: List[List[np.ndarray]]
        calibrated scs data:
            list with P=3 polarizations (in the order HH, XX, VV),
            each containing a list with M=2 or 3 acquisitions,
            each of dimensions [N_az x N_rg]

    """
    start_time = datetime.now()
    # output data will be saved in inverse list order respect to scs_acq_list
    #  scs_acq_list list of acqiisitions containing polarizations
    #  output list: list of polarizations containing acqiisitions

    num_pols = len(scs_acq_list[0])  # get from fists acquisition
    scs_pol_list = []

    if apply_calibration_screen != CalibrationScreenType.NONE.value:
        bps_logger.info(
            f"Compute input stack acquisitions calibration using AUX PP configuration '{apply_calibration_screen}'"
        )

        num_acq = len(scs_acq_list)
        if num_pols == 3:
            pol_names_for_log = ["HH", "XP", "VV"]  # just needee for logging reasons
        if num_pols == 4:
            pol_names_for_log = [
                "HH",
                "HV",
                "VH",
                "VV",
            ]  # just needee for logging reasons

        # interpolate ground phase screens over SCS data axis, if needed
        if not scs_acq_list[0][0].shape == geometry_gnd_phase_screens_rad[0].shape:
            bps_logger.info("    preliminary interpolation to bring ground phase screens onto the same L1c grid")
            # interpolation:
            # the geometry_gnd_phase_screens_rad is assumed unwrapped to be interpolated as a float real-valued field.
            # the skp_gnd_phase_screens_rad component is not unwrapped, interpolation must be performed separately for each field after unwrapping if useful.

            geometry_sum_nan_before = np.sum(np.isnan(geometry_gnd_phase_screens_rad))
            # DSI
            gnd_phase_screens_exp = parallel_build_screen_exp(
                geometry_gnd_phase_screens_rad,
                screen_axis_az_s,
                screen_axis_sr_s,
                scs_axis_az_s,
                scs_axis_sr_s,
            )
            geometry_sum_nan_after = np.sum(np.isnan(gnd_phase_screens_exp))
            if geometry_sum_nan_after > 0:
                bps_logger.warning(
                    f"    invalid samples before / after Geometry LUT interpolation: {geometry_sum_nan_before / (geometry_gnd_phase_screens_rad[0].size * len(geometry_gnd_phase_screens_rad)) * 100:2.3f}% / {geometry_sum_nan_after / (gnd_phase_screens_exp[0].size * len(gnd_phase_screens_exp)) * 100:2.3f}%"
                )

            if apply_calibration_screen == CalibrationScreenType.SKP.value:
                # SKP
                skp_sum_nan_before = np.sum(np.isnan(geometry_gnd_phase_screens_rad))
                skp_gnd_phase_screens_exp = parallel_build_screen_exp_sin_cos(
                    skp_gnd_phase_screens_rad,
                    screen_axis_az_s,
                    screen_axis_sr_s,
                    scs_axis_az_s,
                    scs_axis_sr_s,
                )
                skp_sum_nan_after = np.sum(np.isnan(skp_gnd_phase_screens_exp))
                if geometry_sum_nan_after > 0:
                    bps_logger.warning(
                        f"    invalid samples before / after SKP LUT interpolation: {skp_sum_nan_before / (skp_gnd_phase_screens_rad[0].size * len(skp_gnd_phase_screens_rad)) * 100:2.3f}% / {skp_sum_nan_after / (skp_gnd_phase_screens_exp[0].size * len(skp_gnd_phase_screens_exp)) * 100:2.3f}%"
                    )
                # Sum togheter SKP + DSP (multiplication because they are exponentials already)
                gnd_phase_screens_exp = [
                    gnd_exp * skp_exp for gnd_exp, skp_exp in zip(gnd_phase_screens_exp, skp_gnd_phase_screens_exp)
                ]

        else:
            gnd_phase_screens_exp = [np.exp(1j * gps).astype(np.complex64) for gps in geometry_gnd_phase_screens_rad]

            if apply_calibration_screen == CalibrationScreenType.SKP.value:
                skp_gnd_phase_screens_exp = [np.exp(1j * gps).astype(np.complex64) for gps in skp_gnd_phase_screens_rad]
                # Sum togheter SKP + DSP (multiplication because they are exponentials already)
                gnd_phase_screens_exp = [
                    gnd_exp * skp_exp for gnd_exp, skp_exp in zip(gnd_phase_screens_exp, skp_gnd_phase_screens_exp)
                ]

        # output data will be saved in inverse list order (list of pols containing acq and not vice versa)
        for pol_idx in range(num_pols):
            # fill the polarization list with M acquisitions
            bps_logger.info(f"    calibration of {num_acq} acquisitions for polarization {pol_names_for_log[pol_idx]}")
            scs_pol_list.append(
                [
                    acquisition_pols[pol_idx] * gnd_phase_screen_exp
                    for acquisition_pols, gnd_phase_screen_exp in zip(scs_acq_list, gnd_phase_screens_exp)
                ]
            )
        stop_time = datetime.now()
        elapsed_time = (stop_time - start_time).total_seconds()
        bps_logger.info(f"Calibration processing time: {elapsed_time:2.1f} s")

    else:
        bps_logger.info("Input stack acquisitions calibration disabled from AUX PP2 2A configuration.")
        for pol_idx in range(num_pols):
            # fill the polarization list with M acquisitions
            scs_pol_list.append([acquisition_pols[pol_idx] for acquisition_pols in scs_acq_list])

    return scs_pol_list


def mpmb_covariance_estimation(
    input_data_list: list[list[np.ndarray]],
    fa_normalized: np.ndarray,
    num_az_subsampled: int,
    fr_normalized_transposed: np.ndarray,
    num_rg_subsampled: int,
) -> np.ndarray:
    """Multi-Polarimetric Multi-Baseline covariance estimation

    Parameters
    ----------
    input_data_list: List[List[np.ndarray]]
        For FD it is the ground cancelled data
        For FH it is the SCS input data
        In both cases, the input_data_list is a
            list of P=3 polarizations (in the order HH, XP, VV),
            each containing a list of M acquisitions (M=1 for FD, M=2 or 3 for FH),
            each of dimensions [N_az x N_rg]
        Note: it can contain also only 2 polarization, in this case, the list is still of P=3, but one element is None
    fa_normalized: np.ndarray
        Azimuth normalized sparse matrix (see build_filtering_sparse_matrices)
    num_az_subsampled: int
        number of azimuth samples of the dacimated output mpmb_covariance data
    fr_normalized_transposed: np.ndarray
         Slant range normalized sparse matrix (see build_filtering_sparse_matrices)
    num_rg_subsampled: int
        number of slant range samples of the dacimated output mpmb_covariance data

    Returns
    -------
    mpmb_covariance: np.ndarray
        Multi-Polarimetric -> Multi-Baseline estimated covariance matrix:
        it is as a PxP matrix, where each element is a MxM matrix
        So, total mensions are [(P x M) x (P x M) x num_az_subsampled x num_rg_subsampled]
        Subsampled respect input_data_list [N_az x N_rg] dimensions.
    """

    # Inputs computation
    num_pols = len(input_data_list)
    num_imms = len(input_data_list[0])
    num_az_in, num_rg_in = input_data_list[0][0].shape

    # output initialization
    mpmb_covariance = np.zeros(
        (
            num_pols * num_imms,
            num_pols * num_imms,
            num_az_subsampled,
            num_rg_subsampled,
        ),
        dtype=np.complex64,
    )

    # masking for invalid values
    nodata_values_mask = np.zeros((num_az_in, num_rg_in, num_imms), dtype=bool)
    for pol_idx, acq_list in enumerate(input_data_list):
        all_acq_data = np.zeros((num_az_in, num_rg_in, num_imms), dtype=complex)
        for acq_idx, image in enumerate(acq_list):
            all_acq_data[:, :, acq_idx] = image
        nodata_values_mask = np.logical_or(nodata_values_mask, np.isnan(all_acq_data))

    for pol_idx, acq_list in enumerate(input_data_list):
        for acq_idx, image in enumerate(acq_list):
            input_data_list[pol_idx][acq_idx][nodata_values_mask[:, :, acq_idx]] = 0

    # compute polarimetrix covariance matrix
    bps_logger.info("    Polarimetrc covariance matrix computation:")
    for ch_p, pol_idx_p in enumerate(range(num_pols)):
        bps_logger.info(f"        polarization {ch_p + 1} of {num_pols}")
        ind_p = np.arange(num_imms) + ch_p * num_imms
        for ch_q in np.arange(ch_p, num_pols):
            pol_idx_q = ch_q
            ind_q = np.arange(num_imms) + ch_q * num_imms
            II = np.zeros(
                (num_imms, num_imms, num_az_subsampled, num_rg_subsampled),
                dtype=np.complex64,
            )
            for n_idx in range(num_imms):
                if ch_p == ch_q:
                    m_min = n_idx
                else:
                    m_min = 0

                for m_idx in np.arange(m_min, num_imms):
                    temp = input_data_list[pol_idx_p][n_idx].astype("complex128") * np.conjugate(
                        input_data_list[pol_idx_q][m_idx].astype("complex128")
                    )
                    temp = fa_normalized @ temp
                    temp = temp @ fr_normalized_transposed
                    II[n_idx, m_idx, :, :] = temp.astype("complex64")
            mpmb_covariance[ind_p[:, np.newaxis], ind_q[np.newaxis, :], :, :] = II

    # Symmetric part generation
    diag_mask = np.tile(
        (np.eye(num_pols * num_imms).reshape(num_pols * num_imms, num_pols * num_imms, 1, 1)) > 0,
        [1, 1, num_az_subsampled, num_rg_subsampled],
    )
    mpmb_covariance = mpmb_covariance + np.conjugate(np.moveaxis(mpmb_covariance, [1, 0, 2, 3], [0, 1, 2, 3]))
    mpmb_covariance[diag_mask] = mpmb_covariance[diag_mask] / 2

    bps_logger.info(
        f"    data shape after decimation: Azimuth {mpmb_covariance.shape[2]} samples, Slant-range {mpmb_covariance.shape[3]} samples"
    )

    return mpmb_covariance


def _build_filtering_matrix(
    num_rows: int, average_window_size: int, decimation_factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Internal function to build a sparse matrix
    to carry out one-dimensional moving average.
    It deals with regularly sampled data.

    Parameters
    ----------
    num_rows: int
        number of rows of the matrix to be filtered
    average_window_size: int
        one-sided length of the average window
    decimation_factor: int
        subsampling factor of the filtered signal

    Returns
    -------
    sparse_filtering_matrix: np.ndarray
    filtered_signal_axis: np.ndarray
    normalizing_matrix:  np.ndarray
    """

    xin = np.arange(num_rows)
    Nxin = xin.size
    filtered_signal_axis = np.arange(0, num_rows, decimation_factor)
    Nxout = filtered_signal_axis.size

    average_window_size_used = average_window_size - 1
    col = np.kron(filtered_signal_axis, np.ones((1, 2 * average_window_size_used + 1))) + np.kron(
        np.ones((1, Nxout)),
        np.arange(-average_window_size_used, average_window_size_used + 1),
    )
    row = np.kron(np.arange(Nxout), np.ones((1, 2 * average_window_size_used + 1)))

    ok_mask = (col >= 0) * (col < Nxin)
    Nok_mask = np.sum(ok_mask)

    sparse_filtering_matrix = csr_matrix(
        (np.ones((1, Nok_mask)).flatten(), (row[ok_mask], col[ok_mask])),
        shape=(Nxout, Nxin),
    )

    normalizing_matrix = csr_matrix(
        (
            1 / np.sum(sparse_filtering_matrix.toarray(), axis=1),
            (np.arange(Nxout), np.arange(Nxout)),
        )
    )

    return sparse_filtering_matrix, filtered_signal_axis, normalizing_matrix


def build_filtering_sparse_matrices(
    num_az_in: int,
    num_rg_in: int,
    averaging_window_size_rg: int,
    decimation_factor_rg: int,
    averaging_window_size_az: int,
    decimation_factor_az: int,
) -> tuple[np.ndarray, int, np.ndarray, int]:
    """
    Compute sparse filtering matrices in slant range and azimuth

    Parameters
    ----------
    num_az_in: int
        Input azimuth matrix dimension
    num_rg_in: int
        Input slant range matrix dimension
    averaging_window_size_az: int
        Averaging windows size in azimuth direction
    decimation_factor_rg: int
        Slant Range decimation factor for the multi look computed from AUX PP input upsampling factor
    averaging_window_size_rg: int
        Averaging windows size in slant range direction
    decimation_factor_az: int
        Azimuth decimation factor for the multi look computed from AUX PP input upsampling factor

    Returns
    -------
    fa_normalized: np.ndarray
        Azimuth normalized sparse matrix (see build_filtering_sparse_matrices)
    fr_normalized_transposed: np.ndarray
         Slant range normalized sparse matrix (see build_filtering_sparse_matrices)
    axis_az_subsampling_indexes: int
        indices for the original azimuth axis, to obtain the decimated output mpmb_covariance data axis
    axis_rg_subsampling_indexes: int
        indices for the original slant range axis, to obtain the decimated output mpmb_covariance data axis
    """

    # sparse_filtering_matrix, filtered_signal_axis, normalizing_matrix
    (
        sparse_filtering_matrix_rg,
        axis_rg_subsampling_indexes,
        normalizing_matrix_rg,
    ) = _build_filtering_matrix(num_rg_in, averaging_window_size_rg, decimation_factor_rg)
    fr_normalized_transposed = (normalizing_matrix_rg @ sparse_filtering_matrix_rg).T

    # Azimuth filter matrix
    (
        sparse_filtering_matrix_az,
        axis_az_subsampling_indexes,
        normalizing_matrix_az,
    ) = _build_filtering_matrix(num_az_in, averaging_window_size_az, decimation_factor_az)
    fa_normalized = normalizing_matrix_az @ sparse_filtering_matrix_az

    return (
        fa_normalized,
        axis_az_subsampling_indexes,
        fr_normalized_transposed,
        axis_rg_subsampling_indexes,
    )


def get_dgg_sampling(
    latlon_coverage: list[float],
    dgg_sampling_dict: dict,
    invert_latitude: bool,
    invert_longitude: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Get DGG Sampling
    Get the DGG sampling parameters at the correct latitude region
    and the latitude, longitude axes covering the input data footprint.

    Parameters
    ----------
    latlon_coverage: List[float]
        latitude/Longitude coverage, in degrees
         [lat_min, lat_max, lon_min, lon_max]
    dgg_sampling: dict
        Dictionary containing all the possible DGG parameters, for a 1x1 deg DGG tile at each latitude region
        dgg_sampling["0-50"]
        dgg_sampling["50-60"]
        dgg_sampling["60-70"]
        each containing a sub dict with: latitude_spacing, longitude_spacing, n_lat , n_lon
        See create_dgg_sampling_dict function for details.
    invert_latitude: bool
        Flag used for the orientation of latitude axis
        Faslse if south to north
        True if north to south
    invert_longitude: bool
        Flag used  for the orientation of longitude axis
        False if west to east
        True if east to west

    Returns
    -------
    dgg_latitude_axis_deg: np.ndarray
        Whole latitude vector [deg] in DGG sampling, covering the input footprint region
    dgg_longitude_axis_deg: np.ndarray
        Whole longitude vector [deg] in DGG sampling, covering the input footprint region
    """

    bps_logger.info("Geocoding initialization: get DGG sampling")

    lat_min = latlon_coverage[0]
    lat_max = latlon_coverage[1]
    lon_min = latlon_coverage[2]
    lon_max = latlon_coverage[3]

    dgg_sampling_extracted = {}
    if lat_min >= 60 or lat_max <= -60:
        # Data completely inside 60 to 70 band
        dgg_sampling_extracted = dgg_sampling_dict["60-70"]
        bps_logger.info(
            "    DGG sampling set to parameters of additional latitudinal band, between  60° and 70° N and S"
        )
    elif (lat_min < 60 and lat_max > 60) or (lat_max < -60 and lat_min > -60):
        # Data across the 60 latitude: setting to "60-70"
        dgg_sampling_extracted = dgg_sampling_dict["60-70"]
        bps_logger.info(
            "    DGG sampling set to parameters of additional latitudinal band, between  60° and 70° N and S"
        )
    elif lat_min >= 50 or lat_max <= -50:
        # Data completely inside 50 to 60 band
        dgg_sampling_extracted = dgg_sampling_dict["50-60"]
        bps_logger.info(
            "    DGG sampling set to parameters of additional latitudinal band, between  50° and 60° N and S"
        )
    elif (lat_min < 50 and lat_max > 50) or (lat_max < -50 and lat_min > -50):
        # Data across the 50 latitude: setting to "50-60"
        dgg_sampling_extracted = dgg_sampling_dict["50-60"]
        bps_logger.info(
            "    DGG sampling set to parameters of additional latitudinal band, between  60° and 70° N and S"
        )
    elif lat_min > -50 or lat_max < 50:
        dgg_sampling_extracted = dgg_sampling_dict["0-50"]
        bps_logger.info("    DGG sampling set to parameters of central latitudinal band, between 50°S and 50°N")

    # compute DGG: latitude and longitude vectors to cover the whole footprint

    # create a gap before and end, to avoid cropping final data during geocoding
    lat_min = lat_min - dgg_sampling_extracted["latitude_spacing_deg"] * 5
    lat_max = lat_max + dgg_sampling_extracted["latitude_spacing_deg"] * 5
    lon_min = lon_min - dgg_sampling_extracted["longitude_spacing_deg"] * 5
    lon_max = lon_max + dgg_sampling_extracted["longitude_spacing_deg"] * 5

    latitude_n = int((lat_max - lat_min) / dgg_sampling_extracted["latitude_spacing_deg"])
    dgg_latitude_axis_deg = lat_min + dgg_sampling_extracted["latitude_spacing_deg"] * np.arange(latitude_n).astype(
        np.float64
    )
    if invert_latitude:
        dgg_latitude_axis_deg = np.flip(dgg_latitude_axis_deg)

    longitude_n = int((lon_max - lon_min) / dgg_sampling_extracted["longitude_spacing_deg"])
    dgg_longitude_axis_deg = lon_min + dgg_sampling_extracted["latitude_spacing_deg"] * np.arange(longitude_n).astype(
        np.float64
    )
    if invert_longitude:
        dgg_longitude_axis_deg = np.flip(dgg_longitude_axis_deg)

    bps_logger.info(f"    DGG axis size: latitude {latitude_n} samples, longitude {longitude_n} samples")
    return dgg_latitude_axis_deg, dgg_longitude_axis_deg


def _geocoding_prepare_inputs(
    state_vectors: StateVectors,
    dem_height_m: np.ndarray,
    dem_latitude_deg: np.ndarray,
    dem_longitude_deg: np.ndarray,
    dem_axis_az_s: np.ndarray,
    dem_axis_sr_s: np.ndarray,
    scs_axis_az_s: np.ndarray,
    scs_axis_sr_s: np.ndarray,
) -> tuple[GeneralSarOrbit, np.ndarray, np.ndarray, np.ndarray]:
    """Geocoding initialization

    Prepare geocoding inputs:
    - interpolate DEM over the slant range, azimuth grid of the data to be geocoded
        (it is the subsampled, multilooked SCS grid)
    - initiate general sar orbit object
    Note: input DEM lat lon are in degrees, output ones are in radiants (to speed-up computations)

    Parameters
    ----------
    state_vectors: StateVectors
        State vectors from L1C orbit file [N_az,1]
    dem_height_m: np.ndarray
        DEM height [m], in slant-range, azimuth projection, from LUT [N_az_dem, N_rg_dem]
    dem_latitude_deg: np.ndarray
        DEM latitude [deg], in slant-range, azimuth projection, from LUT [N_az_dem, N_rg_dem]
    dem_longitude_deg: np.ndarray
        DEM longitude [deg], in slant-range, azimuth projection, from LUT [N_az_dem, N_rg_dem]
    dem_axis_az_s: np.ndarray
        Azimuth axis [s] of the DEM (slant-range azimith projection)  [N_az_dem,1]
    dem_axis_sr_s: np.ndarray
        Slant.range axis [s] of the DEM (slant-range azimith projection)  [N_rg_dem,1]
    scs_axis_az_s: np.ndarray
        Azimuth axis [s] of the L1C SCS, already subsampled by multilook step [N_az_sub]
    scs_axis_sr_s: np.ndarray
        Slant.range axis [s] of the L1C SCS, already subsampled by multilook step [N_rg_sub]

    Returns
    -------
    gso: GeneralSarOrbit
        General Sar Orbit class object, initiated from the State Vectors
    dem_height_interp_m: np.ndarray
        Interpolated DEM height [m],
        (from DEM LUT, interpolated over L1c subsalpled grid)
        dimensions [Naz_subsampled x N_rg_subsampled]
    dem_latitude_interp_rad: np.ndarray
        Interpolated DEM latitude, and converted from deg to rad [rad]
        (from DEM LUT, interpolated over L1c subsalpled grid)
        dimensions [Naz_subsampled x N_rg_subsampled]
    dem_longitude_interp_rad: np.ndarray
        Interpolated DEM longitude, and converted from deg to rad [rad]
        (from DEM LUT, interpolated over L1c subsalpled grid)
        dimensions [Naz_subsampled x N_rg_subsampled]
    """

    bps_logger.info("Geocoding initialization: interpolate DEM over L1C grid")
    dem_height_interp_m = parallel_reinterpolate(
        [dem_height_m],  # function works with lists
        dem_axis_az_s,
        dem_axis_sr_s,
        scs_axis_az_s,
        scs_axis_sr_s,
    )[0]  # function works with lists

    dem_latitude_interp_deg = parallel_reinterpolate(
        [dem_latitude_deg],  # function works with lists
        dem_axis_az_s,
        dem_axis_sr_s,
        scs_axis_az_s,
        scs_axis_sr_s,
    )[0]  # function works with lists

    dem_longitude_interp_deg = parallel_reinterpolate(
        [dem_longitude_deg],  # function works with lists
        dem_axis_az_s,
        dem_axis_sr_s,
        scs_axis_az_s,
        scs_axis_sr_s,
    )[0]  # function works with lists

    dem_valid_values_mask = np.invert(
        np.logical_and(
            np.logical_and(np.isnan(dem_height_interp_m), np.isnan(dem_latitude_interp_deg)),
            np.isnan(dem_longitude_interp_deg),
        )
    )

    gso = create_general_sar_orbit(state_vectors)

    return (
        gso,
        dem_height_interp_m,
        np.deg2rad(dem_latitude_interp_deg),
        np.deg2rad(dem_longitude_interp_deg),
        dem_valid_values_mask,
    )


def geocoding_update_dem_coordinates(
    lut_dem_height_m: np.ndarray,
    lut_dem_latitude_deg: np.ndarray,
    lut_dem_longitude_deg: np.ndarray,
    lut_dem_axis_az_s: np.ndarray,
    lut_dem_axis_sr_s: np.ndarray,
    scs_axis_az_s: np.ndarray,
    scs_axis_sr_s: np.ndarray,
    scs_az_axis_mjd: PreciseDateTime,
    state_vectors: StateVectors,
    emphasized_forest_height: float | np.ndarray,
    dgg_latitude_axis_rad,
    dgg_longitude_axis_rad,
) -> tuple[Delaunay, np.ndarray, np.ndarray]:
    """Geocoding first step: update dem coordinates
        Note: A preliminary interpolation step is needed to bring DEM latitude-longitude-height
        onto the same L1c grid (the subsampled grid after multilook)

    Parameters
    ----------
    lut_dem_height_m: np.ndarray
        DEM height [m], in slant-range, azimuth projection, from LUT [N_az_dem, N_rg_dem]
    lut_dem_latitude_deg: np.ndarray
        DEM latitude [deg], in slant-range, azimuth projection, from LUT [N_az_dem, N_rg_dem]
    lut_dem_longitude_deg: np.ndarray
        DEM longitude [deg], in slant-range, azimuth projection, from LUT [N_az_dem, N_rg_dem]
    lut_dem_axis_az_s: np.ndarray
        Azimuth axis [s] of the DEM (slant-range azimith projection)  [N_az_dem,1]
    lut_dem_axis_sr_s: np.ndarray
        Slant range axis [s] of the DEM (slant-range azimith projection)  [N_rg_dem,1]
    scs_axis_az_s: np.ndarray
        Azimuth axis [s] of the L1C SCS, already subsampled by multilook step [N_az_sub]
    scs_axis_sr_s: np.ndarray
        Slant.range axis [s] of the L1C SCS, already subsampled by multilook step [N_rg_sub]
    scs_az_axis_mjd: PreciseDateTime
        Mjd azimuth time of each data azimuth position [Naz_subsampled x 1]
    state_vectors: StateVectors
        State vectors from L1C orbit file [N_az,1]
    emphasized_forest_height: Union[float, np.ndarray]
        single float value [m] for Forest Disturbance algorithm
        Naz_sub x Nrg_sub matrix of float for Forest Height algorithm (is the estimated Forest Height)
    dgg_latitude_axis_rad: np.ndarray
        DGG sampling: output latitude vector [rad] where to geocode the input data
        [num_lat_out,1]
    dgg_longitude_axis_rad: np.ndarray
        DGG sampling: output longitude vector [rad] where to geocode the input data
        [num_lon_out,1]

    Returns
    -------
    delaunay: Delaunay
        Delaunay tessellation performed over the updated DEM Lat Lon coordinates
    dgg_latitude_mesh_rad:np.ndarray
        meshgrid of the dgg_latitude_axis_rad [rad] of dimensions [num_lat_out, num_lon_out]
    dgg_longitude_mesh_rad:np.ndarray
        meshgrid of the dgg_longitude_axis_rad [rad] of dimensions [num_lat_out, num_lon_out]
    """

    # Geocoding STEP 1/4: initialization (DEM interpolation and GSO initialization)
    (
        gso,
        dem_height_m,
        dem_latitude_rad,
        dem_longitude_rad,
        dem_valid_values_mask,
    ) = _geocoding_prepare_inputs(
        state_vectors,
        lut_dem_height_m,
        lut_dem_latitude_deg,
        lut_dem_longitude_deg,
        lut_dem_axis_az_s,
        lut_dem_axis_sr_s,
        scs_axis_az_s,
        scs_axis_sr_s,
    )

    bps_logger.info("Geocoding initialization: update DEM Coordinates")
    bps_logger.info(f"    using AUX PP2 2A emphasized forest height: {emphasized_forest_height} [m]")

    start_time = datetime.now()
    # Geocoding STEP 2/4: update DEM coordinates P0 -> P1
    num_az = dem_height_m.shape[0]
    num_rg = dem_height_m.shape[1]

    # dem positions in X, Y, Z: is a 3 x num_az x num_rg matrix
    p0_xyz_mat = llh2xyz(  # input is (3, N), radiants, meters
        np.array(
            [
                dem_latitude_rad.flatten(),
                dem_longitude_rad.flatten(),
                dem_height_m.flatten(),
            ]
        )
    ).reshape(3, num_az, num_rg)

    # sensor position X, Y, Z for each azimuth:
    # 2D matrix 3 x num_az repeated for each range, to get 3 x num_az x num_rg matrix
    s0_xyz_mat = np.repeat(
        gso.get_position(scs_az_axis_mjd)[:, :, np.newaxis],
        num_rg,
        axis=2,
    )

    # sensor direction versor, for each azimuth (and repeated for each range)
    i_az_mat = gso.get_velocity(scs_az_axis_mjd)  # velocity (3,Naz)
    i_az_mat = np.divide(i_az_mat, np.linalg.norm(i_az_mat, axis=0))  # velocity versor (3,Naz)
    i_az_mat = np.repeat(  # velocity versor (3,Naz, Nrg)
        i_az_mat[:, :, np.newaxis],
        num_rg,
        axis=2,
    )

    # line of sight versor, for each azimuth (and repeated for each range)
    i_los_mat = np.divide((p0_xyz_mat - s0_xyz_mat), np.linalg.norm(p0_xyz_mat - s0_xyz_mat, axis=0))

    # cross range versor, for each azimuth (and repeated for each range)
    i_xr_mat = np.cross(i_az_mat, i_los_mat, axis=0)

    # normal direction to ellipsoid reported to DEM, for each point
    i_z_mat = np.transpose(
        compute_ellipsoid_normals_at_height(
            np.transpose(p0_xyz_mat.reshape(3, p0_xyz_mat.shape[1] * p0_xyz_mat.shape[2]))
        )
    ).reshape(3, num_az, num_rg)

    # sin of the angle between i_z and i_xr, for each azimuth (and repeated for each range)
    # product returns cos(90-psi) = sin(psi)
    # no normalization needed, they are versors
    sin_psi_mat = np.sum(np.multiply(i_z_mat, i_xr_mat), axis=0)

    # This is computed as z_tomo/(i_z * i_xr/|i_z|*|i_xr|), which is equivalent to z_tomo/(|i_xr X i_e|)
    gamma_xr_mat = emphasized_forest_height / sin_psi_mat

    # Updat the DEM positions from P0 to P1:
    p1_xyz_mat = p0_xyz_mat + i_xr_mat * gamma_xr_mat  # 3 x num_az x num_rg

    # Updated DEM positions, in LLH
    p1_llh_rad_mat = xyz2llh(
        np.array([p1_xyz_mat[0].flatten(), p1_xyz_mat[1].flatten(), p1_xyz_mat[2].flatten()])
    ).reshape(3, num_az, num_rg)

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Update DEM coordinates processing time: {elapsed_time:2.1f} s")

    bps_logger.info("Geocoding initialization: prepare interpolator")
    start_time = datetime.now()
    # Geocoding STEP 3/4: interpolation over DGG
    delaunay = Delaunay(
        np.array(
            [
                p1_llh_rad_mat[0, :, :][dem_valid_values_mask].flatten(),  # Lat
                p1_llh_rad_mat[1, :, :][dem_valid_values_mask].flatten(),  # Lon
            ]
        ).T
    )

    dgg_longitude_mesh_rad, dgg_latitude_mesh_rad = np.meshgrid(
        dgg_longitude_axis_rad,
        dgg_latitude_axis_rad,
    )
    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Prepare interpolator processing time: {elapsed_time:2.1f} s")

    # Geocoding STEP 4/4: interpolator function evaluation, see geocoding() function.

    return (
        delaunay,
        dgg_latitude_mesh_rad,
        dgg_longitude_mesh_rad,
        dem_valid_values_mask,
    )


def geocoding(
    data: list[np.ndarray],
    delaunay: Delaunay,
    dgg_latitude_mesh_rad: np.ndarray,
    dgg_longitude_mesh_rad: np.ndarray,
    dem_valid_values_mask: np.ndarray,
    fill_value,
    interpolation_method: str = "linear",
) -> list[np.ndarray]:
    """Geocoding (last step)
    Evaluate the interpolator over the output DGG axes:
    it is the only step to be called for each data to be geocoded
    All the rest (update P0->P1, delaunay tessellation, DGG axis meshing) is done once,
    before this function.
    See also: geocoding_update_dem_coordinates()

    Parameters
    ----------
    data: np.ndarray
        slant-range, azimuth data to be geocoded
        dimensions Nvalues x [Naz_subsampled x N_rg_subsampled] (already subsampled respect to input L1c data)
    delaunay: Delaunay
        Delaunay tessellation performed in "geocoding_update_dem_coordinates()"
        over the updated DEM Lat Lon coordinates
    dgg_latitude_mesh_rad:np.ndarray
        meshgrid of the dgg_latitude_axis [rad] of dimensions [num_lat_out, num_lon_out]
    dgg_longitude_mesh_rad:np.ndarray
        meshgrid of the dgg_longitude_axis [rad] of dimensions [num_lat_out, num_lon_out]
    fill_value:
        Interpolator fill value to be used (for example np.nan, FLOAT_NODATA_VALUE...)
    interpolation_method: str = "linear",
        "linear" (default for BPS) or "cubic"
    Returns
    -------
    data_geocoded: np.ndarray
        Data geocoded [num_lat_out, num_lon_out]
    """

    start_time = datetime.now()

    #    Build a 2D array (npoints, Nvalues) from list
    data = [elem[dem_valid_values_mask].flatten() for elem in data]
    D = np.column_stack(data)

    #    Interpolate with delaunay
    if interpolation_method == "linear":
        interp_fun = LinearNDInterpolator(delaunay, D, fill_value=fill_value)
    elif interpolation_method == "cubic":
        interp_fun = CloughTocher2DInterpolator(delaunay, D, fill_value=fill_value)
    data_geocoded = interp_fun(dgg_latitude_mesh_rad, dgg_longitude_mesh_rad)

    #    return a list of geocoded values

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"    processing time: {elapsed_time:2.1f} s")
    return list(np.moveaxis(data_geocoded, -1, 0))


def fnf_annotation(
    fnf: FnFMask,
    latitude_vec: np.ndarray,
    longitude_vec: np.ndarray,
    forest_mask_interpolation_threshold: float,
) -> dict:
    """FNF annotation
    Parameters
    ----------
    """

    bps_logger.info("Compute FNF annotation step:")

    # the FNF needs to be extracted from the whole one, over the Product latitude longitude vectors
    lut_fnf_dict = interpolate_fnf(
        fnf.mask,
        fnf.lat_axis,
        fnf.lon_axis,
        forest_mask_interpolation_threshold,
        latitude_vec,
        longitude_vec,
    )

    return lut_fnf_dict


def generate_fnf(land_cover_map_path: Path, output_fnf_path: Path):
    """Generate BIOMASS L2a FNF product from ESA land Cover map

    Description
    -----------
    Land Cover MAP ESA:
    Download website:
    https://maps.elie.ucl.ac.be/CCI/viewer/
    Go to:
        Download data -> Climate Research data package
    Insert credential in "Data access" module
    For the updated map, go to:
        https://maps.elie.ucl.ac.be/CCI/viewer/
    Login or subscribe, than you can download the Map (2020, v2.11) in NETCDF format.
    To extract the tiff from NETCDF, use this command (on a command window with python avaliable,
    in an environment with GDAL: "conda install GDAL"):
    gdalwarp -of Gtiff -co COMPRESS=LZW -co TILED=YES -ot Byte -te -180.0000000 -90.0000000 180.0000000
    90.0000000 -tr 0.002777777777778 0.002777777777778 -t_srs EPSG:4326
    NETCDF:C:/MyMapFolder/LandCoverMap2020/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc:lccs_class
    C:/MyMapFolder/LandCoverMap2020/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.tif

    all_numbers = [
        10,
        11,
        12,
        20,
        30,
        40,
        50,
        60,
        61,
        62,
        70,
        71,
        72,
        80,
        81,
        82,
        90,
        100,
        110,
        120,
        121,
        122,
        130,
        140,
        150,
        151,
        152,
        153,
        160,
        170,
        180,
        190,
        200,
        201,
        202,
        210,
        220,
    ]

        all_names = {
        "cropland_rainfed": 0,
        "cropland_rainfed_herbaceous_cover": 0,
        "cropland_rainfed_tree_or_shrub_cover": 0,
        "cropland_irrigated": 0,
        "mosaic_cropland": 1,
        "mosaic_natural_vegetation": 1,
        "tree_broadleaved_evergreen_closed_to_open": 1,
        "tree_broadleaved_deciduous_closed_to_open": 1,
        "tree_broadleaved_deciduous_closed": 1,
        "tree_broadleaved_deciduous_open": 1,
        "tree_needleleaved_evergreen_closed_to_open": 1,
        "tree_needleleaved_evergreen_closed": 1,
        "tree_needleleaved_evergreen_open": 1,
        "tree_needleleaved_deciduous_closed_to_open": 1,
        "tree_needleleaved_deciduous_closed": 1,
        "tree_needleleaved_deciduous_open": 1,
        "tree_mixed": 1,
        "mosaic_tree_and_shrub": 1,
        "mosaic_herbaceous": 1,
        "shrubland": 1,
        "shrubland_evergreen": 1,
        "shrubland_deciduous": 1,
        "grassland": 1,
        "lichens_and_mosses": 1,
        "sparse_vegetation": 1,
        "sparse_tree": 1,
        "sparse_shrub": 1,
        "sparse_herbaceous": 1,
        "tree_cover_flooded_fresh_or_brakish_water": 1,
        "tree_cover_flooded_saline_water": 1,
        "shrub_or_herbaceous_cover_flooded": 0,
        "urban": 0,
        "bare_areas": 0,
        "bare_areas_consolidated": 0,
        "bare_areas_unconsolidated": 0,
        "water": 0,
        "snow_and_ice": 0,
    }

    Parameters
    ----------

    Returns
    -------

    Raises
    ------

    """

    # Read input Land Cover Map, data and metadata
    data_driver = gdal.Open(str(land_cover_map_path), 0)
    data = data_driver.GetRasterBand(1).ReadAsArray()

    md = data_driver.GetMetadata()
    projection = data_driver.GetProjection()
    # geotransform = [Longitude in, Longitude step, 0, Latitude in, 0 latitude step]
    geotransform = data_driver.GetGeoTransform()

    data_driver = None

    # Output dir and filenames
    lcm_date_start = md["NC_GLOBAL#time_coverage_start"]
    lcm_date_end = md["NC_GLOBAL#time_coverage_end"]
    dir_name = _set_fnf_name(lcm_date_start, lcm_date_end, "dir")
    file_name = _set_fnf_name(lcm_date_start, lcm_date_end, "file")

    fnf_dir_name = output_fnf_path.joinpath(dir_name, "data")
    fnf_dir_name.mkdir(parents=True, exist_ok=True)
    fnf_file_name = fnf_dir_name.joinpath(file_name + ".tiff")
    fnf_support_dir_name = output_fnf_path.joinpath(dir_name, "support")
    fnf_support_dir_name.mkdir(parents=True, exist_ok=True)

    # convert Land Cover map to binary Forest/Non.Forest Map:
    # 0: non forest
    # 1: forest
    # 255: no valid data
    lines_per_block = 1000
    num_blocks = (data.shape[0] + lines_per_block - 1) // lines_per_block
    for k in range(num_blocks):
        print(f" Convert Land Cover Map to Forest/non-forest Map, step : {k + 1} / {num_blocks}")

        # Prepare blocking algorithm
        lines_offset = k * lines_per_block
        num_lines = min(data.shape[0] - lines_offset, lines_per_block)
        data_block = data[lines_offset : lines_offset + num_lines, :]

        #  replace values
        data_block[data_block == 0] = INT_NODATA_VALUE
        data_block[data_block <= 20] = 0
        data_block[np.logical_and(data_block >= 180, data_block < INT_NODATA_VALUE)] = 0
        data_block[np.logical_and(data_block > 20, data_block < 180)] = 1

    # Initialize GDAL
    latitude_n, longitude_n = data.shape
    driver_out = gdal.GetDriverByName("GTiff")
    outdata = driver_out.Create(
        str(fnf_file_name),
        longitude_n,
        latitude_n,
        1,
        eType=gdal.GDT_Byte,
        options=[
            "INTERLEAVE=BAND",
            f"COMPRESS={COMPRESSION_SCHEMA_MDS_ZSTD}",
            "ZSTD_LEVEL=5",
            "MAX_Z_ERROR=0",
            "Tiled=YES",
            f"BLOCKXSIZE={256}",
            f"BLOCKYSIZE={256}",
        ],
    )
    outdata.GetRasterBand(1).WriteArray(data)
    outdata.GetRasterBand(1).SetNoDataValue(INT_NODATA_VALUE)

    outdata.SetGeoTransform(geotransform)
    outdata.SetProjection(projection)

    # Update metadata
    md.pop("lccs_class#flag_colors")
    md["lccs_class#flag_meanings"] = "non_forest forest invalid"
    md["lccs_class#flag_values"] = "{0,1,255}"
    md["lccs_class#long_name"] = "Forest/Non-Forest Mask computed from Land Cover LCSS"
    md["lccs_class#standard_name"] = "Forest/Non-Forest Mask"
    md["lccs_class#valid_max"] = "255"
    md["lccs_class#valid_min"] = "0"
    outdata.SetMetadata(md)

    # Finalize GDAL
    outdata.FlushCache()
    outdata = None

    # DEBUG plotting
    # if plot_flag:
    #     png_output_path = fnf_support_dir_name.joinpath(file_name + ".png")
    #     from matplotlib import pyplot as plt
    #     from matplotlib.colors import ListedColormap
    #     # Generate the PNG
    #     green_str = "#42fd00"
    #     brown_str = "#9d550a"

    #     cmap = ListedColormap([brown_str, green_str])

    #     plot_subsampling = int(10)
    #     plt.figure()
    #     plt.imshow(
    #         data[1:latitude_n:plot_subsampling, 1:longitude_n:plot_subsampling],
    #         interpolation="none",
    #         origin="upper",
    #         cmap=cmap,
    #         # norm=norm,
    #         extent=[-180, 180, -90, 90],
    #     )
    #     plt.colorbar()
    #     plt.xlabel("longitude [°]")
    #     plt.ylabel("latitude [°]")
    #     plt.title("FNF from Land Cover MAP")
    #     plt.savefig(png_output_path)

    print(f"Forest/Non-Forest Map generated {fnf_dir_name}.")


def _set_fnf_name(lcm_date_start: str, lcm_date_end: str, name_type: str):
    if name_type == "dir":
        name = "_".join(
            [
                "BIO",
                "AUX_L2_FNF",
                _convert_lcm_date(lcm_date_start),
                _convert_lcm_date(lcm_date_end),
                pdt_to_compact_date(PreciseDateTime.from_numeric_datetime(year=int(lcm_date_start[0:4]))),
            ]
        )

    elif name_type == "file":
        name = "_".join(
            [
                "BIO",
                "AUX_L2_FNF",
                _convert_lcm_date(lcm_date_start),
                _convert_lcm_date(lcm_date_end),
            ]
        )
        name = name.lower()

    return name


def _convert_lcm_date(lcm_date):
    year = lcm_date[0:4]
    month = lcm_date[4:6]
    day = lcm_date[6:8]
    seconds_str = "000000"

    return year + month + day + "T" + seconds_str


def scs_axis_generation(
    stack_product: BIOMASSStackProduct,
) -> tuple[np.ndarray, np.ndarray, PreciseDateTime]:
    """Retrieve L1c SCS data slant range and azimuth temporal axis
    Parameters
    ----------
    stack_product: BIOMASSStackProduct
        the procuct contains the axis definition, retrieved from sarImage main nnotations

    Returns
    -------
    scs_axis_sr_s: np.ndarray
        slant range temporal axis, in seconds
    scs_axis_az_s: np.ndarray
        azimuth temporal axis, in seconds
    scs_axis_az_mjd: PreciseDateTime
        azimuth temporal axis, in MJD
    """

    scs_axis_sr_s = np.linspace(
        stack_product.first_sample_sr_time,
        stack_product.first_sample_sr_time + stack_product.rg_time_interval * stack_product.number_of_samples,
        num=stack_product.number_of_samples,
    ).astype(np.float64)
    scs_axis_az_s = np.linspace(
        0,
        stack_product.az_time_interval * stack_product.number_of_lines,
        num=stack_product.number_of_lines,
    ).astype(np.float64)

    start_time_az_mjd = stack_product.first_line_az_time
    scs_axis_az_mjd = scs_axis_az_s + start_time_az_mjd

    return scs_axis_sr_s, scs_axis_az_s, scs_axis_az_mjd


def check_lat_lon_orientation(latitude_axis, longitude_axis):
    invert_latitude = False
    invert_longitude = False
    if latitude_axis[-1] - latitude_axis[0] < 0:
        invert_latitude = True
    if longitude_axis[-1] - longitude_axis[0] < 0:
        invert_longitude = True

    return invert_latitude, invert_longitude


def compute_ellipsoid_normals(
    coords: npt.ArrayLike,
    *,
    semi_major_axis: npt.ArrayLike = WGS84.semi_major_axis,
    semi_minor_axis: npt.ArrayLike = WGS84.semi_minor_axis,
) -> np.ndarray:
    """Compute normals to ellipsoid at given points

    the axis can be arrays to allow for custom semi-axis at each point

    Parameters
    ----------
    coords : npt.ArrayLike
        (3,), (N, 3) one or more points
    semi_major_axis : npt.ArrayLike, defaults to WGS84
        scalar, (N,) one or more semi_major_axis (keyword only)
    semi_minor_axis : npt.ArrayLike, defaults to WGS84
        scalar, (N,) one or more semi_minor_axis (keyword only)

    Returns
    -------
    np.ndarray
        (3,), (N, 3) normal versors
    """

    semi_major_axis, semi_minor_axis = np.broadcast_arrays(semi_major_axis, semi_minor_axis)
    normals = (
        np.asarray(coords)
        / np.column_stack(
            (
                np.square(semi_major_axis),
                np.square(semi_major_axis),
                np.square(semi_minor_axis),
            )
        ).squeeze()
    )

    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def compute_ellipsoid_normals_at_height(points: npt.ArrayLike) -> np.ndarray:
    """Compute ellipsoid normals at correct height

    Parameters
    ----------
    points : npt.ArrayLike
        (3,), (N, 3) one or more points

    Returns
    ------- np.ndarray
        (3,), (N, 3) one or more normal versors
    """
    llh_coordinates = xyz2llh(np.transpose(points))  # type: ignore
    height = llh_coordinates[2]
    return compute_ellipsoid_normals(
        points,
        semi_major_axis=WGS84.semi_major_axis + height,
        semi_minor_axis=WGS84.semi_minor_axis + height,
    )


# SKP
def parallel_build_screen_exp_sin_cos(data_lst, axis1_in, axis2_in, axis1_out, axis2_out):
    N = len(data_lst)
    execution_lst = [[] for _ in range(N)]
    for i in range(N):
        execution_lst[i].extend([data_lst[i], axis1_in, axis2_in, axis1_out, axis2_out])

    results_lst = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for number, res in zip(execution_lst, executor.map(lambda f: build_screen_core_sin_cos(*f), execution_lst)):
            results_lst.append(res)

    return results_lst


# DSI
def parallel_build_screen_exp(data_lst, axis1_in, axis2_in, axis1_out, axis2_out):
    N = len(data_lst)
    execution_lst = [[] for _ in range(N)]
    for i in range(N):
        execution_lst[i].extend([data_lst[i], axis1_in, axis2_in, axis1_out, axis2_out])

    results_lst = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for number, res in zip(execution_lst, executor.map(lambda f: build_screen_core(*f), execution_lst)):
            results_lst.append(res)

    return results_lst


def build_screen_core_sin_cos(data, dx_az, dx_rg, sx_az, sx_rg):
    # SKP
    return np.exp(
        1j
        * np.arctan2(
            _linear_interpolation_2d(np.sin(data), dx_az, dx_rg, sx_az, sx_rg, fill_value=0.0, tolerance=0.0000001),
            _linear_interpolation_2d(np.cos(data), dx_az, dx_rg, sx_az, sx_rg, fill_value=0.0, tolerance=0.0000001),
        )
    ).astype(np.complex64)


def build_screen_core(data, dx_az, dx_rg, sx_az, sx_rg):
    # DSI
    return np.exp(
        1j * _linear_interpolation_2d(data, dx_az, dx_rg, sx_az, sx_rg, fill_value=0.0, tolerance=0.0000001)
    ).astype(np.complex64)


def parallel_reinterpolate(data_lst, axis1_in, axis2_in, axis1_out, axis2_out, fill_value=np.nan):
    N = len(data_lst)
    execution_lst = [[] for _ in range(N)]
    for i in range(N):
        execution_lst[i].extend([data_lst[i], axis1_in, axis2_in, axis1_out, axis2_out])

    results_lst = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for number, res in zip(
            execution_lst,
            executor.map(lambda f: reinterpolate_core(*f, fill_value), execution_lst),
        ):
            results_lst.append(res)

    return results_lst


def reinterpolate_core(data, dx_az, dx_rg, sx_az, sx_rg, fill_value):
    return _linear_interpolation_2d(data, dx_az, dx_rg, sx_az, sx_rg, fill_value).astype(type(data[0, 0]))


@nb.njit(nogil=True, cache=True, parallel=False)
def _regular_linear_interpolator(outputM, inputM, ax0in, ax1in, ax0out, ax1out, outOfBounds, tolerance):
    innerM = np.zeros((inputM.shape[0], outputM.shape[1]), dtype=inputM.dtype)

    dx0 = np.mean(ax0in[1:] - ax0in[:-1])
    dx1 = np.mean(ax1in[1:] - ax1in[:-1])

    sign0 = 1.0
    if dx0 < 0.0:
        sign0 = -1.0

    sign1 = 1.0
    if dx1 < 0.0:
        sign1 = -1.0

    tol0 = sign0 * dx0 * tolerance
    tol1 = sign1 * dx1 * tolerance

    mx0 = sign0 * ax0in[0]
    Mx0 = sign0 * ax0in[-1]

    mx1 = sign1 * ax1in[0]
    Mx1 = sign1 * ax1in[-1]

    dx0 *= sign0
    dx1 *= sign1

    for i1 in range(outputM.shape[1]):
        px1 = sign1 * ax1out[i1]
        if px1 < mx1 - tol1:
            innerM[:, i1] = outOfBounds
        elif px1 <= mx1:
            innerM[:, i1] = inputM[:, 0]
        elif px1 > Mx1 + tol1:
            innerM[:, i1] = outOfBounds
        elif px1 >= Mx1:
            innerM[:, i1] = inputM[:, -1]
        else:
            c = int((px1 - mx1) // dx1)
            w = (px1 - sign1 * ax1in[c]) / dx1
            innerM[:, i1] = (1.0 - w) * inputM[:, c] + w * inputM[:, c + 1]

    for i0 in range(outputM.shape[0]):
        px0 = sign0 * ax0out[i0]
        if px0 < mx0 - tol0:
            outputM[i0, :] = outOfBounds
        elif px0 <= mx0:
            outputM[i0, :] = innerM[0, :]
        elif px0 > Mx0 + tol0:
            outputM[i0, :] = outOfBounds
        elif px0 >= Mx0:
            outputM[i0, :] = innerM[-1, :]
        else:
            c = int((px0 - mx0) // dx0)
            w = (px0 - sign0 * ax0in[c]) / dx0
            outputM[i0, :] = (1.0 - w) * innerM[c, :] + w * innerM[c + 1, :]

    return


def _linear_interpolation_2d(
    data_in,
    axis_az_in,
    axis_sr_in,
    axis_sub_az_out,
    axis_sr_out,
    fill_value=np.nan,
    tolerance=0.001,
):
    data_out = np.zeros((axis_sub_az_out.shape[0], axis_sr_out.shape[0]), dtype=data_in.dtype)

    _regular_linear_interpolator(
        data_out,
        data_in,
        axis_az_in,
        axis_sr_in,
        axis_sub_az_out,
        axis_sr_out,
        fill_value,
        tolerance,
    )

    return data_out


def averaging_windows_sizes(
    b_az: float,
    b_rg: float,
    f_az: float,
    f_rg: float,
    product_resolution: float,
    average_az_velocity: float,
    average_incidence_angle_rad: float,
) -> tuple[int, int, int]:
    """
    Compute averaging windows sizes for the convolutons of data

    Parameters
    ----------
    b_az: float
        L1c azimuth bandwidth [Hz]
    b_rg: float
        L1c range bandwidth [Hz]
    f_az: float
        L1c azimuth axis frequency [Hz]
    f_rg: float
        L1c range axis frequency [Hz]
    product_resolution: float
        Value in [m] to be used as the resolution on ground range map
        and also to perform the covariance averaging in radar coordinates
    average_az_velocity: float
        Average azimuth velocity [m/s]
    average_incidence_angle_rad: float
        average value of the incidence angle [rad]

    Returns
    -------
    averaging_window_size_az: int
        Averaging windows size in azimuth direction
    averaging_window_size_rg: int
        Averaging windows size in slant range direction
    number_of_looks: int
        ENL, equivalent number of looks
    """
    equivalent_resolution_rg = LIGHTSPEED / (2 * f_rg)
    equivalent_resolution_az = average_az_velocity / f_az

    averaging_window_size_rg = int(
        np.ceil((product_resolution / equivalent_resolution_rg) * np.sin(average_incidence_angle_rad))
    )
    averaging_window_size_az = int(np.ceil(product_resolution / equivalent_resolution_az))

    resolution_rg = LIGHTSPEED / (2 * b_rg)
    resolution_az = average_az_velocity / b_az

    number_of_looks = int(
        np.ceil((product_resolution**2) / (resolution_az * resolution_rg) * np.sin(average_incidence_angle_rad))
    )

    return (
        averaging_window_size_az,
        averaging_window_size_rg,
        number_of_looks,
    )


def fast_read_baseline_ordering_index(acquisition_path: Path):
    start_str_to_find = "<baselineOrderingIndex>"
    stop_str_to_find = "</baselineOrderingIndex>"

    main_annotation_xml_path = acquisition_path.joinpath(
        "annotation_coregistered",
        acquisition_path.name[0:70].lower() + "_annot.xml",
    )
    current_file_content = main_annotation_xml_path.read_text(encoding="utf-8")
    idx_start = current_file_content.find(start_str_to_find) + len(start_str_to_find)
    idx_stop = current_file_content.find(stop_str_to_find)

    return int(current_file_content[idx_start:idx_stop])


def refine_dgg_search_tiles(
    tiles_dict: dict,
    lat_vec: np.ndarray,
    lon_vec: np.ndarray,
) -> list[str]:
    """Refine the list of tiles falling inside input STA coverage (footprint):
    dgg_search_tiles selected the list of tiles falling in the lat lon bounding box of the STA inputs;
    this function refines the search by keeping only the tiles intersecting the footprint os the STA inputs

    We use gdal ogr polygons, to compare STA footprint with each tile footprint

    Inputs
    ------
    tiles_dict:dict
        Dictionary containing, for each of the tiles selected by dgg_search_tiles: tile_id as key, latitude_vector and longitude_vector as values
    lat_vec: float
        Vector of latitudes of the STA coverage (footprint)
    lon_vec: float
       Vector of longitudes of the STA coverage (footprint)

    Returns
    -------
    tile_id_list: list[str]
        List of tile ids falling in the STA footprint
    """

    tile_id_list = []

    # ogr polygon of the input STA footprint
    wkt_stack_input = f"POLYGON (({lon_vec[0]} {lat_vec[0]}, {lon_vec[1]} {lat_vec[1]}, {lon_vec[2]} {lat_vec[2]}, {lon_vec[3]} {lat_vec[3]}, {lon_vec[0]} {lat_vec[0]}))"
    poly_stack = ogr.CreateGeometryFromWkt(wkt_stack_input)

    for tile_id_curr, tile_obj in tiles_dict.items():
        # ogr polygon of the current tile footprint
        lat_min_tile = np.min(tile_obj["latitude_vector"])
        lat_max_tile = np.max(tile_obj["latitude_vector"])
        lon_min_tile = np.min(tile_obj["longitude_vector"])
        lon_max_tile = np.max(tile_obj["longitude_vector"])
        wkt_tile = f"POLYGON (({lon_min_tile} {lat_min_tile}, {lon_min_tile} {lat_max_tile}, {lon_max_tile} {lat_max_tile}, {lon_max_tile} {lat_min_tile}, {lon_min_tile} {lat_min_tile}))"
        poly_tile = ogr.CreateGeometryFromWkt(wkt_tile)

        # Check if the tile footprint intersects the STA footprint
        if not poly_tile.Intersection(poly_stack).ExportToWkt() == "POLYGON EMPTY":
            tile_id_list.append(tile_id_curr)

    return tile_id_list
