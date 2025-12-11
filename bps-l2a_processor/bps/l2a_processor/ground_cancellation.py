# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import warnings
from datetime import datetime
from pathlib import Path

import numexpr as ne
import numpy as np
from bps.common import bps_logger
from bps.l2a_processor.core.translate_aux_pp2_2a import (
    GroundCancellationConfAGB,
    GroundCancellationConfFD,
    OperationalModeType,
)
from bps.l2a_processor.l2a_common_functionalities import parallel_reinterpolate
from bps.transcoder.sarproduct.biomass_l2aproduct_writer import FLOAT_NODATA_VALUE
from bps.transcoder.sarproduct.biomass_stackproduct_reader import (
    BIOMASSStackProductReader,
)


def ground_cancellation(
    scs_pol_list: list[list[np.ndarray]],
    scs_axis_sr_s: np.ndarray,
    scs_axis_az_s: np.ndarray,
    primary_image_index: int,
    vertical_wavenumber_list: list[np.ndarray],
    vert_wavenumber_axis_sr_s: np.ndarray,
    vert_wavenumber_axis_az_s: np.ndarray,
    aux_pp2_conf_gn: GroundCancellationConfAGB | GroundCancellationConfFD,
    acquisition_paths_selected_not_sorted: list[Path],
) -> tuple[list[np.ndarray], int | None]:
    """Ground cancellation

    Parameters
    ----------
    scs_pol_list: List[List[np.ndarray]]
        Stack of calibrated scs data
        list with P=3 polarizations (in the order HH, XP, VV),
        each containing a list with M=2 or 3 acquisitions,
        each of dimensions [N_az x N_rg]
        Note: it can contain also only 2 polarization, in this case, the list is still of P=3, but one element is None
        Can be None, if images_pair_pol_list is specified instead
    scs_axis_sr_s: np.ndarray
        slant range time axis of each scs data in scs_pol_list [s]
    scs_axis_az_s: np.ndarray
        azimuth time axis of each scs data in scs_pol_list[s]
    primary_image_index: int
        index of the acquisition list element which corresponds to the reference acquisition
    vertical_wavenumber_list: List[np.ndarray]
        list of M vertical wavenumbers
        the one corresponding to primary_image_index should be all zeros by definition
    vert_wavenumber_axis_sr_s: np.ndarray
        slant range time axis of each vertical wavenumber data in vertical_wavenumber_list [s]
    vert_wavenumber_axis_az_s: np.ndarray
        azimuth time axis of each vertical wavenumber data in vertical_wavenumber_list [s]
    aux_pp2_conf_gn: Union[GroundCancellationConfAGB, GroundCancellationConfFD]
        ground notching AUX PP2 2A configuration file:
        can be the FD or the AGB one
        FD:
            emphasized_forest_height
            operational_mode
            images_pair_selection: optional, not used if operational_mode is not "insar_pair"
            disable_ground_cancellation_flag: optional, default False
        AGB:
            computeGNpowerFlag
            emphasized_forest_height
            operational_mode
            images_pair_selection: optional, not used if operational_mode is not "insar_pair"
            disable_ground_cancellation_flag: optional, default False
            radiometricCalibrationFlag: not used here, but in sigma naught normalisation

    Returns
    -------
    ground_cancelled: List[np.ndarray]
        ground cancelled scs data
        list with P=3 polarizations (in the order HH, XP, VV)
    """

    start_time = datetime.now()

    # output initialization: list will contain three polarizations in the order HH, XP, VV
    ground_cancelled_list = []
    idx_reference_to_save = None
    if not aux_pp2_conf_gn.disable_ground_cancellation_flag:
        bps_logger.info("Compute Ground Cancellation:")

        # Preliminary computations
        # Ground cancellation parameters
        bps_logger.info(
            f"    using AUX PP2 2A emphasized forest height: {aux_pp2_conf_gn.emphasized_forest_height} [m]"
        )
        demodulation_height = aux_pp2_conf_gn.emphasized_forest_height / 2  # [m]
        kz0 = (
            np.pi / aux_pp2_conf_gn.emphasized_forest_height
        )  # emphasized forest height conversion to wavenumber (desired phase-to-height)

        bps_logger.info(f"    using AUX PP2 2A operational mode: {aux_pp2_conf_gn.operational_mode.value}")
        if aux_pp2_conf_gn.operational_mode == OperationalModeType.INSAR_PAIR:
            # In case of "insar pair", read here the two images specified in "images_pair_selection"
            path_1 = Path(aux_pp2_conf_gn.images_pair_selection.acquisition[0].value)
            path_2 = Path(aux_pp2_conf_gn.images_pair_selection.acquisition[0].value)
            bps_logger.info(
                f"    reading now following insar pair acquisitions, from AUX PP2 2A: \r\n{path_1} \r\n{path_2}"
            )
            del scs_pol_list

            if not path_1.is_absolute():
                path_1 = acquisition_paths_selected_not_sorted[0].parent.joinpath(path_1)
            if not path_2.is_absolute():
                path_2 = acquisition_paths_selected_not_sorted[0].parent.joinpath(path_2)
            if not path_1.exists():
                raise RuntimeError(
                    f"Insar acquisition {path_1.name} not found in {acquisition_paths_selected_not_sorted[0].parent}"
                )
            if not path_2.exists():
                raise RuntimeError(
                    f"Insar acquisition {path_2.name} not found in {acquisition_paths_selected_not_sorted[0].parent}"
                )

            product_1 = BIOMASSStackProductReader(path_1).read()
            # Convert no data values to nan, for the processing
            for idx, data in enumerate(product_1.data_list):
                nan_mask = data == FLOAT_NODATA_VALUE
                product_1.data_list[idx][nan_mask] = np.nan

            product_2 = BIOMASSStackProductReader(path_2).read()
            # Convert no data values to nan, for the processing
            for idx, data in enumerate(product_2.data_list):
                nan_mask = data == FLOAT_NODATA_VALUE
                product_2.data_list[idx][nan_mask] = np.nan

            # Only reshape of the list of lists
            scs_pol_list = []
            for data1_curr_pol, data2_curr_pol in zip(product_1.data_list, product_2.data_list):
                scs_pol_list.append([data1_curr_pol, data2_curr_pol])
            del product_1, product_2

        # Check the presence of HH, XP and VV polarizations (one of three can be None)
        pol_is_present_list = []
        for scs_acq_list in scs_pol_list:
            if scs_acq_list is None:
                pol_is_present_list.append(False)
            else:
                pol_is_present_list.append(True)
        if sum(pol_is_present_list) < 2:
            raise RuntimeError("Ground cancellation: a minimum of two polarizartions are needed by L2a processor")

        # Get number of acquisitons and matrix shape
        for idx, pol_is_present in enumerate(pol_is_present_list):
            if pol_is_present:
                first_pol_present_idx = idx
                num_acq = len(scs_pol_list[first_pol_present_idx])
                Naz, Nrg = scs_pol_list[first_pol_present_idx][0].shape
                break  # we suppose same dimensions for each polarization

        # Logging operational mode
        if not aux_pp2_conf_gn.operational_mode == OperationalModeType.INSAR_PAIR and num_acq == 2:
            # Forcing insar pair
            # Message for the user, the code does not need to re set the flag
            bps_logger.info(
                "    operational mode from AUX PP2 2A is ignored and automatically set to INSAR PAIR (two acquisitions are available)"
            )
        elif (
            type(aux_pp2_conf_gn) == GroundCancellationConfFD  # noqa: E721
            and aux_pp2_conf_gn.operational_mode == OperationalModeType.MULTI_REFERENCE
        ):
            # Is the default for AGB and applicable only for AGB
            raise ValueError(
                f"    operational mode {aux_pp2_conf_gn.operational_mode.value} is incompatible with FD processor"
            )
        # Preliminary interpolation to bring vertical wavenumbers onto the same L1c grid
        # Not needed if insar pair or if two acquisitions in scs_pol_list
        if (
            not (num_acq == 2 or aux_pp2_conf_gn.operational_mode == OperationalModeType.INSAR_PAIR)
            and not scs_pol_list[first_pol_present_idx][0].shape
            == vertical_wavenumber_list[first_pol_present_idx].shape
        ):
            bps_logger.info("    preliminary interpolation to bring vertical wavenumbers onto the same L1c grid")
            sum_nan_before_list = [np.sum(np.isnan(wv)) for wv in vertical_wavenumber_list]
            vertical_wavenumber_list = parallel_reinterpolate(
                vertical_wavenumber_list,
                vert_wavenumber_axis_az_s,
                vert_wavenumber_axis_sr_s,
                scs_axis_az_s,
                scs_axis_sr_s,
            )
            sum_nan_after_list = [np.sum(np.isnan(wv)) for wv in vertical_wavenumber_list]
            for idx, (sum_nan_before, sum_nan_after) in enumerate(zip(sum_nan_before_list, sum_nan_after_list)):
                if sum_nan_after - sum_nan_before > 0:
                    if sum_nan_after > 0:
                        bps_logger.warning(
                            f"    invalid samples before / after LUT interpolation: {sum_nan_before / (len(vert_wavenumber_axis_az_s) * len(vert_wavenumber_axis_sr_s) * len(vertical_wavenumber_list)) * 100:2.3f}% / {sum_nan_after / (vertical_wavenumber_list[0].size * len(vertical_wavenumber_list)) * 100:2.3f}%, for wavenumber {idx + 1} of {len(vertical_wavenumber_list)}"
                        )

                warnings.filterwarnings(action="ignore", message="All-NaN axis encountered")

        # Ground cancellation performed independently for each polarization
        pol_names_for_log = ["HH", "XP", "VV"]  # just needee for logging reasons
        if num_acq == 2:
            bps_logger.info("    Ground Cancellation is performed by simple subtraction of one image from the other")
            # this is done here under

        if num_acq > 2 and aux_pp2_conf_gn.operational_mode == OperationalModeType.SINGLE_REFERENCE:
            # Choose optimal reference image in SINGLE_REFERENCE case.
            # This is based on whether the kz0 is within the kz interval of the respective reference-non_reference combination
            # Also updates the sign og kz0 if needed
            idx_reference, kz0 = reference_acquisition_selection(
                vertical_wavenumber_list, kz0, acquisition_paths_selected_not_sorted
            )
            # Final vertical wavenumbers difference computation with the selected reference
            vertical_wavenumber_diff_list = []
            for vertical_wavenumber_current in vertical_wavenumber_list:
                vertical_wavenumber_diff_list.append(
                    vertical_wavenumber_current - vertical_wavenumber_list[idx_reference]
                )

            idx_reference_to_save = int(idx_reference)

        # Ground cancellation computation for each polarization:
        for idx_pol, (pol_is_present, scs_acq_list) in enumerate(
            zip(
                pol_is_present_list,
                scs_pol_list,
            )
        ):
            bps_logger.info(f"    ground cancellation of polarization {pol_names_for_log[idx_pol]}")
            if not pol_is_present:
                bps_logger.info("        skipping missing polarization")
                ground_cancelled_list.append(None)

            elif num_acq == 2:
                # Simplified algorithm: difference from the two, the order is not influent
                ground_cancelled_list.append(scs_acq_list[0] - scs_acq_list[1])
                mask_final = np.zeros((Naz, Nrg), dtype=np.uint8)

            elif aux_pp2_conf_gn.operational_mode == OperationalModeType.SINGLE_REFERENCE:
                # compute ground notch
                (
                    notch_final,
                    mask_final,
                ) = ground_cancellation_core(
                    scs_acq_list,
                    vertical_wavenumber_diff_list,
                    kz0,
                    idx_reference,
                    demodulation_height,
                )

                notch_final[mask_final == 1] = 0

                if isinstance(aux_pp2_conf_gn, GroundCancellationConfAGB) and aux_pp2_conf_gn.compute_gn_power_flag:
                    notch_final = np.abs(notch_final) ** 2

                ground_cancelled_list.append(notch_final)

            elif aux_pp2_conf_gn.operational_mode == OperationalModeType.MULTI_REFERENCE:
                notch_final = np.zeros((Naz, Nrg))
                mask_final = np.zeros((Naz, Nrg))

                for idx_ref, vertical_wavenumber_ref in enumerate(vertical_wavenumber_list):
                    # current reference (changes at each iteration)
                    idx_reference = idx_ref
                    # at each iteration compute this diff, with a different reference:
                    vertical_wavenumber_diff_list = []
                    for vertical_wavenumber_current in vertical_wavenumber_list:
                        vertical_wavenumber_diff_list.append(vertical_wavenumber_current - vertical_wavenumber_ref)
                    # Check if it is better to generate +kz0 or -kz0
                    kz0 = check_pm_kz0(vertical_wavenumber_diff_list, kz0)

                    (
                        notch_temp,
                        notch_mask,
                    ) = ground_cancellation_core(
                        scs_acq_list,
                        vertical_wavenumber_diff_list,
                        kz0,
                        idx_reference,
                        demodulation_height,
                    )
                    notch_temp[notch_mask == 1] = 0

                    # Current reference contribute sum
                    notch_final = notch_final + np.abs(notch_temp) ** 2
                    i_valid = 1 - notch_mask
                    mask_final = mask_final + i_valid

                mask_final[mask_final == 0] = 1

                ground_cancelled_list.append(np.sqrt(notch_final / mask_final))

        stop_time = datetime.now()
        elapsed_time = (stop_time - start_time).total_seconds()
        bps_logger.info(f"Ground Cancellation processing time: {elapsed_time:2.1f} s")
        return ground_cancelled_list, idx_reference_to_save

    else:
        bps_logger.info("L2a Ground Cancellation skipped due to AUX PP2 2A configuration:")
        bps_logger.info("   setting output to L1C reference image acquisition, for each polarization.")

        # Check the presence of HH, XP and VV polarizations (one of three can be None)
        pol_is_present_list = []

        for scs_acq_list in scs_pol_list:
            if scs_acq_list is None:
                pol_is_present_list.append(False)
            else:
                pol_is_present_list.append(True)

        for idx_pol, (pol_is_present, scs_acq_list) in enumerate(
            zip(
                pol_is_present_list,
                scs_pol_list,
            )
        ):
            if pol_is_present:
                ground_cancelled_list.append(scs_acq_list[primary_image_index])
            else:
                ground_cancelled_list.append(None)

        warnings.simplefilter("always")
        return ground_cancelled_list, idx_reference_to_save


def ground_cancellation_core(
    scs_acq_list: list[np.ndarray],
    vertical_wavenumber_list: list[np.ndarray],
    kz0: float,
    idx_reference: int,
    demodulation_height: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
     Core function of the ground_cancellation APP:
         it computes the ground cancellation from input data stack acquisitions of a single polarization.

    Parameters
    ----------
    scs_acq_list: List[np.ndarray]
        Stack of calibrated, ground steered slc acquisitions for a single polarization.
        it is a list with M=2 or 3 acquisitions
        each of dimensions [N_az x N_rg]
    vertical_wavenumber_list: List[np.ndarray]
        list of M vertical wavenumbers, each of shape [N_az x N_rg]
    kz0:float
        emphasized forest height converted to wavenumber (desired phase-to-height)
    idx_reference: int
        Index of the scs_acq_list and vertical_wavenumber_list, corresponding to the reference image
    demodulation_height: float
        Vertical spectrum demodulation height to perform interpolation

    Returns
    -------
    ground_cancelled: np.ndarray
        [Naz x Nrg] ground cancelled slc image.
    mask_extrap: np.ndarray
        [Naz x Nrg] logical mask: it is true if the desired phase-to-height is out of the available range.
    """

    # Demodulation
    for idx, data in enumerate(scs_acq_list):
        scs_acq_list[idx] = data * np.exp(-1j * vertical_wavenumber_list[idx] * demodulation_height)

    # Generating synthetic SLC through interpolation
    Ikz0, mask_extrap = kzInterp(scs_acq_list, vertical_wavenumber_list, kz0)

    # Modulation
    for idx, data in enumerate(scs_acq_list):
        scs_acq_list[idx] = data * np.exp(1j * vertical_wavenumber_list[idx] * demodulation_height)

    Ikz0 = Ikz0 * np.exp(1j * kz0 * demodulation_height)

    ground_cancelled = scs_acq_list[idx_reference] - Ikz0

    return ground_cancelled, mask_extrap


def kzInterp(
    scs_acq_list_in: list[np.ndarray],
    vertical_wavenumber_list_in: list[np.ndarray],
    kz0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    It generates a synthetic SLC stack of acquisitions by interpolating the
    original stack of SLC acquisitions defined over the kz axis specified by
    kz_stack in correspondence of the desired phase-to-height.

    Parameters
    ----------
    scs_acq_list: List[np.ndarray]
        Stack of calibrated, ground steered and demodulated slc acquisitions for a single polarization.
        it is a list with M=2 or 3 acquisitions
        each of dimensions [N_az x N_rg]
    vertical_wavenumber_list: List[np.ndarray]
        list of M vertical wavenumbers, each of shape [N_az x N_rg]
    kz0:float
        emphasized forest height converted to wavenumber (desired phase-to-height)
        with the correct sign, checked with "check_pm_kz0"

    Returns
    -------
    Ikz0: np.ndarray
        [Nrg x Naz] synthetic SLC, to be demdulated
    mask_extrap: np.ndarray
        [Naz x Nrg] logical mask: it is true if the desired phase-to-height is out of the available range.
    """

    num_acq = len(scs_acq_list_in)
    Naz, Nrg = scs_acq_list_in[0].shape

    # convert input List in a 3d matrix,
    # in order to have better performances in the computatios that follows
    scs_acq_matrix = np.zeros((Naz, Nrg, num_acq), dtype=np.complex64)
    vertical_wavenumber_matrix = np.zeros((Naz, Nrg, num_acq))
    for idx_acq, (scs_data, vertical_wavenumber) in enumerate(zip(scs_acq_list_in, vertical_wavenumber_list_in)):
        scs_acq_matrix[:, :, idx_acq] = scs_data
        vertical_wavenumber_matrix[:, :, idx_acq] = vertical_wavenumber
    del scs_acq_list_in, vertical_wavenumber_list_in

    Nr, Nc, N = scs_acq_matrix.shape

    # Linear interpolation
    pre_kz_ind = np.zeros((Nr, Nc), dtype=np.int8)
    post_kz_ind = np.zeros((Nr, Nc), dtype=np.int8)
    pre_kz_abs_diff = np.zeros((Nr, Nc)) + np.inf
    post_kz_abs_diff = np.zeros((Nr, Nc)) + np.inf

    for n in range(N):
        curr_kz_diff = vertical_wavenumber_matrix[:, :, n] - kz0
        curr_kz_abs_diff = np.abs(curr_kz_diff)

        pre_kz_mask = curr_kz_diff < 0
        post_kz_mask = np.logical_not(pre_kz_mask)

        # To Be Replaced
        pre_tbr = (np.abs(curr_kz_diff) < pre_kz_abs_diff) & pre_kz_mask
        post_tbr = (np.abs(curr_kz_diff) < post_kz_abs_diff) & post_kz_mask

        pre_kz_ind[pre_tbr] = n
        post_kz_ind[post_tbr] = n

        pre_kz_abs_diff[pre_tbr] = curr_kz_abs_diff[pre_tbr]
        post_kz_abs_diff[post_tbr] = curr_kz_abs_diff[post_tbr]

    # Desired vertical_wavenumber_matrix out of range (to be extrapolated)
    pre_tbe = np.isinf(pre_kz_abs_diff)
    post_tbe = np.isinf(post_kz_abs_diff)

    pre_kz_ind[pre_tbe] = 0
    post_kz_ind[post_tbe] = N - 1

    [C, R] = np.meshgrid(np.arange(Nc), np.arange(Nr))

    kz_pre = vertical_wavenumber_matrix[R, C, pre_kz_ind]
    kz_post = vertical_wavenumber_matrix[R, C, post_kz_ind]
    frac_part = (kz0 - kz_pre) / (kz_post - kz_pre)

    Ikz0 = (1 - frac_part) * scs_acq_matrix[R, C, pre_kz_ind] + frac_part * scs_acq_matrix[R, C, post_kz_ind]

    mask_extrap = pre_tbe | post_tbe

    Ikz0[mask_extrap] = np.spacing(1)

    return Ikz0, mask_extrap


def check_pm_kz0(vertical_wavenumber_list: list[np.ndarray], kz0: float) -> float:
    """
    Check if it is better to generate +kz0 or -kz0

    vertical_wavenumber_list:List[np.ndarray]
        list of M vertical wavenumbers
    kz0:float
        emphasized forest height converted to wavenumber (desired phase-to-height)
    """

    condition = np.nansum(
        (np.nanmax(vertical_wavenumber_list, axis=0) >= kz0) & (np.nanmin(vertical_wavenumber_list, axis=0) <= kz0)
    ) < np.nansum(
        (np.nanmax(vertical_wavenumber_list, axis=0) >= -kz0) & (np.nanmin(vertical_wavenumber_list, axis=0) <= -kz0)
    )

    if condition:
        return -kz0

    else:
        return kz0


def kz0_crit(vertical_wavenumber_list: list[np.ndarray], kz0: float) -> int:
    """
    Computing pixel count needed for reference acquisition selection,
    with the current vertical wavenumbers diff computed with a selected reference

    vertical_wavenumber_list:List[np.ndarray]
        list of M vertical wavenumbers
    kz0:float
        emphasized forest height converted to wavenumber (desired phase-to-height)
    """

    # First  check if it is better to generate +kz0 or -kz0
    kz0 = check_pm_kz0(vertical_wavenumber_list, kz0)

    # get the minimums and maximums of the modified vertical wavenumber return as a 2D array
    min_array = np.nanmin(vertical_wavenumber_list, axis=0)
    max_array = np.nanmax(vertical_wavenumber_list, axis=0)

    # check for how many pixels the kz0 is in between
    num_array = (min_array < kz0) & (kz0 < max_array)
    pixel_count = np.sum(num_array)

    return pixel_count


def reference_acquisition_selection(
    vertical_wavenumber_list: list[np.ndarray],
    kz0: float,
    acquisition_paths_selected_not_sorted: list[Path],
) -> tuple[np.int64, float]:
    bps_logger.info("    reference acquisition selection, guaranteeing the widest scene coverage:")

    # get the minimums and maximums of the vertical wavenumber return as a 2D array
    min_array = np.nanmin(vertical_wavenumber_list, axis=0)  # noqa: F841 (variable is used in the evaluate)
    max_array = np.nanmax(vertical_wavenumber_list, axis=0)  # noqa: F841 (variable is used in the evaluate)
    pixels_count_kz_plus = np.zeros(len(vertical_wavenumber_list))
    pixels_count_kz_minus = np.zeros(len(vertical_wavenumber_list))
    for idx_ref, vertical_wavenumber_ref in enumerate(vertical_wavenumber_list):
        pixels_count_kz_plus[idx_ref] = np.count_nonzero(
            ne.evaluate("((min_array < vertical_wavenumber_ref + kz0) & (vertical_wavenumber_ref + kz0 < max_array))")
        )
        pixels_count_kz_minus[idx_ref] = np.count_nonzero(
            ne.evaluate("((min_array < vertical_wavenumber_ref - kz0) & (vertical_wavenumber_ref - kz0 < max_array))")
        )
        bps_logger.info(
            f"        for reference acquisition {idx_ref}, {acquisition_paths_selected_not_sorted[idx_ref].name}, the number of pixels satisfying the criterion is #{max(pixels_count_kz_plus[idx_ref], pixels_count_kz_minus[idx_ref])} over #{vertical_wavenumber_ref.shape[0] * vertical_wavenumber_ref.shape[1]}"
        )

    idx_reference_plus = np.argmax(pixels_count_kz_plus)
    idx_reference_minus = np.argmax(pixels_count_kz_minus)

    if pixels_count_kz_plus[idx_reference_plus] >= pixels_count_kz_minus[idx_reference_minus]:
        idx_reference = idx_reference_plus

    else:
        idx_reference = idx_reference_minus
        # No more need of "check_pm_kz0" to check if it is better to generate +kz0 or -kz0
        kz0 = -kz0

    bps_logger.info(f"    selected acquisition {idx_reference} as optimal reference SLC")
    bps_logger.info("    performing ground cancellation with the selected reference")

    return idx_reference, kz0
