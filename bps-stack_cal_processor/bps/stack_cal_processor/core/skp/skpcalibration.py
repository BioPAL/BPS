# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Sum of Krokecker Products calibration module (SKP)
------------------------------------------------------
"""

from dataclasses import asdict
from datetime import timedelta
from timeit import default_timer

import numba as nb
import numpy as np
import numpy.typing as npt
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import (
    SKP_NAME,
    StackCalConf,
    StackDataSpecs,
    log_calibration_params,
)
from bps.stack_cal_processor.core.filtering import (
    ConvolutionBorderType,
    build_sparse_uniform_filter_matrix,
)
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_numeric_types_equal,
)
from bps.stack_cal_processor.core.skp.mpmb import (
    assemble_mpmb_coherence_matrix_multithreaded,
)
from bps.stack_cal_processor.core.skp.postprocessing import (
    apply_median_filter_skp_phases_multithreaded,
)
from bps.stack_cal_processor.core.skp.preprocessing import (
    remove_synthetic_phase_multithreaded,
)
from bps.stack_cal_processor.core.skp.skpcorrection import (
    apply_skp_correction_multithreaded,
    upsample_skp_phases_multithreaded,
)
from bps.stack_cal_processor.core.skp.skpdecomposition import (
    compute_skp_calibration_phases_multithreaded,
)
from bps.stack_cal_processor.core.skp.skpquality import (
    SkpFnFQualityMask,
    compute_skp_fnf_quality,
)
from bps.stack_cal_processor.core.skp.utils import (
    compute_skp_calibration_phases_statistics,
    estimation_subsampling_steps,
)
from bps.stack_cal_processor.core.utils import (
    compute_spatial_azimuth_shifts,
    compute_spatial_range_shifts,
    percentage_valid,
)


def skp_calibration(
    *,
    stack: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    vertical_wavenumbers: tuple[npt.NDArray[float], ...],
    conf: StackCalConf.SkpConf,
    stack_specs: StackDataSpecs,
    coreg_primary_image_index: int,
    skp_fnf_mask: SkpFnFQualityMask | None,
    max_num_threads: int,
) -> dict:
    """
    Run the Sum-of-Kronecker-Products calibration (SKP). If the correction
    flag is enabled, the correction is executed in-place.

    Parameters
    ----------
    stack: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    vertical_wavenumbers: tuple[np.NDArray[float], ...] [rad/m]
        The Nimg vertical wavenumbers of shape [Nazm x Nrng].

    conf: StackCalConf.SkpConf
        Configuration of the SKP from the Aux PPS.

    stack_specs: StackDataSpecs
        The stack parameters (e.g. bandwidths, prf's etc.).

    coreg_primary_image_index: int
        The index primary image used for coregistration.

    skp_fnf_mask: Optional[SkpFnFQualityMask]
       Optionally, a Forest-Nonforest mask to compute the qualities.

    max_num_threads: int
        Maximum number of workers that will be used.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    dict
        The SKP calibration products.

    """
    # Start the SKP calibration.
    start_skp = default_timer()
    bps_logger.info("%s started", SKP_NAME)
    log_calibration_params(conf_dict=asdict(conf))

    # Store the stack size.
    num_images = len(stack)
    num_polarizations = len(stack[0])
    num_azimuths, num_ranges = stack[0][0].shape
    bps_logger.info(
        "Stack size: (Nimg=%d, Npol=%d). Image size: (Nazm=%d, Nrng=%d)",
        num_images,
        num_polarizations,
        num_azimuths,
        num_ranges,
    )

    # The overall product quality indices, packed as a [Nimg x 6] bitset
    # matrix. Bit #1 is invalid estimation windows, #2 is for estimation
    # failures, bit #3 is for correction failures, bit #4 is for estimation
    # quality below threshold, and bit #5 is for invalid number of
    # polarizations. Bit #6 is left unassigned.
    quality_bitset = np.zeros((num_images, 6), dtype=np.uint8)

    # The SKP works best with 3 polarizations (i.e. X-pol merging) and it works
    # only with 3 or 4 polarizations. If we have 2 or less (or more than 4), we
    # skip.
    if num_polarizations not in {3, 4}:
        bps_logger.warning("Stack must have 3 or 4 polarizations. Skipped")
        quality_bitset[:, 4] = 1
        quality_bitset[:, 2] = np.uint8(conf.skp_phase_correction_flag)
        return {
            "quality_bitset": quality_bitset,
            "is_solution_usable": False,
            "skp_decomposition_index": 0,
        }

    # Validate the selected decimation factors. These must be at most 90% of
    # the Nyquist and better to be below 50% of it.
    output_azimuth_window = compute_spatial_azimuth_shifts(
        conf.output_azimuth_subsampling_step,
        stack_specs.azimuth_sampling_step,
        ground_speed=stack_specs.satellite_ground_speeds[coreg_primary_image_index],
    )
    azimuth_window_ratio = output_azimuth_window / conf.estimation_window_size
    if conf.nyquist_window_bounds[0] < azimuth_window_ratio <= conf.nyquist_window_bounds[1]:
        quality_bitset[:, 0] = 1
        quality_bitset[:, 2] = np.uint8(conf.skp_phase_correction_flag)
        bps_logger.warning(
            "Selected azimuth LUT decimation factor is large (%.1f%s of the Nyquist)",
            azimuth_window_ratio,
            "%",
        )
    if azimuth_window_ratio > conf.nyquist_window_bounds[1]:
        quality_bitset[:, 0] = 1
        quality_bitset[:, 2] = np.uint8(conf.skp_phase_correction_flag)
        bps_logger.warning("Selected azimuth LUT decimation factor is above 90% of the Nyquist")
        return {
            "quality_bitset": quality_bitset,
            "is_solution_usable": False,
            "skp_decomposition_index": 0,
        }

    output_range_window = compute_spatial_range_shifts(
        conf.output_range_subsampling_step,
        stack_specs.range_sampling_step,
        incidence_angle=stack_specs.incidence_angles[coreg_primary_image_index],
    )
    range_window_ratio = output_range_window / conf.estimation_window_size
    if conf.nyquist_window_bounds[0] < range_window_ratio <= conf.nyquist_window_bounds[1]:
        quality_bitset[:, 0] = 1
        quality_bitset[:, 2] = np.uint8(conf.skp_phase_correction_flag)
        bps_logger.warning(
            "Selected range LUT decimation factor is large (%.1f%s of the Nyquist)",
            range_window_ratio,
            "%",
        )
    if range_window_ratio > conf.nyquist_window_bounds[1]:
        quality_bitset[:, 0] = 1
        quality_bitset[:, 2] = np.uint8(conf.skp_phase_correction_flag)
        bps_logger.warning(
            "Selected range LUT decimation factor is above %.1f%s of the Nyquist",
            100 * conf.nyquist_window_bounds[1],
            "%",
        )
        return {
            "quality_bitset": quality_bitset,
            "is_solution_usable": False,
            "skp_decomposition_index": 0,
        }

    # Cap the number of threads available to numba.
    nb.set_num_threads(max_num_threads)

    # The estimation precision.
    estimation_dtypes = EstimationDType.from_32bit_flag(use_32bit_flag=conf.use_32bit_precision)
    bps_logger.info(
        "Using %s for estimating the interferometric models",
        estimation_dtypes,
    )

    # Run the preprocessing step (multithreaded).
    bps_logger.info("Removing the geometric phases")

    preproc_images = remove_synthetic_phase_multithreaded(
        stack_images=stack,
        synth_phases=synth_phases,
        num_worker_threads=max_num_threads,
        dtypes=estimation_dtypes,
    )

    # Run the SKP decomposition on each interferometric pair.
    bps_logger.info("Computing the MPMB coherences")

    # Reference incidence angle and satellite ground speed.
    incidence_angle_coreg_primary = stack_specs.incidence_angles[coreg_primary_image_index]
    satellite_ground_speed_coreg_primary = stack_specs.satellite_ground_speeds[coreg_primary_image_index]

    # Construct the azimuth and range filtering matrices for the uniform filter w/
    # subsampling. These are common for all interferometric pairs.
    azimuth_subsampling_step, range_subsampling_step = estimation_subsampling_steps(
        satellite_ground_speed=satellite_ground_speed_coreg_primary,
        azimuth_sampling_step=stack_specs.azimuth_sampling_step,
        range_sampling_step=stack_specs.range_sampling_step,
        window_size=conf.estimation_window_size,
        incidence_angle=incidence_angle_coreg_primary,
    )

    (
        azimuth_filter_matrix,
        azimuth_estimation_indices,
    ) = build_sparse_uniform_filter_matrix(
        input_size=num_azimuths,
        axis=0,
        subsampling_step=azimuth_subsampling_step,
        uniform_filter_window_size=2 * azimuth_subsampling_step + 1,
        border_type=ConvolutionBorderType.ISOLATED,
        dtype=estimation_dtypes.float_dtype,
    )
    (
        range_filter_matrix,
        range_estimation_indices,
    ) = build_sparse_uniform_filter_matrix(
        input_size=num_ranges,
        axis=1,
        subsampling_step=range_subsampling_step,
        uniform_filter_window_size=2 * range_subsampling_step + 1,
        border_type=ConvolutionBorderType.ISOLATED,
        dtype=estimation_dtypes.float_dtype,
    )

    mpmb_coherences, mpmb_ok = assemble_mpmb_coherence_matrix_multithreaded(
        images=preproc_images,
        azimuth_filter_matrix=azimuth_filter_matrix,
        range_filter_matrix=range_filter_matrix,
        exclude_polarization_cross_cov=conf.exclude_mpmb_polarization_cross_covariance_flag,
        dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    bps_logger.info("Running the SKP decomposition")

    # The subsampled vertical wavenumbers. We store this as a 3D numpy array
    # for efficiency.
    # NOTE: There is no significant speed-up if axes are rearranged.
    subsampled_vertical_wavenumbers = np.array(
        [vwn[azimuth_estimation_indices, range_estimation_indices] for vwn in vertical_wavenumbers]
    )

    # Store the calibration phases.
    subsampled_calibration_phases, errors = compute_skp_calibration_phases_multithreaded(
        mpmb_coherences=mpmb_coherences,
        vertical_wavenumbers=subsampled_vertical_wavenumbers,
        coreg_primary_image_index=coreg_primary_image_index,
        azimuth_estimation_indices=azimuth_estimation_indices,
        range_estimation_indices=range_estimation_indices,
        num_images=num_images,
        num_polarizations=num_polarizations,
        num_worker_threads=max_num_threads,
        dtypes=estimation_dtypes,
    )
    assert_numeric_types_equal(subsampled_calibration_phases, expected_dtype=estimation_dtypes.float_dtype)

    # Update the quality bitset.
    quality_bitset[:, 1] = np.uint8(errors)

    # NOTE: Internal scipy functions do not allow for using float32, so from
    # now on all is compute in double precision, irrespectively on the user
    # selection. Anyways, from this point on, the algorithm is quite cheap
    # memory-wise.

    # The original axes. These axis can be relative.
    azimuth_axis = np.arange(num_azimuths) * stack_specs.azimuth_sampling_step
    range_axis = np.arange(num_ranges) * stack_specs.range_sampling_step

    # Compute the groundPhasesScreen, that is, the phase screen due to ground
    # and forest.
    #
    # Since the SKP phase is defined on a lower resolution grid, we filter the DSI
    # using the same filtering parameters as for the wavenumbers etc.
    bps_logger.info("Assembling the SKP ground correction phases")
    (
        skp_calibration_phases,
        skp_flattening_phases,
        azimuth_output_indices,
        range_output_indices,
    ) = upsample_skp_phases_multithreaded(
        synth_phases=synth_phases,
        skp_calibration_phases=subsampled_calibration_phases,
        azimuth_axis=azimuth_axis,
        range_axis=range_axis,
        azimuth_estimation_axis=azimuth_axis[azimuth_estimation_indices],
        range_estimation_axis=range_axis[range_estimation_indices],
        output_azimuth_subsampling_step=conf.output_azimuth_subsampling_step,
        output_range_subsampling_step=conf.output_range_subsampling_step,
        dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    if conf.median_filter_flag:
        bps_logger.info("Apply median filter to estimated SKP phases")
        skp_calibration_phases = apply_median_filter_skp_phases_multithreaded(
            skp_calibration_phases,
            azimuth_sampling_step=stack_specs.azimuth_sampling_step,
            azimuth_subsampling_step=conf.output_azimuth_subsampling_step,
            range_sampling_step=stack_specs.range_sampling_step,
            range_subsampling_step=conf.output_range_subsampling_step,
            satellite_ground_speed=satellite_ground_speed_coreg_primary,
            incidence_angle=incidence_angle_coreg_primary,
            filter_window_size=conf.median_filter_window_size,
            dtypes=estimation_dtypes,
            num_worker_threads=max_num_threads,
        )

    # Computing quality and valid ratio.
    if skp_fnf_mask is None:
        bps_logger.warning("No FNF mask provided, setting all qualities to 1.0")
        skp_calibration_quality = np.full_like(skp_calibration_phases[0], 1.0)
    else:
        bps_logger.info("Extracting qualities from FNF mask")
        skp_calibration_quality = compute_skp_fnf_quality(
            skp_fnf_mask=skp_fnf_mask,
            skp_azimuth_axis=azimuth_axis[azimuth_output_indices],
            skp_range_axis=range_axis[range_output_indices],
        )

    # The validity mask.
    skp_calibration_quality_mask = skp_calibration_quality >= conf.skp_calibration_phase_screen_quality_threshold

    # Analyze the estimation quality.
    percentage_valid_estimation = percentage_valid(skp_calibration_quality_mask)
    invalid_ratio = 1 - percentage_valid_estimation / 100
    overall_validity = (percentage_valid_estimation / 100) >= conf.overall_product_quality_threshold
    quality_bitset[:, 3] = np.uint8(not overall_validity)
    if not np.any(overall_validity):
        bps_logger.warning("Estimated SKP phases are not valid (<= {:.2f} {}").format(
            conf.overall_product_quality_threshold, "%s"
        )

    # Optionally, correct the stack images.
    if conf.skp_phase_correction_flag or conf.only_flattening_phase_correction_flag:
        bps_logger.info(
            "Applying the %s correction to the stack",
            "DSI" if conf.only_flattening_phase_correction_flag else "DSI+SKP",
        )

        if conf.only_flattening_phase_correction_flag:
            skp_calibration_correction_phases = np.zeros_like(skp_calibration_phases)
        else:
            skp_calibration_correction_phases = np.asarray(skp_calibration_phases) * skp_calibration_quality_mask
        apply_skp_correction_multithreaded(
            stack=stack,
            skp_flattening_phases=skp_flattening_phases,
            skp_calibration_phases=skp_calibration_correction_phases,
            azimuth_axis=azimuth_axis,
            range_axis=range_axis,
            azimuth_subsampling_indices=azimuth_output_indices,
            range_subsampling_indices=range_output_indices,
            quality_threshold=conf.skp_calibration_phase_screen_quality_threshold,
            num_worker_threads=max_num_threads,
            dtypes=estimation_dtypes,
        )

    # If requested to correct the SKP phase screen, possibly flag the quality
    # bitset if something went wrong in the estimation.
    if conf.skp_phase_correction_flag:
        bps_logger.info(
            "Estimated phase-screen above quality threshold [%s]: %.2f",
            "%",
            percentage_valid_estimation,
        )
        quality_bitset[:, 2] = np.uint8(errors)

    end_skp = default_timer()
    bps_logger.info(
        "%s completed. Elapsed time [h:mm:ss]: %s",
        SKP_NAME,
        timedelta(seconds=end_skp - start_skp),
    )

    return {
        "skp_flattening_phase_screen": skp_flattening_phases,
        "skp_azimuth_axis": azimuth_axis[azimuth_output_indices],
        "skp_range_axis": range_axis[range_output_indices],
        "skp_calibration_phase_screen": skp_calibration_phases,
        "skp_calibration_phase_screen_quality": skp_calibration_quality,
        "skp_calibration_phases_statistics": compute_skp_calibration_phases_statistics(skp_calibration_phases),
        "invalid_skp_calibration_phase_screen_ratio": invalid_ratio,
        "skp_decomposition_index": int(mpmb_ok & bool(errors)),
        "quality_bitset": quality_bitset,
        "is_solution_usable": True,
    }
