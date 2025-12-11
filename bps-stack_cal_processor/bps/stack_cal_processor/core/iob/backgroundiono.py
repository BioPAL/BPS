# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Slow-varying Ionosphere Estimation and Correction
-------------------------------------------------
"""

from dataclasses import asdict
from datetime import timedelta
from timeit import default_timer

import numpy as np
import numpy.typing as npt
from arepytools.geometry.conversions import xyz2llh
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import (
    IOB_NAME,
    BaselineMethodType,
    StackCalConf,
    StackDataSpecs,
    log_calibration_params,
)
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.interferometric_pairing import (
    multi_baseline_interferometric_pair_indices,
    single_baseline_interferometric_pair_indices,
)
from bps.stack_cal_processor.core.iob.estimation import (
    combine_pairwise_ionosphere_estimations,
    has_failed_pairwise_estimations,
    pairwise_ionosphere_estimations_multithreaded,
)
from bps.stack_cal_processor.core.iob.ionocorrection import (
    compensate_background_ionosphere_multithreaded,
)
from bps.stack_cal_processor.core.iob.preprocessing import (
    preprocess_input_stack_multithreaded,
)
from bps.stack_cal_processor.core.iob.splitspectrum import compute_split_spectrum_biases
from bps.stack_cal_processor.core.iob.utils import IobRuntimeError


def remove_background_ionosphere(
    *,
    stack: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    vertical_wavenumbers: tuple[npt.NDArray[float], ...],
    range_coreg_shifts: tuple[npt.NDArray[float], ...],
    l1_iono_phases: tuple[npt.NDArray[float] | None, ...],
    l1_iono_shifts: tuple[npt.NDArray[float] | None, ...],
    conf: StackCalConf.IobConf,
    stack_specs: StackDataSpecs,
    calib_reference_image_index: int,
    max_num_threads: int,
) -> dict:
    """
    Run in-place the Slow Ionosphere Removal (IOB) algorithm. The algorithm
    performs the following steps:

    1) Preprocess the two SCS images by whitening the range spectrum and
       removing the synthetic phases from DEM
    2) Compute range sub-looks by selecting bands at the extremities of the range
       spectrum (high/low range sub-looks)
    3) Compute the interferometric coherences associated to the upper and lower
       sub-looks, and their phases
    4) Apply the Split-Spectrum method (see Gomba et al. "Toward
       Operational Compensation of Ionospheric Effects in SAR Interferograms: The
       Split-Spectrum Method", IEEE 2016) on those phases to isolate the dispersive
       component of the ionosphere from the nondispersive one,
    5) Estimate the slow-varying (aka background) components of the ionosphere
       via Least-Square (LS) estimation of a phase plane on the entire swath and
       compensate the input stack accordingly.

    Parameters
    ----------
    stack: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    vertical_wavenumbers: tuple[np.NDArray[float], ...] [rad/m]
        The Nimg vertical wavenumbers of shape [Nazm x Nrng].

    range_coreg_shifts: tuple[npt.NDArray[float], ...] [px]
        The Nimg coregistration range shifts of shape [Nazm x Nrng].

    l1_iono_phases: tuple[npt.NDArray[float] | None, ...]
        Optionally, Nimg ionospheric phases from L1 of shape [Nazm x Nrng].

    l1_iono_shifts: tuple[npt.NDArray[float] | None, ...],
        Optionally, Nimg range shifts corresponding to the L1 ionosphere
        estimations, each of shape [Nazm x Nrng].

    stack_specs: StackDataSpecs
        The stack parameters (e.g. bandwidths, PRF's etc.).

    calib_reference_image_index: int
        The index of the calibration reference image.

    max_num_threads: int
        The maximum number of threads available.

    Raises
    ------
    IobRuntimeError, ValueError

    Return
    ------
    dict
        An execution summary with the following fields:

        - slow_ionosphere_azimuth_phase_screens: np.NDArray[float] [rad/s]
             The [1 x Nimg] ionosphere slopes in azimuth direction.

        - slow_ionosphere_range_phase_screens: np.NDArray[float] [rad/s]
             The [1 x Nimg] ionosphere slopes in range direction.

        - slow_ionosphere_qualities: npt.NDArray[float] [adim]
             The [1 x Nimg] ionosphere quality estimations.

        - interferometric_pairs: tuple[tuple[int, int]]
             The interferometric pairs used for the estimation (both in
             Single-baseline and Multi-baseline)

        - quality_bitset: np.NDArray[np.uint8]
             The [Nimg x 4] bitsets corresponding to the IOB quality index

        - is_solution_usable: bool
             Quick-look flag to check as to whether the solution is usable
             for the processor.

    """
    # Start the IOB calibration.
    start_iob = default_timer()
    bps_logger.info("%s started", IOB_NAME)
    log_calibration_params(conf_dict=asdict(conf))

    # Setup the printing options.
    np.set_printoptions(formatter={"float": "{:0.4f}".format})

    # Just storing the number of ranges. This will be used later.
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

    # The overall product quality indices, packed as a [Nimg x 4] bitset
    # matrix. Bit #1 is for missing LUTs, #2 for estimation quality below
    # threshold, and #3 failed estimation. Bit #4 reports whether the frames
    # are outside of the region affected by slow-varying ionosphere.
    quality_bitset = np.zeros((num_images, 4), dtype=np.uint8)

    # The slow-varying ionosphere may be not significant at low latitudes. If
    # we are observing the world at low latitudes, we just skip.
    reference_position = stack_specs.satellite_positions[calib_reference_image_index]
    if np.abs(xyz2llh(reference_position)[0]) > conf.ionosphere_latitude_threshold:
        quality_bitset[:, 3] = 1
        bps_logger.warning("Calibration reference image is outside of region affected by ionosphere: Skipped")
        return {
            "slow_ionosphere_removal_azimuth_phase_screens": None,
            "slow_ionosphere_removal_range_phase_screens": None,
            "slow_ionosphere_removal_range_phase_qualities": None,
            "interferometric_pairs": None,
            "quality_bitset": quality_bitset,
            "is_solution_usable": False,
        }

    # Check input are available, in case they are not, update the quality
    # bitset accordingly.
    if conf.compensate_l1_iono_phase_screen_flag:
        missing_l1_iono_data = [
            i for i, (ph, dr) in enumerate(zip(l1_iono_phases, l1_iono_shifts)) if ph is None or dr is None
        ]
        quality_bitset[missing_l1_iono_data, 0] = 1
        if len(missing_l1_iono_data) > 0:
            bps_logger.warning("Missing L1 iono data for stack images %s", missing_l1_iono_data)
    else:
        # We simply disable the L1 iono data.
        l1_iono_phases = [None] * num_images
        l1_iono_shifts = [None] * num_images

    # The estimation precision. The BPS allows for computing all estimations in
    # single or double precision. Using double precision for the IOB is anyways
    # discouraged since there is a significant memory overhead whilst the
    # increase in accuracy is negligible.
    estimation_dtypes = EstimationDType.from_32bit_flag(use_32bit_flag=conf.use_32bit_precision)
    bps_logger.info(
        "Using %s for estimating the interferometric models",
        estimation_dtypes,
    )

    # Compute the split-spectrum biases due to coregistration shift and,
    # possibly, the L1 iono correction.
    bps_logger.info(
        "Compute phase biases due to coregistration%s",
        ("" if not conf.compensate_l1_iono_phase_screen_flag else " and L1 iono corrections"),
    )
    split_spectrum_biases = tuple(
        compute_split_spectrum_biases(
            range_coreg_shifts=rg_shifts,
            synth_phases=dsi,
            l1_iono_phases=l1_phi,
            l1_iono_shifts=l1_shifts,
            stack_specs=stack_specs,
        )
        for rg_shifts, dsi, l1_phi, l1_shifts in zip(range_coreg_shifts, synth_phases, l1_iono_phases, l1_iono_shifts)
    )

    # Run the preprocessing step in parallel. In this step we remove the
    # spectral window from the input images and compensated the synthetic
    # phases from DEM. The preprocessing step only uses the polarization
    # selected for the estimations. As a consequence, the pre-processed images
    # is a stack of size [Nimg x Namz x Nrng].
    bps_logger.info("Preprocessing the image frames")
    preproc_stack = preprocess_input_stack_multithreaded(
        stack_images=stack,
        synth_phases=synth_phases,
        reference_polarization_index=conf.polarization_index,
        stack_specs=stack_specs,
        estimation_dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    # Cache all interferometric pairs, defined according to the baseline method
    # employed for calibration.
    if conf.baseline_method == BaselineMethodType.SINGLE_BASELINE:
        interferometric_pair_indices = single_baseline_interferometric_pair_indices(
            num_images, reference_image_index=calib_reference_image_index
        )
    elif conf.baseline_method == BaselineMethodType.MULTI_BASELINE:
        interferometric_pair_indices = multi_baseline_interferometric_pair_indices(
            num_images,
            vertical_wavenumbers=vertical_wavenumbers,
            range_bandwidth=stack_specs.range_bandwidth,
            cb_ratio_threshold=conf.multi_baseline_cb_ratio_threshold,
        )
        bps_logger.debug("Selected MB pairs: %s", interferometric_pair_indices)
    else:
        raise IobRuntimeError(f"Unsupported baseline methond {conf.baseline_method}")

    # Compute the estimations.
    bps_logger.info("Running interferometric-based iono model estimation")
    iob_estimations = pairwise_ionosphere_estimations_multithreaded(
        stack_images=preproc_stack,
        synth_phases=synth_phases,
        split_spectrum_biases=split_spectrum_biases,
        interferometric_pair_indices=interferometric_pair_indices,
        iob_conf=conf,
        stack_specs=stack_specs,
        estimation_dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    # Compute the final phase slopes.
    bps_logger.info(
        "Combine the pairwise interferometric estimations via %s",
        conf.baseline_method.value,
    )
    (
        iono_azimuth_slopes,
        iono_range_slopes,
        iono_qualities,
        interferometric_pair_indices,
    ) = combine_pairwise_ionosphere_estimations(
        iob_estimations=iob_estimations,
        iob_conf=conf,
        calib_reference_index=calib_reference_image_index,
        num_images=num_images,
    )

    # Flag in the quality bit if there are invalid pairwise estimations.
    quality_bitset[has_failed_pairwise_estimations(iob_estimations, calib_reference_image_index), 2] = 1

    # Set the estimations w/ quality below threshold and update the
    # quality bitset accordingly.
    invalid_mask = iono_qualities < conf.quality_threshold
    iono_azimuth_slopes[invalid_mask] = 0.0
    iono_range_slopes[invalid_mask] = 0.0
    quality_bitset[invalid_mask, 1] = 1

    # If all estimated ionosphere slopes are invalid, we just stop here.
    if np.all(invalid_mask):
        bps_logger.warning(
            "All estimated iono phase slopes are below threshold. Skipping %s",
            IOB_NAME,
        )
        return {
            "slow_ionosphere_azimuth_phase_screens": iono_azimuth_slopes,
            "slow_ionosphere_range_phase_screens": iono_range_slopes,
            "slow_ionosphere_qualities": iono_qualities,
            "interferometric_pair": interferometric_pair_indices,
            "quality_bitset": quality_bitset,
            "is_solution_usable": False,
        }

    # If we hit some invalid estimations, we report to the user.
    if np.any(invalid_mask):
        bps_logger.warning(
            "Estimations below quality threshold for images %s. Setting coeffs to 0.0",
            [i for i in range(invalid_mask.size) if invalid_mask[i]],
        )

    bps_logger.info("Azimuth slopes [rad/s]: %s", iono_azimuth_slopes)
    bps_logger.info("Range slopes [rad/s]: %s", iono_range_slopes)
    bps_logger.info("Estimation qualities: %s", iono_qualities)

    # Remove the ionosphere from all images except for the reference image,
    # since that is considered inosphere free (as per definition of
    # multibaseline).

    bps_logger.info("Removing the ionospheric phase screen")
    compensate_background_ionosphere_multithreaded(
        stack_images=stack,
        iono_azimuth_slopes=iono_azimuth_slopes,
        iono_range_slopes=iono_range_slopes,
        stack_specs=stack_specs,
        calib_reference_index=calib_reference_image_index,
        estimation_dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    end_iob = default_timer()
    bps_logger.info(
        "%s completed. Elapsed time [h:mm:ss]: %s",
        IOB_NAME,
        timedelta(seconds=end_iob - start_iob),
    )

    return {
        "slow_ionosphere_azimuth_phase_screens": iono_azimuth_slopes,
        "slow_ionosphere_range_phase_screens": iono_range_slopes,
        "slow_ionosphere_qualities": iono_qualities,
        "interferometric_pairs": interferometric_pair_indices,
        "quality_bitset": quality_bitset,
        "is_solution_usable": True,
    }
