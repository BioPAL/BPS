# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Azimuth Spectral Filtering module (AZF)
-------------------------------------------
"""

from dataclasses import asdict
from datetime import timedelta
from timeit import default_timer

import numba as nb
import numpy as np
import numpy.typing as npt
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import (
    AZF_NAME,
    StackCalConf,
    StackDataSpecs,
    log_calibration_params,
)
from bps.stack_cal_processor.core.azf.utils import (
    apply_filter_bank_multithreaded,
    compute_cross_shift_stack,
    compute_filter_banks_multithreaded,
)
from bps.stack_cal_processor.core.floating_precision import EstimationDType

# Below this value we report to user that the overlap is probably too small.
MIN_AZIMUTH_COMMON_BANDWIDTH = 10  # [Hz]


def azimuth_spectral_filtering(
    *,
    stack: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    doppler_centroids: npt.NDArray[float],
    conf: StackCalConf.AzfConf,
    stack_specs: StackDataSpecs,
    coreg_primary_image_index: int,
    max_num_threads: int,
    update_stack_specs: bool,
) -> dict:
    """
    Run in-place the azimuth spectral filtering module (AZF).

    Parameters
    ----------
    stack: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    doppler_centroids: npt.NDArray[float] [Hz]
        The Doppler centroids for each image, packed as a numpy array of
        shape [Nimg x Nrng]

    conf: StackCalConf.AzfConf
        Configuration of the AZF from the aux pps.

    stack_specs: StackDataSpecs
        The stack parameters (e.g. bandwidths, prf's etc.).

    coreg_primary_image_index: int
        The index of the primary image used during coregistration.

    max_num_threads: int
        The maximum number of threads available.

    update_stack_specs: bool
        Downstream of the AZF, the images will have shifted bandwiths. If
        this flag is enabled, the azimuth bandwidths and the azimuth central
        frequency in the stack specs will be updated accordingly.

    Raises
    ------
    AzfRuntimeError

    Return
    ------
    dict
        A dictionary containing the common bandwidth and the central frequency,
        both expressed in [Hz] together with the common spectral window and
        parameter (if applicable).

    """
    # Start the AZF processing.
    start_azf = default_timer()
    bps_logger.info("%s started", AZF_NAME)
    log_calibration_params(conf_dict=asdict(conf))

    # A few parameters.
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

    # Cap the maximum number of threads for numba.
    nb.set_num_threads(max_num_threads)

    # Setup the printing options.
    np.set_printoptions(formatter={"float": "{:0.4f}".format})

    # The estimation precision.
    estimation_dtypes = EstimationDType.from_32bit_flag(use_32bit_flag=conf.use_32bit_precision)
    bps_logger.info(
        "Using %s for estimating the interferometric models",
        estimation_dtypes,
    )

    # The overall product quality indices, packed as a [Nimg x 2] bitset
    # matrix. Both bits are unassigned for now.
    quality_bitset = np.zeros((num_images, 2), dtype=np.int8)

    # First we need to compute the stack's cross shifts and
    # auxiliary values per block. This is a [1 x Nrng] vector.
    bps_logger.info(
        "Computing the stack's cross shifts, and upper/lower spectral bounds",
    )

    frequencies_xshifts, frequencies_high, frequencies_low = compute_cross_shift_stack(
        synth_phases=synth_phases,
        doppler_centroids=doppler_centroids,
        azimuth_sampling_step=stack_specs.azimuth_sampling_step,
        azimuth_bandwidths=stack_specs.azimuth_bandwidths,
        coreg_primary_image_index=coreg_primary_image_index,
        dtypes=estimation_dtypes,
    )

    # Compute the filter banks. These varies depending on image and block.
    bps_logger.info("Computing the filter banks")

    # The compression window of the primary image.
    common_window_type = (
        stack_specs.azimuth_compression_window_types[coreg_primary_image_index]
        if conf.use_primary_spectral_weighting_window_flag
        else conf.window_type
    )
    common_window_param = (
        stack_specs.azimuth_compression_window_parameters[coreg_primary_image_index]
        if conf.use_primary_spectral_weighting_window_flag
        else conf.window_parameter
    )

    # Compute the filter bank that will be applied to the stack images.
    filter_banks = compute_filter_banks_multithreaded(
        doppler_centroids=doppler_centroids,
        frequencies_xshifts=frequencies_xshifts,
        frequencies_high=frequencies_high,
        frequencies_low=frequencies_low,
        azimuth_bandwidths=stack_specs.azimuth_bandwidths,
        azimuth_window_types=stack_specs.azimuth_compression_window_types,
        azimuth_window_params=stack_specs.azimuth_compression_window_parameters,
        common_azimuth_window_type=common_window_type,
        common_azimuth_window_param=common_window_param,
        azimuth_sampling_step=stack_specs.azimuth_sampling_step,
        num_azimuths=num_azimuths,
        num_worker_threads=max_num_threads,
        dtypes=estimation_dtypes,
    )

    # Apply the filter bank to the images.
    bps_logger.info("Applying the filters")

    common_bandwidths, central_frequencies = apply_filter_bank_multithreaded(
        stack_images=stack,
        filter_banks=filter_banks,
        frequencies_high=frequencies_high,
        frequencies_low=frequencies_low,
        azimuth_sampling_step=stack_specs.azimuth_sampling_step,
        num_worker_threads=max_num_threads,
    )

    # Compute the stack's common bandwidth and central frequency.
    common_bandwidth = np.min(common_bandwidths)
    central_frequency = np.mean(central_frequencies)
    if common_bandwidth < MIN_AZIMUTH_COMMON_BANDWIDTH:
        bps_logger.warning(
            "Estimated common azimuth bandwidth is almost zero (%f Hz)",
            common_bandwidth,
        )
    else:
        bps_logger.info(
            "Estimated azimuth common bandwidth [Hz]: %f",
            common_bandwidth,
        )
    bps_logger.info(
        "Estimated azimuth central frequency [Hz]: %f",
        central_frequency,
    )

    # Possibly, update the stack specs to account for the new azimuth
    # bandwidths.
    if update_stack_specs:
        stack_specs.azimuth_central_frequency = central_frequency
        stack_specs.azimuth_bandwidths = [common_bandwidth] * num_images

    end_azf = default_timer()
    bps_logger.info(
        "%s completed. Elapsed time [h:mm:ss]: %s ",
        AZF_NAME,
        timedelta(seconds=end_azf - start_azf),
    )

    return {
        "azimuth_common_bandwidth": common_bandwidth,
        "azimuth_central_frequency": central_frequency,
        "common_window_type": common_window_type,
        "common_window_parameter": common_window_param,
        "quality_bitset": quality_bitset,
        "is_solution_usable": True,
    }
