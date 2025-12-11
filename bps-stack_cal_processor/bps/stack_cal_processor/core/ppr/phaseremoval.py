# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Phase Plane Rmoval (PPR)
------------------------
"""

from dataclasses import asdict
from datetime import timedelta
from timeit import default_timer

import numpy as np
import numpy.typing as npt
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import (
    PPR_NAME,
    StackCalConf,
    StackDataSpecs,
    log_calibration_params,
)
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.ppr.correction import compensate_phase_slopes_multithreaded
from bps.stack_cal_processor.core.ppr.estimation import estimate_phase_slopes_multithreaded
from bps.stack_cal_processor.core.ppr.flattening import compensate_flattening_phase_multithreaded


def remove_phase_plane(
    *,
    stack: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    conf: StackCalConf.PprConf,
    stack_specs: StackDataSpecs,
    coreg_primary_image_index: int,
    max_num_threads: int,
) -> dict:
    """
    Estimate a phase plane from interferograms and compensate the stack accordingly.

    Parameters
    ----------
    stack: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...]  [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    conf: StackCalConf.PprConf
        The configurable parameters of the PPR module.

    stack_specs: StackDataSpecs
        The stack parameters (e.g. range/azimuth sampling steps etc.)

    coreg_primary_image_index: int
        The index of the coregistration primary image.

    max_num_threads: int
        The maximum number of threads assigned for the job.

    Return
    ------
    dict
        An execution summary with the following fields:

        - azimuth_phase_screens: npt.NDArray[float]  [rad/s]
              The [1 x Nimg] phase screen slopes in along-track direction.

        - range_phase_screens: npt.NDArray[float]  [rad/s]
              The [1 x Nimg] phase screen slopes in range direction.

        - quality_bitset: npt.NDArray[np.uint8]
              The [Nimg x 2] bisets corresponding to the PPR quality index.

        - is_solution_usable: bool
              This field is always set to True and it is populate for
              consistency with the output of the other modules.

    """
    # Start the IOB calibration.
    start_ppr = default_timer()
    bps_logger.info("%s started", PPR_NAME)
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

    # The estimation precision. The BPS allows for computing all estimations in
    # single or double precision. Using double precision for PPR is anyways
    # discouraged since there is a significant memory overhead whilst the
    # increase in accuracy is negligible.
    estimation_dtypes = EstimationDType.from_32bit_flag(use_32bit_flag=conf.use_32bit_precision)
    bps_logger.info(
        "Using %s for estimating the interferometric models",
        estimation_dtypes,
    )

    # The overall product quality indices, packed as a [Nimg x 2] bitset
    # matrix. As of now, all bits are unassigned.
    quality_bitset = np.zeros((num_images, 2), dtype=np.uint8)

    # Run the preprocessing step in parallel and compensate the synthetic
    # phases from DEM. The pre-processed images is a stack of size
    # [Nimg x Npol x Namz x Nrng].
    bps_logger.info("Compensate flattening phase screens (DSI)")
    dsi_comp_images = compensate_flattening_phase_multithreaded(
        stack_images=stack,
        synth_phases=synth_phases,
        reference_polarization_index=conf.polarization_index,
        coreg_primary_image_index=coreg_primary_image_index,
        dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    # Estimate the phase screen slopes.
    bps_logger.info("Estimate the phase planes")
    phase_slopes_azm, phase_slopes_rng = estimate_phase_slopes_multithreaded(
        images=dsi_comp_images,
        stack_specs=stack_specs,
        coreg_primary_image_index=coreg_primary_image_index,
        fft2_zero_padding_upsampling_factor=conf.fft2_zero_padding_upsampling_factor,
        fft2_peak_window_size=conf.fft2_peak_window_size,
        dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    bps_logger.info("Azimuth slopes [rad/s]: %s", phase_slopes_azm)
    bps_logger.info("Range slopes [rad/s]: %s", phase_slopes_rng)

    # Compensate the stack with the new phase screens.
    bps_logger.info("Compensating the phase slopes")
    compensate_phase_slopes_multithreaded(
        stack_images=stack,
        azimuth_phase_slopes=phase_slopes_azm,
        range_phase_slopes=phase_slopes_rng,
        stack_specs=stack_specs,
        coreg_primary_image_index=coreg_primary_image_index,
        dtypes=estimation_dtypes,
        num_worker_threads=max_num_threads,
    )

    end_ppr = default_timer()
    bps_logger.info(
        "%s completed. Elapsed time [h:mm:ss]: %s",
        PPR_NAME,
        timedelta(seconds=end_ppr - start_ppr),
    )

    return {
        "azimuth_phase_slopes": phase_slopes_azm,
        "range_phase_slopes": phase_slopes_rng,
        "quality_bitset": quality_bitset,
        "is_solution_usable": True,
    }
