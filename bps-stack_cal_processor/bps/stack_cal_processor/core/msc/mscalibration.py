# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Multi-Squint Calibration Module
-----------------------------------
"""

from dataclasses import asdict
from datetime import timedelta
from timeit import default_timer

import numba as nb
import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import (
    MSC_NAME,
    StackCalConf,
    StackDataSpecs,
    log_calibration_params,
)
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.interferometric_pairing import sequential_baseline_interferometric_pair_indices
from bps.stack_cal_processor.core.msc.flattening import (
    apply_inplace_dsi_multithreaded,
    compensate_inplace_dsi_multithreaded,
)
from bps.stack_cal_processor.core.msc.ionocorrection import ionosphere_inplace_correction_multithreaded
from bps.stack_cal_processor.core.msc.ionoestimation import (
    IterativeInversionConf,
    LayerEstimationConf,
    ModelInversionConf,
    apply_model_inversion_multithreaded,
)
from bps.stack_cal_processor.core.msc.mscoherence import (
    MultiSquintCoherenceEstimationConf,
    compute_coherence_improvement,
    compute_multisquint_coherence_multithreaded,
)
from bps.stack_cal_processor.core.msc.msshifts import (
    apply_inplace_centroid_correction,
    compute_azimuth_shifts,
)
from bps.stack_cal_processor.core.msc.phaseplane import (
    estimate_phase_slopes_pairwise,
    phase_plane_inplace_compensation,
)
from bps.stack_cal_processor.core.msc.utils import (
    azimuth_meters_to_pixels,
    coherence_window_size_px,
    compute_azimuth_space_axis,
    compute_azimuth_space_resolution,
    compute_slant_range_space_axis,
    nanmean,
    run_multi_squint_iono_correction,
)
from bps.stack_cal_processor.core.msc.warping import apply_constant_azimuth_shifts_interp_inplace_multithreaded


def multi_squint_calibration(
    *,
    stack: tuple[tuple[npt.NDArray[complex], ...], ...],
    stack_noshifts: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    conf: StackCalConf.MscConf,
    stack_specs: StackDataSpecs,
    coreg_primary_image_index: int,
    max_num_threads: int,
) -> dict:
    """
    Run the Multi-Squint Calibration Module in-place.

    This module performs sequential calibration using two input stacks:
    (1) a stack coregistered in both geometry and data, and
    (2) a stack coregistered in geometry and data in the slant-range direction,
        and in geometry only in the azimuth direction (no azimuth shifts applied).

    The processing first compensates the flattening phase screens from both stacks,
    then computes the multi-squint coherence and evaluates it against the expected
    coherence level. If the measured coherence exceeds the expectation, a linear
    centroid correction is estimated and the corresponding image shift is applied.
    A model inversion is then performed to estimate the ionospheric layers and the
    associated phase screen. The ionospheric contribution is compensated using
    either single-layer correction or sub-band refocusing.

    Finally, a residual phase plane is estimated and removed to complete the
    calibration refinement.

    Parameters
    ----------
    stack: tuple[tuple[npt.NDArray[complex], ...], ...]
        A stack of Nimg frames of shape [Naz x Nrg]. This stack is obtained
        by coregistering both along-track and slant-range direction with speckle
        tracking.

    stack_noshifts: tuple[tuple[npt.NDArray[complex], ...], ...]
        A stack of Nimg frames of shape [Naz x Nrg]. This stack is obtained
        by coregistering the slant-range direction with speckle tracking and
        the along-track direction with orbit info/geometry only.

    synth_phases: tuple[npt.NDArray[float], ...]  [rad]
        The synthetic interferogram from the DEM.

    conf: StackCalConf.MscConf
        Configuration of the MSC method from the Aux PPS.

    stack_specs: StackDataSpecs
        The stack parameters (e.g. bandwidths, PRF's etc.).

    coreg_primary_image_index: int
        The index primary image used for coregistration.

    max_num_threads: int
        Maximum number of workers that will be used.

    Raises
    ------
    MscRuntimeError

    Return
    ------
    dict
        The MSC calibration products.

    """
    # Start the MSC calibration.
    start_msc = default_timer()
    bps_logger.info("%s started", MSC_NAME)
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

    # Cap the number of threads available to numba.
    nb.set_num_threads(max_num_threads)

    # The estimation and correction precision. The BPS allows for computing all
    # estimations in single or double precision.
    estimation_dtypes = EstimationDType.from_32bit_flag(
        use_32bit_flag=conf.use_32bit_precision_estimation,
    )
    bps_logger.info(
        "Using %s for estimating the interferometric models",
        estimation_dtypes,
    )
    correction_dtypes = EstimationDType.from_32bit_flag(
        use_32bit_flag=conf.use_32bit_precision_correction,
    )
    bps_logger.info(
        "Using %s for calibrating the stack",
        correction_dtypes,
    )

    # As first step, we need to compensate the stacks.
    bps_logger.info("Compensating the flattening phase screens (DSI)")
    compensate_inplace_dsi_multithreaded(
        stack_images=stack,
        synth_phases=synth_phases,
        max_num_threads=max_num_threads,
    )
    compensate_inplace_dsi_multithreaded(
        stack_images=stack_noshifts,
        synth_phases=synth_phases,
        max_num_threads=max_num_threads,
    )

    # Cache the azimuth and slant-range space axes.
    azimuth_space_axis = compute_azimuth_space_axis(
        azimuth_sampling_step=stack_specs.azimuth_sampling_step,
        num_azimuths=num_azimuths,
        satellite_ground_speed=stack_specs.satellite_ground_speeds[coreg_primary_image_index],
        dtype=estimation_dtypes.float_dtype,
    )
    slant_range_space_axis = compute_slant_range_space_axis(
        range_sampling_start=stack_specs.range_sampling_starts[coreg_primary_image_index],
        range_sampling_step=stack_specs.range_sampling_step,
        num_ranges=num_ranges,
        dtype=estimation_dtypes.float_dtype,
    )

    # Only sequential baseline pairing is used in the MSC module.
    interferometric_pair_indices = sequential_baseline_interferometric_pair_indices(
        num_images=num_images,
        reference_image_index=coreg_primary_image_index,
        flatten=True,
    )

    # Caching few quantities.
    wavelength = sp.constants.speed_of_light / stack_specs.central_frequency
    satellite_ground_speed = stack_specs.satellite_ground_speeds[coreg_primary_image_index]
    azimuth_space_resolution = compute_azimuth_space_resolution(
        azimuth_space_step=np.diff(azimuth_space_axis[:2])[0],
        azimuth_sampling_step=stack_specs.azimuth_sampling_step,
        azimuth_bandwidth=stack_specs.azimuth_bandwidths[coreg_primary_image_index],
    )

    # Storage for the estimation output.
    coherence_checks = np.full((num_images,), None)
    low_resolution_switches = np.full((num_images,), False)
    az_phase_slopes = np.zeros((num_images,), dtype=estimation_dtypes.float_dtype)  # [rad/px].
    rg_phase_slopes = np.zeros((num_images,), dtype=estimation_dtypes.float_dtype)  # [rad/px].
    ms_shift_outputs = np.full((num_images,), None)
    iono_estimation_outputs = np.full((num_images,), None)
    ionosphere_slant_range_space_axes = np.full((num_images,), None)

    for index_p, index_s in interferometric_pair_indices:
        bps_logger.info("Calibrating secondary=%d vs primary=%d", index_s, index_p)

        # Compute the Multi-Squint coherence on the stack with only geometrical
        # coregistration in azimuth direction.
        bps_logger.info("Computing the MS-coherence matrix")
        ms_coherence_conf = MultiSquintCoherenceEstimationConf.from_msc_high_res_ms_coherence_conf(
            conf, azimuth_space_resolution
        )
        ms_coherence = compute_multisquint_coherence_multithreaded(
            image_p=stack[index_p][conf.polarization_index],
            image_s=stack_noshifts[index_s][conf.polarization_index],
            azimuth_space_axis=azimuth_space_axis,
            slant_range_space_axis=slant_range_space_axis,
            ms_coherence_conf=ms_coherence_conf,
            dtypes=estimation_dtypes,
            max_num_threads=max_num_threads,
        )
        bps_logger.debug("MS coherence shape (Naz,Nrg,Nf): (%d,%d,%d)", *ms_coherence.matrix.shape)

        # Check the coherence improvement wrt to the stack obtained by
        # coregistering via speckle-tracking also in alont-track direction. This
        # is used to decide whether to perform or not the Multi-Squint
        # calibration.
        bps_logger.info("Check the coherence improvement")
        coherence_checks[index_s] = compute_coherence_improvement(
            image_p=stack[index_p][conf.polarization_index],
            image_s=stack[index_s][conf.polarization_index],
            ms_coherence=ms_coherence,
            ms_coh_validity_threshold=conf.ms_coherence_validity_threshold,
            coh_window_size_px=coherence_window_size_px(
                (conf.coherence_azimuth_window_size, conf.coherence_range_window_size),
                azimuth_space_step=np.diff(azimuth_space_axis)[:2][0],
                slant_range_space_step=np.diff(slant_range_space_axis)[:2][0],
            ),
        )
        if run_multi_squint_iono_correction(
            coherence_improvement=coherence_checks[index_s].improvement,
            coherence_improvement_threshold=conf.ms_coherence_improvement_threshold,
            force_multi_squint_iono_correction=conf.force_multi_squint_iono_correction,
            disable_multi_squint_iono_correction=conf.disable_multi_squint_iono_correction,
        ):
            # Apply the linear controid correction.
            bps_logger.info("Estimating the linear centroid correction")
            ms_shift_outputs[index_s] = compute_azimuth_shifts(
                ms_coherence=ms_coherence,
                azimuth_space_axis=azimuth_space_axis,
                wavelength=wavelength,
            )
            bps_logger.info(
                "Applying constant azimuth shift [m]: %.3f",
                ms_shift_outputs[index_s].azimuth_shift,
            )
            apply_constant_azimuth_shifts_interp_inplace_multithreaded(
                mpol_image=stack_noshifts[index_s],
                azimuth_residual_shift=azimuth_meters_to_pixels(
                    ms_shift_outputs[index_s].azimuth_shift,
                    azimuth_sampling_step=stack_specs.azimuth_sampling_step,
                    satellite_ground_speed=satellite_ground_speed,
                ),
                flattening_phase_screen=synth_phases[index_s],  # Base-banding.
            )
            bps_logger.info(
                "Applying the centroid correction [1/m]: %.6f",
                ms_shift_outputs[index_s].azimuth_frequency_centroid,
            )
            apply_inplace_centroid_correction(
                mpol_image=stack_noshifts[index_s],
                azimuth_frequency_centroid=ms_shift_outputs[index_s].azimuth_frequency_centroid,
                azimuth_space_axis=azimuth_space_axis,
            )

            # Possibly, re-compute the MS-coherence matrix at a lower resolution.
            layer_estimation_conf = LayerEstimationConf.from_msc_high_resolution_conf(conf)
            low_resolution_switches[index_s] = (
                nanmean(coherence_checks[index_s].average_ms_coherence) < conf.ms_coherence_low_res_threshold
            )
            if low_resolution_switches[index_s]:
                layer_estimation_conf = LayerEstimationConf.from_msc_low_resolution_conf(conf)
                ms_coherence_conf = MultiSquintCoherenceEstimationConf.from_msc_low_res_ms_coherence_conf(
                    conf, azimuth_space_resolution
                )

            # Re-compute the MS coherence.
            bps_logger.info(
                "Re-compute the Multi-Squint coherence%s",
                "at lower resolution" if low_resolution_switches[index_s] else "",
            )
            ms_coherence = compute_multisquint_coherence_multithreaded(
                image_p=stack[index_p][conf.polarization_index],
                image_s=stack_noshifts[index_s][conf.polarization_index],
                azimuth_space_axis=azimuth_space_axis,
                slant_range_space_axis=slant_range_space_axis,
                ms_coherence_conf=ms_coherence_conf,
                dtypes=estimation_dtypes,
                max_num_threads=max_num_threads,
            )

            # Run the multi-squint correction.
            bps_logger.info("Running the Multi-Squint ionospheric correction")
            iono_estimation_outputs[index_s] = apply_model_inversion_multithreaded(
                ms_coherence=ms_coherence,
                azimuth_space_axis=azimuth_space_axis,
                wavelength=wavelength,
                model_inversion_conf=ModelInversionConf.from_msc_conf(conf),
                iterative_inversion_iono_ranges_conf=IterativeInversionConf.from_msc_iono_range_estimation_conf(conf),
                iterative_inversion_seed_conf=IterativeInversionConf.from_msc_iono_range_seeding_conf(conf),
                iterative_inversion_propagate_conf=IterativeInversionConf.from_msc_iono_range_propagation_conf(conf),
                layer_estimation_conf=layer_estimation_conf,
                max_num_threads=max_num_threads,
            )
            ionosphere_inplace_correction_multithreaded(
                mpol_image=stack_noshifts[index_s],
                ionosphere_estimation_output=iono_estimation_outputs[index_s],
                azimuth_space_resolution=azimuth_space_resolution,
                azimuth_space_axis=azimuth_space_axis,
                ms_azimuth_indices=ms_coherence.azimuth_indices,
                slant_range_space_axis=slant_range_space_axis,
                ms_range_indices=ms_coherence.range_indices,
                subband_axis=ms_coherence.subband_axis,
                subband_azimuth_resolution=conf.ms_coherence_subband_azimuth_resolution,
                wavelength=wavelength,
                dtypes=correction_dtypes,
                max_num_threads=max_num_threads,
            )

            # Store the ionosphere space axes in slant-range direction for annotations.
            ionosphere_slant_range_space_axes[index_s] = slant_range_space_axis[ms_coherence.range_indices]

            # Update the stack with the iono-calibrated image.
            for image, image_iono_comp in zip(stack[index_s], stack_noshifts[index_s]):
                image[...] = image_iono_comp

        # Run the phase plane removal (final refinement).
        bps_logger.info("Estimating and the phase plane slopes")
        az_phase_slopes[index_s], rg_phase_slopes[index_s] = estimate_phase_slopes_pairwise(
            image_p=stack[index_p][conf.polarization_index],
            image_s=stack[index_s][conf.polarization_index],
            dtypes=estimation_dtypes,
            coherence_window_size=coherence_window_size_px(
                (conf.coherence_azimuth_window_size, conf.coherence_range_window_size),
                azimuth_space_step=np.diff(azimuth_space_axis)[:2][0],
                slant_range_space_step=np.diff(slant_range_space_axis)[:2][0],
            ),
        )
        bps_logger.info(
            "Removing phase slopes [rad/px]: az=%.5f, rg=%.5f",
            az_phase_slopes[index_s],
            rg_phase_slopes[index_s],
        )
        phase_plane_inplace_compensation(
            mpol_image=stack[index_s],
            phase_slopes=(az_phase_slopes[index_s], rg_phase_slopes[index_s]),
            dtypes=correction_dtypes,
        )

    # Finally, we re-apply the flattening phases screen.
    bps_logger.info("Re-applying the flattening phase screens (DSI) to the stack")
    apply_inplace_dsi_multithreaded(
        stack_images=stack,
        synth_phases=synth_phases,
        max_num_threads=max_num_threads,
    )

    end_msc = default_timer()
    bps_logger.info(
        "%s completed. Elapsed time [h:mm:ss]: %s",
        MSC_NAME,
        timedelta(seconds=end_msc - start_msc),
    )

    return {
        "interferometric_pair_indices": interferometric_pair_indices,
        "low_resolution_switches": low_resolution_switches,
        "coherence_checks": coherence_checks,
        "azimuth_phase_slopes": az_phase_slopes / stack_specs.azimuth_sampling_step,
        "range_phase_slopes": rg_phase_slopes / stack_specs.range_sampling_step,
        "azimuth_shift_outputs": ms_shift_outputs,
        "ionosphere_estimation_outputs": iono_estimation_outputs,
        "ionosphere_slant_range_space_axes": ionosphere_slant_range_space_axes,
        "is_solution_usable": True,
    }
