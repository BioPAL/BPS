# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import StackCalConf
from bps.stack_cal_processor.core.msc.mscoherence import MultiSquintCoherenceEstimationOutput
from bps.stack_cal_processor.core.msc.utils import nanmean
from bps.stack_cal_processor.core.utils import percentage_completed_msg


@dataclass
class IonosphereEstimationOutput:
    """Store the output of the ionosphere estimation via model inversion."""

    azimuth_iono_space_axis: npt.NDArray[float]
    """A [1 x Niaz] array storing the along-track axis associated to the ionosphere [m]."""

    iono_layer_phase_screens: npt.NDArray[float]  # [rad].
    """Array of shape [Niaz x Nrg x Nl] storing the ionospheric phase screens for each inospheric layer [rad]."""

    ms_iono_phases: npt.NDArray[float]  # [rad].
    """Array of shape [Naz x Nrg x Nf] storing the ionosphere phase screens projected on to the MS grid [rad]."""

    ionosphere_layer_ranges: npt.NDArray[float]  # [m].
    """Array of shape [1 x Nl] that stores the LOS ranges (from ground) of the estimated ionospheric layers [m]."""


@dataclass
class ModelInversionConf:
    """Store the ionosphere inversion parameters."""

    seed_range_coherence_threshold: float
    edge_guard: int
    valid_azimuth_threshold: float
    method: Literal["projection", "derivative"]

    @classmethod
    def from_msc_conf(cls, conf: StackCalConf.MscConf) -> Self:
        """Initialize from the MSC configuration object."""
        return cls(
            seed_range_coherence_threshold=conf.seed_range_coherence_threshold,
            edge_guard=conf.edge_guard,
            valid_azimuth_threshold=conf.valid_azimuth_threshold,
            method=conf.propagation_step_inversion_method,
        )


@dataclass
class IterativeInversionConf:
    """Store the parameters for the compositve iterative estimation."""

    max_iterations_derivative: int
    max_iterations_projection: int
    convergence_threshold_derivative: float
    convergence_threshold_projection: float
    max_iono_range_offset: float  # [m].
    max_iono_range: float | None = None  # [m].

    @classmethod
    def from_msc_iono_range_estimation_conf(cls, conf: StackCalConf.MscConf) -> Self:
        """Initialize from MSC configuration for ionosphere range estimation."""
        return cls(
            max_iterations_derivative=conf.iono_range_estimation_max_iterations_derivative,
            max_iterations_projection=conf.iono_range_estimation_max_iterations_projection,
            convergence_threshold_derivative=conf.iono_range_estimation_convergence_threshold_derivative,
            convergence_threshold_projection=conf.iono_range_estimation_convergence_threshold_projection,
            max_iono_range_offset=conf.max_iono_range_offset,
        )

    @classmethod
    def from_msc_iono_range_seeding_conf(cls, conf: StackCalConf.MscConf) -> Self:
        """Initialize from MSC configuration for seeding the ionosphere estimation."""
        return cls(
            max_iterations_derivative=conf.iono_range_seeding_max_iterations_derivative,
            max_iterations_projection=conf.iono_range_seeding_max_iterations_projection,
            convergence_threshold_derivative=conf.iono_range_seeding_convergence_threshold_derivative,
            convergence_threshold_projection=conf.iono_range_seeding_convergence_threshold_projection,
            max_iono_range_offset=conf.max_iono_range_offset,
        )

    @classmethod
    def from_msc_iono_range_propagation_conf(cls, conf: StackCalConf.MscConf) -> Self:
        """Initialize from MSC configuration for the ionosphere range propagation."""
        return cls(
            max_iterations_derivative=conf.iono_range_propagation_max_iterations_derivative,
            max_iterations_projection=conf.iono_range_propagation_max_iterations_projection,
            convergence_threshold_derivative=conf.iono_range_propagation_convergence_threshold_derivative,
            convergence_threshold_projection=conf.iono_range_propagation_convergence_threshold_projection,
            max_iono_range_offset=conf.max_iono_range_offset,
        )


@dataclass
class LayerEstimationConf:
    """Configuration of the layer estimation step."""

    iono_range_candidate_min_range: float  # [m].
    iono_range_candidate_max_range: float  # [m].
    iono_range_candidate_ncandidates: int
    layer_addition_threshold: float
    robust_layer_estimation_nslices: int
    max_num_iono_layers: int

    @classmethod
    def from_msc_high_resolution_conf(cls, conf: StackCalConf.MscConf) -> Self:
        """Initialize from an MSC configuration object."""
        return cls(
            iono_range_candidate_min_range=conf.min_iono_range_candidate,
            iono_range_candidate_max_range=conf.max_iono_range_candidate,
            iono_range_candidate_ncandidates=conf.num_iono_range_candidates,
            layer_addition_threshold=conf.layer_addition_threshold,
            robust_layer_estimation_nslices=conf.robust_layer_estimation_high_res_num_slices,
            max_num_iono_layers=conf.max_num_iono_layers,
        )

    @classmethod
    def from_msc_low_resolution_conf(cls, conf: StackCalConf.MscConf) -> Self:
        """Initialize from an MSC configuration object."""
        return cls(
            iono_range_candidate_min_range=conf.min_iono_range_candidate,
            iono_range_candidate_max_range=conf.max_iono_range_candidate,
            iono_range_candidate_ncandidates=conf.num_iono_range_candidates,
            layer_addition_threshold=conf.layer_addition_threshold,
            robust_layer_estimation_nslices=conf.robust_layer_estimation_low_res_num_slices,
            max_num_iono_layers=conf.max_num_iono_layers,
        )


def apply_model_inversion_multithreaded(
    *,
    ms_coherence: MultiSquintCoherenceEstimationOutput,
    azimuth_space_axis: npt.NDArray[float],
    wavelength: float,
    model_inversion_conf: ModelInversionConf,
    iterative_inversion_iono_ranges_conf: IterativeInversionConf,
    iterative_inversion_seed_conf: IterativeInversionConf,
    iterative_inversion_propagate_conf: IterativeInversionConf,
    layer_estimation_conf: LayerEstimationConf,
    max_num_threads: int = 1,
) -> IonosphereEstimationOutput:
    """
    Apply the model inversion to estimate the ionosphere.

    The module first isolates a few range slices to run a 'Greedy Forward
    Selection' algorithm, identifying the physical heights of ionospheric
    disturbances. It then finds the 'Seed Range'—the line with the highest
    percentage of coherent pixels. Finally, it uses a propagation engine that
    clamps the phase of the current row to the previous one, ensuring the final
    3D phase product is spatially smooth and physically consistent.

    Parameters
    ----------
    ms_coherence: MultiSquintCoherenceEstimationOutput
        The output of the Multi-Squint estimation.

    azimuth_space_axis: npt.NDArray[float] [m]
        The space axis in along-track direction.

    wavelength: float [m]
        The RADAR wavelength.

    model_inversion_conf: ModelInversionConf
        The general configuraiton of the model inversion algorithm

    iterative_inversion_iono_ranges_conf: IterativeInversionConf
        The configuration of the iterative inversion used to estimate
        the ionospheric ranges.

    iterative_inversion_seed_conf: IterativeInversionConf
        The configuration of the iterative inversion used for seeding
        the ionosphere estimation step.

    iterative_inversion_propagate_conf: IterativeInversionConf
        The configuration of the iterative inversion used for propagating
        the ionosphere estimation from seed.

    layer_estimation_conf: LayerEstimationConf
        The configuration of the layer estimation step.

    max_num_threads: int = 1
        The number of threads assigned to the task.

    Return
    ------
    IonosphereEstimationOutput
        The output of the ionosphere estimation via model inversion if
        ionospheric ranges were found. None, otherwise.

    """
    # Compute the average coherence across all squints.
    ms_coherence_mean = nanmean(np.abs(ms_coherence.matrix), axis=-1)

    # Compute the number of valid pixel wrt the average MS coherence, excluding
    # the values next to the borders azimuth borders.
    valid_mask = np.logical_and(
        ms_coherence_mean > model_inversion_conf.seed_range_coherence_threshold,
        ms_coherence_mean < 1.0,
    )
    if model_inversion_conf.edge_guard > 0:
        valid_mask[:, 0 : model_inversion_conf.edge_guard] = False
        valid_mask[:, -model_inversion_conf.edge_guard :] = False

    # Find the range index with the maximum number of valid azimuths. If there are
    # not enough valid lines, we fallback to the central sample.
    valid_azimuth_counts = np.sum(valid_mask, axis=0)
    seed_range = np.argmax(valid_azimuth_counts)
    max_valid_azimuths = valid_azimuth_counts[seed_range]

    use_fallback_seed_range = (
        max_valid_azimuths < model_inversion_conf.valid_azimuth_threshold * ms_coherence.azimuth_indices.size
    )
    if use_fallback_seed_range:
        bps_logger.debug("Falling back to mid range for seeding")
        seed_range = round(ms_coherence.range_indices.size / 2)

    # Select representative slices across the range dimension.
    offset = round(ms_coherence.range_indices.size / layer_estimation_conf.robust_layer_estimation_nslices)
    range_indices = np.full((layer_estimation_conf.robust_layer_estimation_nslices,), seed_range, dtype=np.int32)
    for i in range(1, layer_estimation_conf.robust_layer_estimation_nslices):
        range_indices[i] = (seed_range + i * offset) % ms_coherence.range_indices.size

    # Estimate the ionospheric ranges.
    bps_logger.info(
        "Estimating ionosperic LOS ranges. Seed range: %d, Num range samples: %d",
        seed_range,
        range_indices.size,
    )
    ionospheric_ranges = estimate_ionospheric_ranges_multithreaded(
        ms_coherence_samples=ms_coherence.matrix[:, range_indices, :],
        azimuth_space_axis=azimuth_space_axis[ms_coherence.azimuth_indices],
        squint_axis=(wavelength / 2) * ms_coherence.subband_axis,
        iterative_inversion_conf=iterative_inversion_iono_ranges_conf,
        layer_estimation_conf=layer_estimation_conf,
        max_num_threads=max_num_threads,
    )
    bps_logger.info(
        "Using ionospheric LOS ranges (from ground) [Km]: %s",
        [float(round(r / 1000, 3)) for r in ionospheric_ranges],
    )

    bps_logger.info("Run the sequential ionospheric phase screen estimation")
    return sequential_estimation(
        ms_coherence=ms_coherence,
        azimuth_space_axis=azimuth_space_axis,
        ionospheric_ranges=ionospheric_ranges,
        wavelength=wavelength,
        seed_range=seed_range,
        seed_conf=iterative_inversion_seed_conf,
        propagate_conf=iterative_inversion_propagate_conf,
        method=model_inversion_conf.method,
    )


def sequential_estimation(
    *,
    ms_coherence: MultiSquintCoherenceEstimationOutput,
    azimuth_space_axis: npt.NDArray[float],
    ionospheric_ranges: npt.NDArray[float],
    seed_range: int,
    wavelength: float,
    seed_conf: IterativeInversionConf,
    propagate_conf: IterativeInversionConf,
    method: Literal["derivative", "projection"],
) -> IonosphereEstimationOutput:
    """
    Propagate ionospheric estimation across the range.

    This function initializes the inversion at the seedRange and propagates
    the solution across all range bins, ensuring phase continuity by using
    the previous row's result as a starting point.

    Parameters
    ----------
    ms_coherence: MultiSquintCoherenceEstimationOutput
        The output of the Multi-Squint estimation.

    azimuth_space_axis: npt.NDArray[float] [m]
        The space axis in along-track direction.

    ionospheric_ranges: npt.NDArray[float]  [m]
        The estimated ionospheric ranges.

    wavelength: float  [m]
        The RADAR wavelength

    seed_conf: SequentialEstimationConf
        Parameters of the iterative inversion for the range seeding step.

    propagate_conf: SequentialEstimationConf
        Parameters of the iterative inversion for the range propagation step.

    method: Literal["derivative", "projection"]
        The propagation method.

    Return
    ------
    IonosphereEstimationOutput
        The ionosphere estimation output.

    """
    # The floating-point precision.
    float_dtype = ms_coherence.matrix.real.dtype
    complex_dtype = ms_coherence.matrix.dtype

    # Just a few shortcuts.
    num_layers = len(ionospheric_ranges)

    # The axis that store all the squints.
    squint_axis = (wavelength / 2) * ms_coherence.subband_axis

    # Run the composite iterative inversion.
    bps_logger.info("Run the composite iterative inversion on seed range")
    phi_iono_seed, _, layer_phases_seed, azimuth_iono_space_axis = composite_iterative_inversion(
        ms_coherence_slice=ms_coherence.matrix[:, seed_range, :].T,
        squint_axis=squint_axis,
        azimuth_space_axis=azimuth_space_axis[ms_coherence.azimuth_indices],
        ionospheric_ranges=ionospheric_ranges,
        max_iono_range=seed_conf.max_iono_range,
        max_iono_range_offset=seed_conf.max_iono_range_offset,
        max_iterations_derivative=seed_conf.max_iterations_derivative,
        max_iterations_projection=seed_conf.max_iterations_projection,
        convergence_threshold_derivative=seed_conf.convergence_threshold_derivative,
        convergence_threshold_projection=seed_conf.convergence_threshold_projection,
    )

    # Prepare the output.
    ms_iono_phases = np.zeros(ms_coherence.matrix.shape, dtype=float_dtype)
    layer_phases_map = np.zeros(
        (azimuth_iono_space_axis.size, ms_coherence.matrix.shape[1], num_layers),
        dtype=float_dtype,
    )

    # Store seed results.
    ms_iono_phases[:, seed_range, :] = phi_iono_seed.T
    layer_phases_map[:, seed_range, :] = layer_phases_seed

    # Propagate the results.
    down_indices = np.arange(seed_range - 1, -1, -1)
    up_indices = np.arange(seed_range + 1, ms_coherence.range_indices.size, 1)
    propagation_indices = np.concatenate((down_indices, up_indices))

    bps_logger.info(
        "Propagate the results (upward=%d, downward=%d) using '%s' on %d iono range(s). It will take a while",
        down_indices.size,
        up_indices.size,
        method,
        len(ionospheric_ranges),
    )
    for completed, curr_r in enumerate(propagation_indices, start=1):
        # Determine step direction
        s = np.sign(seed_range - curr_r)
        prev_r = curr_r + s

        # Extract current slice [Nf x Naz].
        current_slice = ms_coherence.matrix[:, curr_r, :].T

        # Use previous result as constraint.
        phi_iono_prev = ms_iono_phases[:, prev_r, :].T

        # Pre-align current row to previous
        residual_data = current_slice * np.exp(-1j * phi_iono_prev, dtype=complex_dtype)
        phi_targ_initial = np.angle(np.mean(residual_data, axis=0))
        residual_data = residual_data * np.exp(-1j * phi_targ_initial, dtype=complex_dtype)

        # Perform delta inversion
        if method == "derivative":
            phi_res, _, layer_res, _ = ionospheric_iterative_inversion_derivative(
                ms_coherence_slice=residual_data,
                squint_axis=squint_axis,
                azimuth_space_axis=azimuth_space_axis[ms_coherence.azimuth_indices],
                ionospheric_ranges=ionospheric_ranges,
                max_iono_range=propagate_conf.max_iono_range,
                max_iono_range_offset=propagate_conf.max_iono_range_offset,
                max_iterations=propagate_conf.max_iterations_derivative,
                convergence_threshold=propagate_conf.convergence_threshold_derivative,
            )
        elif method == "projection":
            phi_res, _, layer_res, _ = ionospheric_iterative_inversion_projection(
                ms_coherence_slice=residual_data,
                squint_axis=squint_axis,
                azimuth_space_axis=azimuth_space_axis[ms_coherence.azimuth_indices],
                ionospheric_ranges=ionospheric_ranges,
                max_iono_range=propagate_conf.max_iono_range,
                max_iono_range_offset=propagate_conf.max_iono_range_offset,
                max_iterations=propagate_conf.max_iterations_projection,
                convergence_threshold=propagate_conf.convergence_threshold_projection,
            )
        else:
            raise ValueError(f"Unsupported sequential estimation method: {method}")

        # Update MSIonoPhases
        ms_iono_phases[:, curr_r, :] = (phi_iono_prev + phi_res).T

        # Update Layer Map
        layer_phases_map[:, curr_r, :] = layer_phases_map[:, prev_r, :] + layer_res

        # Log every 10%.
        if completed % round(propagation_indices.size / 10) == 0:
            bps_logger.info(percentage_completed_msg(completed, total=propagation_indices.size))

    bps_logger.info(percentage_completed_msg(propagation_indices.size, total=propagation_indices.size))

    return IonosphereEstimationOutput(
        azimuth_iono_space_axis=azimuth_iono_space_axis,
        iono_layer_phase_screens=layer_phases_map,
        ms_iono_phases=ms_iono_phases,
        ionosphere_layer_ranges=ionospheric_ranges,
    )


def estimate_ionospheric_ranges_multithreaded(
    *,
    ms_coherence_samples: npt.NDArray[complex],
    azimuth_space_axis: npt.NDArray[float],
    squint_axis: npt.NDArray[float],
    iterative_inversion_conf: IterativeInversionConf,
    layer_estimation_conf: LayerEstimationConf,
    max_num_threads: int = 1,
) -> npt.NDArray[float]:
    """
    Identify the dominant ionospheric ranges.

    Parameters
    ----------
    ms_coherence_samples: npt.NDArray[complex]
        The [Naz x Nrg' x Nf] Multi-Squint coherence matrix sampled at the
        iono ranges.

    azimuth_space_axis: npt.NDArray[float]  [m]
        The [1 x Naz] space axis in along-track direction associated to
        the Multi-Squint coherence matrix.

    squint_axis: npt.NDArray[float]  [m]
        The axis storing all squints of the Multi-Squint coherence.

    iterative_inversion_conf: IterativeInversionConf
        The configuration of the ionosphere inversion algorithm.

    layer_estimation_conf: LayerEstimationConf
        The configuration of the layer estimation.

    max_num_threads: int = 1
        The number of threads assiged for the job.

    Return
    ------
    npt.NDArray[float]  [m]
        The ionospheric ranges associated to the layers.

    """
    # The floating point precision.
    float_dtype = ms_coherence_samples.real.dtype
    complex_dtype = ms_coherence_samples.dtype

    # The ionospheric range candidates.
    iono_range_candidates = np.linspace(
        layer_estimation_conf.iono_range_candidate_min_range,
        layer_estimation_conf.iono_range_candidate_max_range,
        layer_estimation_conf.iono_range_candidate_ncandidates,
        dtype=float_dtype,
    ).tolist()

    # Fixed max-ionospheric range.
    max_iono_range = np.max(iono_range_candidates) + iterative_inversion_conf.max_iono_range_offset

    # Run the iterative selection.
    niter = 0
    best_score = 0.0
    iono_ranges = []

    max_niter = len(iono_range_candidates)
    while len(iono_range_candidates) > 0:
        niter += 1
        scores = np.zeros(len(iono_range_candidates), dtype=float_dtype)

        for i, iono_range in enumerate(iono_range_candidates):
            sorted_iono_ranges = np.sort([*iono_ranges, iono_range])

            # Distribute over the ranges to compute the cumulative score.
            with ThreadPoolExecutor(max_workers=max_num_threads) as executor:
                # Compute the score for a MS coherence slice.
                def composite_iterative_inversion_fn(r):
                    ms_coherence_slice = ms_coherence_samples[:, r, :].T
                    iono_phase, iono_target_phase, *_ = composite_iterative_inversion(
                        ms_coherence_slice=ms_coherence_slice,
                        squint_axis=squint_axis,
                        azimuth_space_axis=azimuth_space_axis,
                        ionospheric_ranges=sorted_iono_ranges,
                        max_iono_range=max_iono_range,
                        max_iono_range_offset=iterative_inversion_conf.max_iono_range_offset,
                        max_iterations_derivative=iterative_inversion_conf.max_iterations_derivative,
                        max_iterations_projection=iterative_inversion_conf.max_iterations_projection,
                        convergence_threshold_derivative=iterative_inversion_conf.convergence_threshold_derivative,
                        convergence_threshold_projection=iterative_inversion_conf.convergence_threshold_projection,
                    )
                    residual = ms_coherence_slice * np.exp(-1j * (iono_phase + iono_target_phase), dtype=complex_dtype)
                    if not np.all(np.isnan(residual)):
                        return nanmean(residual.real)
                    return 0

                # Compute the total score.
                total_score = sum(
                    executor.map(
                        composite_iterative_inversion_fn,
                        range(ms_coherence_samples.shape[1]),
                    )
                )

            scores[i] = total_score / ms_coherence_samples.shape[1]

            if (i + 1) % int(len(iono_range_candidates) / 10) == 0:
                bps_logger.info(percentage_completed_msg(i + 1, total=len(iono_range_candidates)))

        # Select the best candidate.
        argmax_score = np.nanargmax(scores)
        max_score = scores[argmax_score]
        iono_range = iono_range_candidates[argmax_score]

        # Stop if we meet the stopping criterion.
        if max_score < best_score + layer_estimation_conf.layer_addition_threshold:
            best_score = max_score
            bps_logger.info("Maximum gain reached. Stopping the iono range estimation")
            break

        bps_logger.debug(
            "Added iono layer. LOS range [Km]: %.1f, gain: %.4f",
            iono_range / 1e3,
            max_score - best_score,
        )
        best_score = max_score
        iono_ranges = np.sort([*iono_ranges, iono_range]).tolist()
        iono_range_candidates.pop(argmax_score)
        bps_logger.info("Completed iteration=%d", niter)

        if len(iono_ranges) >= layer_estimation_conf.max_num_iono_layers:
            bps_logger.info("Reached maximum number of ionospheric layers")
            break

    if niter == max_niter:
        bps_logger.info("Reached maximum number of iterations %d", max_niter)

    return np.array(iono_ranges)


def composite_iterative_inversion(
    *,
    ms_coherence_slice: npt.NDArray[complex],
    squint_axis: npt.NDArray[float],
    azimuth_space_axis: npt.NDArray[float],
    ionospheric_ranges: npt.NDArray[float],
    max_iono_range: float | None,
    max_iono_range_offset: float,
    max_iterations_derivative: int,
    max_iterations_projection: int,
    convergence_threshold_derivative: float,
    convergence_threshold_projection: float,
) -> tuple[
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
]:
    """Apply the iterative inversion for estimating the ionospheric phase
    screen.

    This function implements a cascading inversion strategy to solve for
    ionospheric phase screens. It first uses a 'Derivative' approach to
    capture the coarse, high-energy phase gradients, then refines the
    residual using a 'Projection'  method.

    Parameters
    ----------
    ms_coherence_slice: npt.NDArray[complex]
        A range slice of the Multi-Squint coherence.

    squint_axis: npt.NDArray[float]  [m]
        The [1 x Nf] axis with all squints.

    azimuth_space_axis: npt.NDArray[float]  [m]
        The space axis in along-track direction associated with the
        Multi-Squint coherence.

    ionospheric_ranges: npt.NDArray[float]  [m]
        The ionosphere ranges.

    max_iono_range: float | None  [m]
        Optionally, the maximum ionosphere range.

    max_iono_range_offset: float
        The offset that is applied to the ionosphere range.

    max_iterations_derivative: int
        Maximum number of iterations used by the 'Derivative'-based method.

    max_iterations_projection: int
        Maximum number of iterations used by the 'Projection'-based method.

    convergence_threshold_derivative: float
        The threshold used as stopping criterion for the
        'Derivative'-based method.

    convergence_threshold_projection: float,
        The threshold used as stopping criterion for the
        'Projection'-based method.

    Return
    ------
    npt.NDArray[float]  [rad]
        Total integrated ionospheric phase re-projected to the original
        squint-azimuth grid.

    npt.NDArray[float]  [rad]
        Residual ground/target phase after ionospheric correction [1 x Na]

    npt.NDArray[float]  [rad]
        Estimated phase screens for each height [Niaz x Nri].

    npt.NDArray[float]  [m]
        The space axis in along-track direction at the ionospheric height.
        This axes exstended the along-track space axis at the ground by the
        squint angle of the satellite.

    """
    # The floating-point precision.
    complex_dtype = ms_coherence_slice.dtype

    # Run the derivative based initialization.
    phi_iono_der, _, layer_phi_der, azimuth_iono_space_axis = ionospheric_iterative_inversion_derivative(
        ms_coherence_slice=ms_coherence_slice,
        squint_axis=squint_axis,
        azimuth_space_axis=azimuth_space_axis,
        ionospheric_ranges=ionospheric_ranges,
        max_iterations=max_iterations_derivative,
        convergence_threshold=convergence_threshold_derivative,
        max_iono_range_offset=max_iono_range_offset,
        max_iono_range=max_iono_range,
    )

    # Compute the residual.
    residual_coherence = ms_coherence_slice * np.exp(-1j * phi_iono_der, dtype=complex_dtype)
    intermediate_phi_tgt = np.angle(nanmean(residual_coherence, axis=0))
    residual_coherence = residual_coherence * np.exp(-1j * intermediate_phi_tgt, dtype=complex_dtype)

    # Projection-based refinement.
    phi_iono_res, phi_tgt_res, layer_phi_res, _ = ionospheric_iterative_inversion_projection(
        ms_coherence_slice=residual_coherence,
        squint_axis=squint_axis,
        azimuth_space_axis=azimuth_space_axis,
        ionospheric_ranges=ionospheric_ranges,
        max_iterations=max_iterations_projection,
        convergence_threshold=convergence_threshold_projection,
        max_iono_range_offset=max_iono_range_offset,
        max_iono_range=max_iono_range,
    )

    return (
        phi_iono_der + phi_iono_res,
        intermediate_phi_tgt + phi_tgt_res,
        layer_phi_der + layer_phi_res,
        azimuth_iono_space_axis,
    )


def ionospheric_iterative_inversion_derivative(
    *,
    ms_coherence_slice: npt.NDArray[complex],
    squint_axis: npt.NDArray[float],
    azimuth_space_axis: npt.NDArray[float],
    ionospheric_ranges: npt.NDArray[float],
    max_iterations: int,
    convergence_threshold: float,
    max_iono_range_offset: float,
    max_iono_range: float | None,
) -> tuple[
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
]:
    """
    The ionospheric iterative inversion method using the gradient approach.

    This function serves as the core solver for the ionospheric processor. It
    estimates the 2D phase screens (layerPhases) for multiple atmospheric
    altitudes by iteratively minimizing the residual phase variance across
    the Multi-Squint stack.

    Parameters
    ----------
    ms_coherence_slice: npt.NDArray[complex]
        A [Nf x Na] slice of the Multi-Squint coherence matrix.

    squint_axis: npt.NDArray[float]  [m]
        The [1 x Nf] squint axis.

    azimuth_space_axis: npt.NDArray[float]  [m]
        The [1 x Na] space axis in along-track direction associated to the
        Multi-Squint coherence matrix.

    ionospheric_ranges: npt.NDArray[float]  [m]
        The [1 x Nri] ionospheric ranges.

    max_iterations: int
        Maximum number of iterations.

    convergence_threshold: float
        The threshold used as stopping criterion.

    max_iono_range_offset: float
        Maximum offset in ionospheric offset.

    max_iono_range: float | None
        Optionally, a maximum value for the ionospheric range.

    Return
    ------
    ms_ionospheric_phases: npt.NDArray[float]
        Total integrated ionospheric phase re-projected to the original
        squint-azimuth grid.

    target_phases: npt.NDArray[float]  [rad]
        Residual ground/target phase after ionospheric correction [1 x Na]

    layer_phases: npt.NDArray[float]  [rad]
        Estimated phase screens for each height [Niaz x Nri].

    azimuth_iono_space_axis: npt.NDArray[float]  [m]
        The space axes in along-track direction at the ionospheric height.
        This axes exstended the along-track space axis at the ground by the
        squint angle of the satellite.

    """
    # The floating-point precision for real/scalar arrays.
    float_dtype = ms_coherence_slice.real.dtype
    complex_dtype = ms_coherence_slice.dtype

    # Buffer length to account for squint look-ahead/look-behind at high altitudes.
    azimuth_space_step = np.diff(azimuth_space_axis[:2])[0]  # [m].
    if max_iono_range is None:
        max_iono_range = np.max(ionospheric_ranges) + max_iono_range_offset

    filter_length_smp = round(2 * max_iono_range * np.max(np.abs(squint_axis)) / azimuth_space_step)  # [px].

    # Compute the along-track axis at the ionospheric height.
    x0 = azimuth_space_axis[0] - filter_length_smp * azimuth_space_step
    nx = azimuth_space_axis.size + 2 * filter_length_smp + 1
    dx = azimuth_space_step
    azimuth_iono_space_axis = x0 + np.arange(nx) * dx

    # Prepare the iterative optimization.
    Psi, Xp = np.meshgrid(squint_axis, azimuth_space_axis, indexing="ij")
    Xi, Xq = np.meshgrid(azimuth_iono_space_axis, azimuth_space_axis, indexing="ij")

    layer_phases = np.zeros((len(ionospheric_ranges), nx), dtype=float_dtype)
    layer_phases_loop = np.zeros((len(ionospheric_ranges), nx), dtype=float_dtype)

    # Start the iterative optimization.
    ms_coherence = np.copy(ms_coherence_slice)
    prev_epsilon = -np.inf
    for i in range(max_iterations):
        # Update the phases and residual
        phases = np.angle(nanmean(ms_coherence, axis=0))  # Average over squints.
        residual = ms_coherence * np.exp(-1j * phases, dtype=complex_dtype)

        # Check convergence. If we didn't converge, run another iteration.
        curr_epsilon = np.mean(residual).real
        if curr_epsilon - prev_epsilon <= convergence_threshold:
            break
        prev_epsilon = curr_epsilon

        layer_phases = np.copy(layer_phases_loop)
        for h, r in enumerate(reversed(ionospheric_ranges), start=1):
            # Reverse the index.
            k = len(ionospheric_ranges) - h

            # Reinterpolate the Multi-Squint coherence matrix.
            ms_coherence_interp = _interp2d(
                x_axis=squint_axis,
                y_axis=azimuth_space_axis,
                xy_values=ms_coherence,
                x_points=(Xq - Xi) / r,
                y_points=Xq,
            ).astype(complex_dtype)

            ms_coherence_grad = ms_coherence_interp[1:, :] * np.conj(ms_coherence_interp[:-1, :])
            ms_coherence_grad[np.all(np.isnan(ms_coherence_grad), axis=1)] = 0.0
            ms_coherence_grad = np.nan_to_num(
                nanmean(ms_coherence_grad, axis=1),
                nan=0,
                posinf=0,
                neginf=0,
            )

            delta_iono_phase = np.cumsum(np.concatenate(([0], np.angle(ms_coherence_grad))))
            layer_phases_loop[k, :] += delta_iono_phase

            ms_coherence_delta = sp.interpolate.interp1d(
                azimuth_iono_space_axis,
                np.exp(1j * delta_iono_phase),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )(Xp - r * Psi)

            ms_coherence *= np.conj(ms_coherence_delta, dtype=complex_dtype)

    # Final composition.
    ms_iono_phases = np.zeros_like(ms_coherence_slice, dtype=float_dtype)
    for i, r in enumerate(ionospheric_ranges):
        ms_iono_phases += np.angle(
            sp.interpolate.interp1d(
                azimuth_iono_space_axis,
                np.exp(1j * layer_phases[i, :]),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )(Xp - r * Psi)
        )

    ionosphere_phasors = np.exp(1j * ms_iono_phases, dtype=complex_dtype)
    target_phases = np.angle(nanmean(ms_coherence_slice * np.conj(ionosphere_phasors), axis=0))
    layer_phases = layer_phases.T

    return ms_iono_phases, target_phases, layer_phases, azimuth_iono_space_axis


def ionospheric_iterative_inversion_projection(
    *,
    ms_coherence_slice: npt.NDArray[complex],
    squint_axis: npt.NDArray[float],
    azimuth_space_axis: npt.NDArray[float],
    ionospheric_ranges: npt.NDArray[float],
    max_iterations: int,
    convergence_threshold: float,
    max_iono_range_offset: float,
    max_iono_range: float | None,
) -> tuple[
    npt.NDArray[complex],
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
]:
    """The ionospheric iterative inversion method using the projection approach.

    This function serves as the core solver for the ionospheric processor. It
    estimates the 2D phase screens (layerPhases) for multiple atmospheric
    altitudes by iteratively minimizing the residual phase variance across
    the Multi-Squint stack.

    Parameters
    ----------
    ms_coherence_slice: npt.NDArray[complex]
        A [Nf x Na] slice of the Multi-Squint coherence matrix.

    squint_axis: npt.NDArray[float]  [m]
        The [1 x Nf] squint axis.

    azimuth_space_axis: npt.NDArray[float]  [m]
        The [1 x Na] space axis in along-track direction associated to the
        Multi-Squint coherence matrix.

    ionospheric_ranges: npt.NDArray[float]  [m]
        The [1 x Nri] ionospheric ranges.

    max_iterations: int
        Maximum number of iterations.

    convergence_threshold: float
        The threshold used as stopping criterion.

    max_iono_range_offset: float
        Maximum offset in ionospheric offset.

    max_iono_range: float | None
        Optionally, a maximum value for the ionospheric range.

    Return
    ------
    ionosphere_phasors: npt.NDArray[complex]
        Total integrated ionospheric phase re-projected to the original
        squint-azimuth grid.

    target_phases: npt.NDArray[float]  [rad]
        Residual ground/target phase after ionospheric correction [1 x Na]

    layer_phases: npt.NDArray[float]  [rad]
        Estimated phase screens for each height [Niaz x Nri].

    azimuth_iono_space_axis: npt.NDArray[float]  [m]
        The space axes in along-track direction at the ionospheric height.
        This axes exstended the along-track space axis at the ground by the
        squint angle of the satellite.

    """
    # The floating-point precision for real/scalar arrays.
    float_dtype = ms_coherence_slice.real.dtype
    complex_dtype = ms_coherence_slice.dtype

    # Buffer length to account for squint look-ahead/look-behind at high altitudes.
    azimuth_space_step = np.diff(azimuth_space_axis[:2])[0]  # [m].
    if max_iono_range is None:
        max_iono_range = np.max(ionospheric_ranges) + max_iono_range_offset

    filter_length_smp = round(2 * max_iono_range * np.max(np.abs(squint_axis)) / azimuth_space_step)  # [px].

    # Compute the along-track axis at the ionospheric height.
    x0 = azimuth_space_axis[0] - filter_length_smp * azimuth_space_step
    nx = azimuth_space_axis.size + 2 * filter_length_smp + 1
    dx = azimuth_space_step
    azimuth_iono_space_axis = x0 + np.arange(nx) * dx

    # Prepare the iterative optimization.
    Psi, Xp = np.meshgrid(squint_axis, azimuth_space_axis, indexing="ij")
    Psi_i, Xi = np.meshgrid(squint_axis, azimuth_iono_space_axis, indexing="ij")

    inversion_ionospheric_ranges = [0.0, *ionospheric_ranges]
    layer_phases = np.zeros((len(inversion_ionospheric_ranges), nx), dtype=float_dtype)
    layer_phases_loop = np.zeros((len(inversion_ionospheric_ranges), nx), dtype=float_dtype)

    # Start the iterative optimization.
    ms_coherence = np.copy(ms_coherence_slice)
    prev_epsilon = -np.inf
    for i in range(max_iterations):
        # Update the phases and residual
        phases = np.angle(np.mean(ms_coherence, axis=0))  # Average over squints.
        residual = ms_coherence * np.exp(-1j * phases, dtype=complex_dtype)

        # Check convergence. If we didn't converge, run another iteration.
        curr_epsilon = np.mean(residual).real
        if curr_epsilon - prev_epsilon <= convergence_threshold:
            break
        prev_epsilon = curr_epsilon

        layer_phases = np.copy(layer_phases_loop)
        for h, r in enumerate(reversed(inversion_ionospheric_ranges), start=1):
            # Reverse the index.
            k = len(inversion_ionospheric_ranges) - h

            # Reinterpolate the Multi-Squint coherence matrix.
            ms_coherence_interp = np.nan_to_num(
                _interp2d(
                    x_axis=squint_axis,
                    y_axis=azimuth_space_axis,
                    xy_values=ms_coherence,
                    x_points=Psi_i,
                    y_points=Xi + r * Psi_i,
                ).astype(complex_dtype),
                nan=0,
                posinf=0,
                neginf=0,
            )

            delta_iono_ms_coherence = np.mean(ms_coherence_interp, axis=0)
            delta_iono_phase = np.angle(delta_iono_ms_coherence)
            layer_phases_loop[k, :] += delta_iono_phase

            ms_coherence_delta = sp.interpolate.interp1d(
                azimuth_iono_space_axis,
                np.exp(1j * delta_iono_phase),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )(Xp - r * Psi)

            ms_coherence *= np.conj(ms_coherence_delta, dtype=complex_dtype)

    # Final composition.
    ms_iono_phases = np.zeros_like(ms_coherence_slice, dtype=float_dtype)
    for i, r in enumerate(inversion_ionospheric_ranges[1:], start=1):
        ms_iono_phases += np.angle(
            sp.interpolate.interp1d(
                azimuth_iono_space_axis,
                np.exp(1j * layer_phases[i, :]),
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )(Xp - r * Psi)
        )

    ionosphere_phasors = np.exp(1j * ms_iono_phases, dtype=complex_dtype)
    target_phases = np.angle(nanmean(ms_coherence_slice * np.conj(ionosphere_phasors), axis=0))
    layer_phases = layer_phases.T

    return ms_iono_phases, target_phases, layer_phases[:, 1:], azimuth_iono_space_axis


def _interp2d(
    *,
    x_axis: npt.NDArray[float],
    y_axis: npt.NDArray[float],
    xy_values: npt.NDArray[complex],
    x_points: npt.NDArray[float],
    y_points: npt.NDArray[float],
) -> npt.NDArray[complex]:
    """A 2D complex interpolator using map_coordinates (cubic spline)."""
    # map_coordinates expects coordinates as (ndim, N).
    coords = np.vstack(
        [
            (x_points - x_axis[0]).ravel() / np.diff(x_axis[:2])[0],
            (y_points - y_axis[0]).ravel() / np.diff(y_axis[:2])[0],
        ]
    )

    # Cubic spline interpolation (order=3) for real and imaginary part.
    real_part = sp.ndimage.map_coordinates(
        xy_values.real,
        coords,
        order=3,
        mode="constant",
        cval=np.nan,
    )
    imag_part = sp.ndimage.map_coordinates(
        xy_values.imag,
        coords,
        order=3,
        mode="constant",
        cval=np.nan,
    )

    return (real_part + 1j * imag_part).reshape(x_points.shape)
