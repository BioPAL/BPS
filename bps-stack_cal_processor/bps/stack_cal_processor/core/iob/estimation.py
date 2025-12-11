# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Slow-varying Ionosphere Estimation Library
------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import (
    BaselineMethodType,
    StackCalConf,
    StackDataSpecs,
)
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_list_numeric_types_equal,
    assert_numeric_types_equal,
)
from bps.stack_cal_processor.core.interferometric_pairing import (
    validate_interferometric_pairs,
)
from bps.stack_cal_processor.core.iob.splitspectrum import split_spectrum
from bps.stack_cal_processor.core.iob.utils import (
    IobRuntimeError,
    SubLookConf,
    compute_qualities,
    compute_range_sublook,
    phase_unwrap_validity_mask,
)
from bps.stack_cal_processor.core.signal_processing import (
    compute_coherence,
    compute_range_common_bands_and_sublook_freqs,
    subsample_data,
)
from bps.stack_cal_processor.core.utils import percentage_completed_msg
from skimage.restoration import unwrap_phase


@dataclass
class EstimationOutput:
    """The result of a sinble-baseline estimation and a validity flag."""

    usable: bool
    """Whether the estimation is usable or must be discarded."""

    azimuth_slope: float | None = None  # [rad/s].
    """The estimated ionospheric azimuth slope in [rad/s]."""

    range_slope: float | None = None  # [rad/s].
    """The estimated ionospheric range slope in [rad/s]."""

    quality: float | None = None  # [adim].
    """The estimation quality."""

    mb_weight_azimuth: float | None = None  # [adim].
    """The azimuth multi-baseline weighting."""

    mb_weight_range: float | None = None  # [adim].
    """The range multi-baseline weighting."""

    interferometric_pair: tuple[int, int] | None = None
    """The interferometric pair associated to the estimation."""


def pairwise_ionosphere_estimations_multithreaded(
    *,
    stack_images: tuple[npt.NDArray[complex], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    split_spectrum_biases: tuple[npt.NDArray[float], ...],
    interferometric_pair_indices: tuple[tuple[int, int]],
    iob_conf: StackCalConf.IobConf,
    stack_specs: StackDataSpecs,
    estimation_dtypes: EstimationDType,
    num_worker_threads: int,
) -> tuple[EstimationOutput]:
    """
    Execute the slow-varying ionosphere single-baseline estimation for
    a set of interferometric pairs. Depending on the provided pairs, the
    estimation will be a purely Single-Baseline (i.e. only pair of the
    form <calib-primary, secondary>) or a Multi-Baseline one, or anything
    in between.

    This method applies the Split-Spectrum algorithm as defined
    by Gomba et al. "Toward Operational Compensation of Ionospheric
    Effects in SAR Interferograms: The Split-Spectrum Method", IEEE 2016
    on each interferometric pair.

    Parameters
    ----------
    stack_images: tuple[npt.NDArray[complex], ...]
        The stack of Nimg images for a selected polarization. All
        images must have shape [Nazm x Nrng]. This images are expected
        to have been preprocessed via range spectral whitening and
        the synthetic phases from DEM being compensated.

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM. All images must have
        shape [Nazm x Nrng].

    split_spectrum_biases: tuple[npt.NDArray[float], ...] [rad * s]
        The Nimg split-spectrum biases due, e.g, data coregistration
        and/or L1 ionospheric corrections. All images are encoded
        as [Nazm x Nrng] arrays.

    interferometric_pair_indices: tuple[tuple[int, int]]
        The interferometric pairs for the interferometric
        estimations.

    iob_conf: StackCalConf.IobConf
        The parameter configuration of the algorithm.

    stack_specs: StackDataSpecs
        The stack specs object.

    estimation_dtypes: EstimationDType
        The floating point precision to be used for the estimation.

    num_worker_threads: int
        Number of threads assigned to the job.

    Raises
    ------
    IobRuntimeError

    Return
    ------
    tuple[EstimationOutput]
        A list of Nimg phase screen estimations with quality and a
        validity flag.

    """
    # Compute the bounds of the common bands (f_LH) and the center of the
    # sub-bands (f_lh). The frequency are absolute (i.e. *not* expressed
    # wrt the central frequency).
    f_LH, f_lh = compute_range_common_bands_and_sublook_freqs(
        interferometric_pair_indices=interferometric_pair_indices,
        synth_images=synth_phases,
        sampling_frequency=1 / stack_specs.range_sampling_step,
        bandwidth=stack_specs.range_bandwidth,
        look_band=iob_conf.range_look_band,
        central_frequency=stack_specs.central_frequency,
        dtype=estimation_dtypes.float_dtype,
    )

    # Convert them into sub-look configurations. The low/high sub-looks are
    # centered in f_l/f_h and have bandwidth defined as a percentage of the
    # common band (f_H - f_L).
    sublook_confs_l = {
        p: SubLookConf(
            frequency=f_lh[p]["l"],
            bandwidth=(f_LH[p]["h"] - f_LH[p]["l"]) * iob_conf.range_look_frequency,
            filter_type=iob_conf.sublook_filter_type,
            filter_param=iob_conf.sublook_filter_param,
        )
        for p in interferometric_pair_indices
    }
    sublook_confs_h = {
        p: SubLookConf(
            frequency=f_lh[p]["h"],
            bandwidth=(f_LH[p]["h"] - f_LH[p]["l"]) * iob_conf.range_look_frequency,
            filter_type=iob_conf.sublook_filter_type,
            filter_param=iob_conf.sublook_filter_param,
        )
        for p in interferometric_pair_indices
    }

    # Compute the phase biases bla bla.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The single-baseline estimation core routine. This executes both
        # the range spectral window removal and the synthtetic phase
        # removal.
        def iob_estimation_fn(pair_ps):
            try:
                output = estimate_slow_varying_ionosphere(
                    image_p=stack_images[pair_ps[0]],
                    image_s=stack_images[pair_ps[1]],
                    split_spectrum_bias_p=split_spectrum_biases[pair_ps[0]],
                    split_spectrum_bias_s=split_spectrum_biases[pair_ps[1]],
                    sublook_conf_l=sublook_confs_l[pair_ps],
                    sublook_conf_h=sublook_confs_h[pair_ps],
                    iob_conf=iob_conf,
                    stack_specs=stack_specs,
                    estimation_dtypes=estimation_dtypes,
                )
            # pylint: disable=broad-exception-caught
            except Exception as ex:
                bps_logger.warning("Iono estimation failed for pair %s", pair_ps)
                bps_logger.debug("Got error: %s", ex)
                output = EstimationOutput(usable=False)
            finally:
                output.interferometric_pair = pair_ps
            return output

        # Dispatch the estimation multithreaded.
        iob_estimations = []
        for iob_estimation in executor.map(iob_estimation_fn, interferometric_pair_indices):
            iob_estimations.append(iob_estimation)
            bps_logger.info(
                percentage_completed_msg(
                    len(iob_estimations),
                    total=len(interferometric_pair_indices),
                )
            )

        return tuple(iob_estimations)


def estimate_slow_varying_ionosphere(
    *,
    image_p: npt.NDArray[complex],
    image_s: npt.NDArray[complex],
    split_spectrum_bias_p: npt.NDArray[float],
    split_spectrum_bias_s: npt.NDArray[float],
    sublook_conf_l: SubLookConf,
    sublook_conf_h: SubLookConf,
    iob_conf: StackCalConf.IobConf,
    stack_specs: StackDataSpecs,
    estimation_dtypes: EstimationDType,
) -> EstimationOutput:
    """
    Perform the Single-baseline (pairwise) estimation of the slow-varying
    ionosphere. This method executes the following steps:

    1- Compute the high resolution coherence for the interferometric
       pair. The coherence is used for weighing the the Least-Square
       estimation of the phase slopes (see Step 5- below).
    2- Compute interferometric phases phi_l and phi_h associated
       to the low and high sub-looks of the interferometric pair.
    3- Apply the Split-Spectrum biases to phi_l and phi_h, in order to
       account for possible effects of coregistration shifts and phase
       delays due to the L1 ionosphere estimations.
    4- Extract the dispersive component of the phase using the
       Split-Spectrum algorithm phi_l and phi_h.
    5- Extract the phase slopes in azimuth and range direction as
       well as the estimation quality via Least-Square estimation.

    Parameters
    ----------
    image_p: npt.NDArray[complex]
        Primary image of the interferometric pair.

    image_s: npt.NDArray[complex]
        Secondary image of the interferometric pair.

    split_spectrum_bias_p: npt.NDArray[float] [rad/Hz]
        Bias terms of the Split-Spectrum for the primary image.

    split_spectrum_bias_s: npt.NDArray[float] [rad/Hz]
        Bias terms of the Split-Spectrum for the secondary image.

    sublook_conf_l: SubLookConf
        Configuration of the low sub-look.

    sublook_conf_h: SubLookConf
        Configuration of the high sub-look.

    iob_conf: StackCalConf.IobConf
        The parameter configuration of the algorithm.

    stack_specs: StackDataSpecs
        The stack specs object.

    estimation_dtypes: EstimationDType
        The floating point precision to be used for the estimation.

    Return
    ------
    EstimationOutput
        Object containing the results of the estimation.

    """
    # Compute the full resolution coherences.
    coherence = np.abs(
        compute_coherence(
            image_p,
            image_s,
            decimation_factors=iob_conf.split_spectrum_decimation_factors,
            dtype=estimation_dtypes.complex_dtype,
        )
    )
    assert_numeric_types_equal(coherence, expected_dtype=estimation_dtypes.float_dtype)

    # Compute the interferogram of the low sub-look.
    phi_l = np.angle(
        compute_coherence(
            image_p=compute_range_sublook(
                stack_image=image_p,
                range_sampling_step=stack_specs.range_sampling_step,
                central_frequency=stack_specs.central_frequency,
                sublook_conf=sublook_conf_l,
                dtypes=estimation_dtypes,
            ),
            image_s=compute_range_sublook(
                stack_image=image_s,
                range_sampling_step=stack_specs.range_sampling_step,
                central_frequency=stack_specs.central_frequency,
                sublook_conf=sublook_conf_l,
                dtypes=estimation_dtypes,
            ),
            filter_window_size=iob_conf.sublook_window_sizes,
            decimation_factors=iob_conf.split_spectrum_decimation_factors,
            dtype=estimation_dtypes.complex_dtype,
        ),
    )
    phi_h = np.angle(
        compute_coherence(
            image_p=compute_range_sublook(
                stack_image=image_p,
                range_sampling_step=stack_specs.range_sampling_step,
                central_frequency=stack_specs.central_frequency,
                sublook_conf=sublook_conf_h,
                dtypes=estimation_dtypes,
            ),
            image_s=compute_range_sublook(
                stack_image=image_s,
                range_sampling_step=stack_specs.range_sampling_step,
                central_frequency=stack_specs.central_frequency,
                sublook_conf=sublook_conf_h,
                dtypes=estimation_dtypes,
            ),
            filter_window_size=iob_conf.sublook_window_sizes,
            decimation_factors=iob_conf.split_spectrum_decimation_factors,
            dtype=estimation_dtypes.complex_dtype,
        ),
    )
    assert_list_numeric_types_equal((phi_l, phi_h), expected_dtype=estimation_dtypes.float_dtype)

    # Compensate possible biases from coregistration and L1 iono phase
    # compensation.
    split_spectrum_bias_ps = subsample_data(
        split_spectrum_bias_s - split_spectrum_bias_p,
        decimation_factors=iob_conf.split_spectrum_decimation_factors,
    )
    phi_l += (sublook_conf_l.frequency - stack_specs.central_frequency) * split_spectrum_bias_ps
    phi_h += (sublook_conf_h.frequency - stack_specs.central_frequency) * split_spectrum_bias_ps

    # Mask of pixels usable for plane fitting.
    validity_mask = coherence >= iob_conf.min_coherence_threshold

    # Possibly unwrap the phase screens and flag invalid pixels depending on
    # the phase unwrap test.
    if iob_conf.phase_unwrapping_flag:
        phi_l = unwrap_phase(phi_l)
        phi_h = unwrap_phase(phi_h)
        validity_mask &= phase_unwrap_validity_mask(phi_l, phi_h, max_lh_delta_phase=iob_conf.max_lh_phase_delta)

    validity_ratio = np.count_nonzero(validity_mask) / validity_mask.size
    bps_logger.debug("Validity ratio: %f", validity_ratio)

    # If there are no valid pixels at all, we gotta stop here.
    if validity_ratio == 0:
        return EstimationOutput(
            azimuth_slope=None,
            range_slope=None,
            mb_weight_azimuth=0,
            mb_weight_range=0,
            quality=0,
            usable=False,
        )

    # Compute the dispersive phase via Split-Spectrum.
    phi_disp = split_spectrum(
        f_c=stack_specs.central_frequency,
        f_l=sublook_conf_l.frequency,
        f_h=sublook_conf_h.frequency,
        phi_l=phi_l,
        phi_h=phi_h,
    )

    # Run the least-square estimation.
    azm_slope_px, rng_slope_px = least_square_ionosphere_estimation(
        ionosphere_phase=phi_disp,
        weights=coherence,
        mask=validity_mask,
        decimation_factors=iob_conf.split_spectrum_decimation_factors,
    )
    # Compute the qualities, e.g. for multi-baseline estimation.
    sigma2_azm, sigma2_rng, quality = compute_qualities(
        stack_specs=stack_specs,
        iob_conf=iob_conf,
        coherence=coherence,
        f_l=sublook_conf_l.frequency,
        f_h=sublook_conf_h.frequency,
        estimation_dtypes=estimation_dtypes,
    )

    # NOTE: In principle the standard deviations sigma2_azm and sigma2_rng are
    # greater than zero. To avoid division by zero's or ill-conditioned values,
    # we set a minimum value (just a small one, but not close to machine
    # precision).
    return EstimationOutput(
        azimuth_slope=azm_slope_px / stack_specs.azimuth_sampling_step,
        range_slope=rng_slope_px / stack_specs.range_sampling_step,
        mb_weight_azimuth=1 / max(sigma2_azm, np.finfo(np.float32).eps),
        mb_weight_range=1 / max(sigma2_rng, np.finfo(np.float32).eps),
        quality=quality,
        usable=validity_ratio >= iob_conf.min_usable_pixel_ratio,
    )


def least_square_ionosphere_estimation(
    ionosphere_phase: npt.NDArray[float],
    weights: npt.NDArray[float],
    mask: npt.NDArray[bool],
    decimation_factors: tuple[int, int] = (1, 1),
) -> tuple[float, float]:
    """
    Estimate the slopes of the ionosphere phase plane via weighted
    least-square regression.

    Parameters
    ----------
    ionosphere_phase: npt.NDArray[float] [rad]
        The observed iononosphere stored as an array of shape [Nazm' x Nrng'].

    weights: npt.NDArray[float]
        The regression weights stored as an array of shape [Nazm' x Nrng'].

    mask: npt.NDArray[bool]
        A validity mask, stored as an array of shape [Nazm' x Nrng']. Invalid
        ionosphere observations will be discarded in the regression.

    decimation_factors: tuple[1, 1] = (1, 1)
        Decimation factors used to compute the ionosphere phase.

    Return
    ------
    azm_slope: float [rad/px]
        Slope of the regressed ionosphere plane in azimuth direction.

    rng_slope: float [rad/px]
        Slope of the regressed ionosphere plane in range direction.

    """
    # Assemble the LS problem
    #
    #    argmin_{c} |F^T W \phi{disp} - F^T W F c|^2
    #
    # where the single-baseline matrix F and the weights W are defined as follows:
    #
    #  W[i, j] = abs(interferogram_h[i]) * abs(interferogram_l[j]) * delta[i, j]
    #
    # with delta[i, j] the Kronecker delta matrix, and
    #
    #  F[i, j] = [[1, 0, 0],
    #             [1, 1, 0],
    #             [1, 2, 0],
    #             ...
    #             [1, n, 0],
    #             [1, 0, 1],
    #             [1, 1, 1],
    #             ...
    #             [1, n, 1],
    #             [1, 0, 2],
    #             [1, 1, 2],
    #
    #             ...
    #
    #             [1, n, n]]
    #
    num_azimuths, num_ranges = weights.shape
    axis_indices_azimuth = np.arange(num_azimuths).reshape(-1, 1) * decimation_factors[0]
    axis_indices_range = np.arange(num_ranges).reshape(1, -1) * decimation_factors[1]

    # pylint: disable=invalid-name
    F = np.hstack(
        [
            np.ones_like(weights).reshape(-1, 1),
            (np.ones_like(weights) * axis_indices_range).reshape(-1, 1),
            (np.ones_like(weights) * axis_indices_azimuth).reshape(-1, 1),
        ]
    )
    W = sp.sparse.diags((weights * mask).reshape(-1))

    Ft_W = F.T @ W
    Ft_W_F = Ft_W @ F

    slopes, *_ = np.linalg.lstsq(Ft_W_F, Ft_W @ ionosphere_phase.reshape(-1), rcond=None)

    return slopes[2], slopes[1]


def combine_pairwise_ionosphere_estimations(
    *,
    iob_estimations: tuple[EstimationOutput, ...],
    iob_conf: StackCalConf.IobConf,
    calib_reference_index: int,
    num_images: int,
) -> tuple[
    npt.NDArray[float],
    npt.NDArray[float],
    npt.NDArray[float],
    tuple[tuple[int, int], ...],
]:
    """
    Combine a set of pairwise ionosphere estimations on interferometric
    pairs into single estimation per each image and return the azimuth
    and range slopes of the ionosphere phase screen as well as a
    quality value.

    Parameters
    ----------
    iob_estimations: tuple[EstimationOutput]
        The interferometric pairwise estiamtions.

    iob_conf: StackCalConf.IobConf
        The configuration of the algorithm.

    calib_reference_index: int
        Index of the calibration reference image.

    num_images: int
        The stack original size.

    Raises
    ------
    IobRuntimeError

    Return
    ------
    iono_azimuth_slopes: npt.NDArray[float] [rad/s]
        The [1 x Nimg] ionosphere phase slopes in azimuth direction.

    iono_range_slopes: npt.NDArray[float] [rad/s]
        The [1 x Nimg] ionosphere phase slopes in range directions.

    iono_qualities: npt.NDArray[float]
        The [1 x Nimg] estimation qualities.

    interferometric_pair_indices: tuple[tuple[int, int], ...]
        The interferometric pair actually used for the estimation (i.e.
        the ones associated to valid estimations).

    """
    # First, we need to check that the pairwise estimations are sufficient
    # to estimate the desired quantities for each stack image.
    interferometric_pair_indices = tuple(e.interferometric_pair for e in iob_estimations if e.usable)
    if not validate_interferometric_pairs(
        interferometric_pair_indices,
        num_images=num_images,
    ):
        raise IobRuntimeError("Not enough valid pairwise estimations.")

    if iob_conf.baseline_method == BaselineMethodType.SINGLE_BASELINE:
        sb_output = _single_baseline_estimation(
            iob_estimations,
            calib_reference_index=calib_reference_index,
            num_images=num_images,
        )
        return *sb_output, interferometric_pair_indices

    if iob_conf.baseline_method == BaselineMethodType.MULTI_BASELINE:
        mb_output = _multi_baseline_combine(
            iob_estimations,
            uniform_weighting=iob_conf.multi_baseline_uniform_weighting,
            calib_reference_index=calib_reference_index,
            num_images=num_images,
        )
        return *mb_output, interferometric_pair_indices

    # We just got an invalid input.
    raise IobRuntimeError(f"Invalid baseline method {iob_conf.baseline_method}")


def _multi_baseline_combine(
    iob_estimations: tuple[EstimationOutput, ...],
    *,
    uniform_weighting: bool,
    calib_reference_index: int,
    num_images: int,
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """Combine the IOB pairwise estimation via Multi-Baseline method."""
    # pylint: disable=invalid-name
    # Pack the azimuth and range slope estimations into arrays. Invalid
    # estimations will be taken care of later.
    num_pairs = len(iob_estimations)
    iono_azimuth_slopes = np.zeros(num_pairs)
    iono_range_slopes = np.zeros(num_pairs)

    # Assemble the multi-baseline matrix. The multi-baseline matrix F is defined
    # as follows: If the i-th interferometric pair is (p{i}, s{i}), then
    #
    #   F[i, p] = -1,
    #   F[i, s] = 1,
    #
    # and 0 otherwise. In other words, the matrix F associated to a INT phase
    # (i.e. 3 stack images) will look like
    #
    #   F = [[-1,  1,  0],
    #        [-1,  0,  1],
    #        [ 0, -1,  1]]
    #
    F = np.zeros((num_pairs, num_images))
    Q = np.zeros((num_pairs, num_pairs))
    Wa = np.zeros((num_pairs, num_pairs))
    Wr = np.zeros((num_pairs, num_pairs))
    for i, iob_out in enumerate(iob_estimations):
        F[i, iob_out.interferometric_pair[0]] = -1
        F[i, iob_out.interferometric_pair[1]] = 1
        if iob_out.usable:
            Q[i, i] = iob_out.quality
            # Populate the target values.
            iono_azimuth_slopes[i] = iob_out.azimuth_slope
            iono_range_slopes[i] = iob_out.range_slope
            # Populate the LS/Quality matrices.
            Wa[i, i] = iob_out.mb_weight_azimuth
            Wr[i, i] = iob_out.mb_weight_range

    # Caching some values to avoid double computations. We don't need a sparse
    # matrix here since everything is quite small. The weighted least square
    # problem can be formulated as
    #
    #   argmin{u} |(F^T @ W @ F) @ u - (F^T * W) @ b |^2
    #
    # which approximates F_lhs @ u = F_rhs @ b, with
    #
    #   F_lhs := F^T @ W @ F,
    #   F_rhs := F^T @ W,
    #
    W = np.diag([float(out.usable) for out in iob_estimations])

    # The azimuth slopes for the multi-baseline estimation.
    if not uniform_weighting:
        W = Wa

    Ft_W = F.T @ W
    Ft_W_F = Ft_W @ F

    mb_iono_azimuth_slopes, *_ = np.linalg.lstsq(
        Ft_W_F,
        Ft_W @ iono_azimuth_slopes,
        rcond=None,
    )
    mb_iono_azimuth_slopes -= mb_iono_azimuth_slopes[calib_reference_index]
    if mb_iono_azimuth_slopes.size != num_images:
        raise IobRuntimeError("Invalid number of azimuth slope estimations: {:d}".format(mb_iono_azimuth_slopes.size))

    # The range slopes for the multi-baseline estimation.
    if not uniform_weighting:
        W = Wr

    Ft_W = F.T @ W
    Ft_W_F = Ft_W @ F

    mb_iono_range_slopes, *_ = np.linalg.lstsq(
        Ft_W_F,
        Ft_W @ iono_range_slopes,
        rcond=None,
    )
    mb_iono_range_slopes -= mb_iono_range_slopes[calib_reference_index]
    if mb_iono_range_slopes.size != num_images:
        raise IobRuntimeError("Invalid number range slope estimations: {:d}".format(mb_iono_range_slopes.size))

    return (
        mb_iono_azimuth_slopes,
        mb_iono_range_slopes,
        np.max(np.abs(F.T @ Q), axis=1),
    )


def _single_baseline_estimation(
    iob_estimations: tuple[EstimationOutput, ...],
    *,
    calib_reference_index: int,
    num_images: int,
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """Return the single-baseline estimation."""
    # We expect Nimg - 1 estimations.
    if len(iob_estimations) != num_images - 1:
        raise IobRuntimeError("Invalid number of IOB estimations: {:d}".format(len(iob_estimations)))

    # Prepare the output.
    sb_iono_azimuth_slopes = np.zeros((num_images,))
    sb_iono_range_slopes = np.zeros((num_images,))
    sb_iono_qualities = np.full((num_images,), 0.5)

    # In Single-Baseline mode, we the pairs to be <calib-ref,secondary>.
    for out in iob_estimations:
        index_p, index_s = out.interferometric_pair
        if index_p != calib_reference_index or index_s == calib_reference_index:
            raise IobRuntimeError(
                f"Ill-formed Single-Baseline pair: {(index_p, index_s)}",
            )
        if not out.usable:
            raise IobRuntimeError(
                f"Invalid Single-Baseline pair: {(index_p, index_s)}",
            )

        sb_iono_azimuth_slopes[index_s] = out.azimuth_slope
        sb_iono_range_slopes[index_s] = out.range_slope
        sb_iono_qualities[index_s] = out.quality

    return sb_iono_azimuth_slopes, sb_iono_range_slopes, sb_iono_qualities


def has_failed_pairwise_estimations(
    iob_estimations: tuple[EstimationOutput, ...],
    calib_reference_index: int,
) -> npt.NDArray[int]:
    """
    Return the indices of the images associated to invalid pairwise
    estimations.

    Parameters
    ----------
    iob_estimations: tuple[EstimationOutput, ...]
        The set of IOB pairwise estimations.

    Return
    ------
    npt.NDArray[int]
        A 1D array of indices.

    """
    return np.setdiff1d(
        np.unique([e.interferometric_pair for e in iob_estimations if not e.usable]),
        calib_reference_index,
    ).astype(np.int16)
