# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Compute the Multi-Squint Coherence Matrix
-----------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil
from typing import Self

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.configuration import StackCalConf
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.msc.utils import nanmean
from bps.stack_cal_processor.core.signal_processing import compute_coherence


@dataclass
class MultiSquintCoherenceEstimationConf:
    """Parameters used to estimate the multi-squint coherence."""

    azimuth_bandwidth: float  # [Hz].
    """Total azimuth bandwidth to process [Hz]."""

    subband_azimuth_resolution: float  # [m].
    """Sub-bands azimuth resolution [m]."""

    azimuth_filter_window_size: float  # [m]
    """Size of the filtering window in along-track direction [m]."""

    range_filter_window_size: float  # [m].
    """Size of the filtering window in slant-range direction [m]."""

    @classmethod
    def from_msc_high_res_ms_coherence_conf(
        cls,
        conf: StackCalConf.MscConf,
        azimuth_space_resolution: float,
    ) -> Self:
        """Initialize from an MSC configuration object of the high resolution MS coherence."""
        return cls(
            azimuth_bandwidth=1 / azimuth_space_resolution,
            subband_azimuth_resolution=conf.ms_coherence_subband_azimuth_resolution,
            azimuth_filter_window_size=conf.ms_coherence_high_resolution_azimuth_window_size,
            range_filter_window_size=conf.ms_coherence_high_resolution_range_window_size,
        )

    @classmethod
    def from_msc_low_res_ms_coherence_conf(
        cls,
        conf: StackCalConf.MscConf,
        azimuth_space_resolution: float,
    ) -> Self:
        """Initialize from an MSC configuration object of the low resolution MS coherence."""
        return cls(
            azimuth_bandwidth=1 / azimuth_space_resolution,
            subband_azimuth_resolution=conf.ms_coherence_subband_azimuth_resolution,
            azimuth_filter_window_size=conf.ms_coherence_low_resolution_azimuth_window_size,
            range_filter_window_size=conf.ms_coherence_low_resolution_range_window_size,
        )


@dataclass
class MultiSquintCoherenceEstimationOutput:
    """The multi-squint coherence estimation output."""

    matrix: npt.NDArray[complex]
    """The [Nf x Nazm' x Nrng'] multi-squint coherence matrix."""

    subband_axis: npt.NDArray[float]  # [1/m].
    """The [1 x Nf] spatial frequency axis [1/m]."""

    azimuth_indices: npt.NDArray[int]
    """The [1 x Nazm'] array containing the indices of the subsampled along-track axis."""

    range_indices: npt.NDArray[int]
    """The [1 x Nrng'] array containing the indices of the subsampled range axis."""


@dataclass
class CoherenceImprovementAnlysisOutput:
    """Store the results of the coherence analysis."""

    improvement: float
    """The coherence improvement obtained when no azimuth shifts are applied."""

    average_ms_coherence: npt.NDArray[float]
    """The pixelwise average Multi-Squint coherence."""


def compute_multisquint_coherence_multithreaded(
    image_p: npt.NDArray[complex],
    image_s: npt.NDArray[complex],
    *,
    azimuth_space_axis: float,
    slant_range_space_axis: float,
    ms_coherence_conf: MultiSquintCoherenceEstimationConf,
    dtypes: EstimationDType,
    max_num_threads: int,
) -> npt.NDArray[complex]:
    """
    Compute the multi-squint coherence for an image pair.

    Parameters
    ----------
    image_p: npt.NDArray[complex]
        Primary image of the interferometric pair.

    image_s: npt.NDArray[complex]
        Secondary image of the interferometric pair.

    azimuth_space_axis: float [m]
        Space axis in along-track direction.

    slant_range_space_axis: float [m]
        Space axis in slant-range direction.

    ms_coherence_conf: MultiSquintCoherenceEstimationConf
        The parameters for estimating the multi-squint coherence.

    dtypes: EstimationDType
        The floating-point precision of the estimation.

    max_num_threads: int
        Number of worker threads assigned to the task.

    Return
    ------
    ms_coherence_out: MultiSquintCoherenceEstimationOutput
        The output of the multi-squint estimation.

    """
    # Compute the azimuth sub-band parameters.
    delta_fx = 1 / ms_coherence_conf.subband_azimuth_resolution
    num_freq = 2 * round(ms_coherence_conf.azimuth_bandwidth / delta_fx) + 1
    subband_axis = np.linspace(
        -ms_coherence_conf.azimuth_bandwidth / 2 + delta_fx / 2,
        ms_coherence_conf.azimuth_bandwidth / 2 - delta_fx / 2,
        num_freq,
        dtype=dtypes.float_dtype,
    )

    # Prepare the downsampling step in azimuth direction.
    azimuth_space_step = np.diff(azimuth_space_axis[:2])[0]  # [m].
    azimuth_filter_halfsize = round(0.5 * ms_coherence_conf.subband_azimuth_resolution / azimuth_space_step)
    azimuth_subsampling_indices = np.arange(
        azimuth_filter_halfsize,
        azimuth_space_axis.size - azimuth_filter_halfsize,
        max(azimuth_filter_halfsize, 1),
    )

    # Prepare the downsampling step in azimuth direction.
    slant_range_space_step = np.diff(slant_range_space_axis[:2])[0]  # [m].
    range_filter_size = 2 * round(0.5 * ms_coherence_conf.range_filter_window_size / slant_range_space_step) + 1
    range_subsampling_indices = np.arange(
        0,
        slant_range_space_axis.size,
        max(int(range_filter_size / 2), 1),
    )

    # The modulation matrix [Naw x Nf].
    M_psi = np.exp(
        -2j
        * np.pi
        * np.outer(
            azimuth_space_step
            * np.arange(
                -azimuth_filter_halfsize,
                azimuth_filter_halfsize + 1,
                dtype=dtypes.float_dtype,
            ),
            subband_axis,
        ),
        dtype=dtypes.complex_dtype,
    )

    # The range dowsampling and averaging operator.
    F_rng = _build_range_averaging_sparse_matrix(
        slant_range_space_axis[range_subsampling_indices],
        slant_range_space_axis,
        ms_coherence_conf.range_filter_window_size,
        dtypes.float_dtype,
    )

    # Prepare the output MS coherence.
    ms_coh_shape = (azimuth_subsampling_indices.size, range_subsampling_indices.size, num_freq)
    cov_pp = np.zeros(ms_coh_shape, dtype=dtypes.float_dtype)
    cov_ss = np.zeros(ms_coh_shape, dtype=dtypes.float_dtype)
    cov_ps = np.zeros(ms_coh_shape, dtype=dtypes.complex_dtype)

    with ThreadPoolExecutor(max_workers=max_num_threads) as executor:
        # The core estimation function.
        def process_azimuth_slice_fn(i, az):
            # The azimuth block.
            begin = az - azimuth_filter_halfsize
            end = az + azimuth_filter_halfsize + 1

            # The azimuth slices, propagated over the squints [Nr x Nf].
            image_p_slice = image_p[begin:end, :].T @ M_psi
            image_s_slice = image_s[begin:end, :].T @ M_psi

            # Populate the covariances.
            cov_pp[i, ...] = F_rng @ (image_p_slice * image_p_slice.conj()).real
            cov_ss[i, ...] = F_rng @ (image_s_slice * image_s_slice.conj()).real
            cov_ps[i, ...] = F_rng @ (image_s_slice * image_p_slice.conj())

        for _ in executor.map(
            process_azimuth_slice_fn,
            range(azimuth_subsampling_indices.size),
            azimuth_subsampling_indices,
        ):
            pass

    # Applt averaging alog the azimuth direction.
    azimuth_space_substep = np.diff(azimuth_space_axis[azimuth_subsampling_indices[:2]])[0]
    azimuth_boxcar_size = ceil(ms_coherence_conf.azimuth_filter_window_size / azimuth_space_substep)
    azimuth_boxcar = np.ones((azimuth_boxcar_size,), dtype=dtypes.float_dtype)
    if azimuth_boxcar.size > 1:
        cov_pp = sp.ndimage.convolve1d(cov_pp, azimuth_boxcar, axis=0, mode="constant")
        cov_ss = sp.ndimage.convolve1d(cov_ss, azimuth_boxcar, axis=0, mode="constant")
        cov_ps = sp.ndimage.convolve1d(cov_ps, azimuth_boxcar, axis=0, mode="constant")

    # Prepare the output.
    eps = np.finfo(dtypes.float_dtype).tiny
    valid = np.logical_and(cov_pp > eps, cov_ss > eps)

    # Compute the coherence.
    coh = np.zeros_like(cov_ps)
    coh[valid] = cov_ps[valid] / np.sqrt(cov_pp[valid] * cov_ss[valid])

    return MultiSquintCoherenceEstimationOutput(
        matrix=coh,
        subband_axis=subband_axis,
        azimuth_indices=azimuth_subsampling_indices,
        range_indices=range_subsampling_indices,
    )


def compute_coherence_improvement(
    image_p: npt.NDArray[complex],
    image_s: npt.NDArray[complex],
    ms_coherence: MultiSquintCoherenceEstimationOutput,
    *,
    ms_coh_validity_threshold: float = 0.3,
    coh_window_size_px: tuple[int, int] = (5, 5),
    dsi_p: npt.NDArray[float] | None = None,
    dsi_s: npt.NDArray[float] | None = None,
) -> CoherenceImprovementAnlysisOutput:
    """
    Compute the coherence improvement to decide for the Multi-Squint calibration or
    phase plane removal.

    Parameters
    ----------
    image_p: npt.NDArray[complex]
        The primary/reference image (with DSI compensation).

    image_p: npt.NDArray[complex]
        The secondary image (with DSI compensation).

    ms_coherence: MultiSquintCoherenceEstimationOutput
        The result of the Multi-Squint coherence estimation.

    ms_coh_validity_threshold: float = 0.3
        Threshold to identify the areas of valid Multi-Squint coherence.

    coh_window_size_px: tuple[int, int] = (5, 5)  [px]
        The multilook window used to compute the standard coherence.

    Returns
    -------
    CoherenceImprovementAnlysisOutput
        The result of the coherence analsys.

    """
    # Optionally, apply the DSI compensation.
    if dsi_p is not None:
        image_p = image_p * np.exp(1j * dsi_p)
    if dsi_s is not None:
        image_s = image_s * np.exp(1j * dsi_s)

    # Compute the cohrence and resample over the MS coherence axes.
    coherence = np.abs(
        compute_coherence(
            image_p=image_p,
            image_s=image_s,
            filter_window_size=coh_window_size_px,
        )[ms_coherence.azimuth_indices, :][:, ms_coherence.range_indices]
    )

    # Compute the average coherence with respect to the squints.
    average_ms_coherence = nanmean(np.abs(ms_coherence.matrix), axis=2)
    valid_ms_coh_mask = average_ms_coherence > ms_coh_validity_threshold

    output = CoherenceImprovementAnlysisOutput(
        average_ms_coherence=average_ms_coherence[valid_ms_coh_mask],
        improvement=0,
    )
    if valid_ms_coh_mask.size > 0:
        output.improvement = nanmean(output.average_ms_coherence) - nanmean(coherence[valid_ms_coh_mask])
    return output


def _build_range_averaging_sparse_matrix(
    slant_range_subsampled_space_axis: npt.NDArray[float],
    slant_range_space_axis: npt.NDArray[float],
    range_filter_window_size: float,
    dtype: np.dtype,
) -> sp.sparse.csr:
    """
    Construct sparse matrix Fr where

        Fr[i,j] = 1  if |r[j] - r_sub[i]| <= win/2
                  0  otherwise

    Returns CSR sparse matrix suitable for left multiplication.

    """
    window_half = range_filter_window_size / 2.0

    rows = []
    cols = []
    for i, val in enumerate(slant_range_subsampled_space_axis):
        left = np.searchsorted(slant_range_space_axis, val - window_half, side="left")
        right = np.searchsorted(slant_range_space_axis, val + window_half, side="right")
        js = np.arange(left, right)
        rows.extend([i] * len(js))
        cols.extend(js)

    return sp.sparse.coo_matrix(
        (np.ones(len(rows), dtype=dtype), (rows, cols)),
        shape=(slant_range_subsampled_space_axis.size, slant_range_space_axis.size),
    ).tocsr()
