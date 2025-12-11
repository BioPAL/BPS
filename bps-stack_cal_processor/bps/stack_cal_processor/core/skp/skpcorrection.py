# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities for Applying the SKP Ground Correction
------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from math import floor

import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.core.filtering import (
    ConvolutionBorderType,
    build_sparse_uniform_filter_matrix,
)
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_list_numeric_types_equal,
)
from bps.stack_cal_processor.core.skp.utils import SkpRuntimeError
from bps.stack_cal_processor.core.utils import interpolate_on_grid


def apply_skp_correction_multithreaded(
    *,
    stack: tuple[tuple[npt.NDArray[complex]], ...],
    skp_flattening_phases: tuple[npt.NDArray[float], ...],
    skp_calibration_phases: tuple[npt.NDArray[float], ...],
    azimuth_axis: npt.NDArray[float],
    range_axis: npt.NDArray[float],
    azimuth_subsampling_indices: npt.NDArray[int],
    range_subsampling_indices: npt.NDArray[int],
    quality_threshold: float,
    dtypes: EstimationDType,
    num_worker_threads: int = 1,
):
    """
    Apply (in-place) the SKP ground correction to the input images.

    Parameters
    ----------
    stack: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    skp_flattening_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg flattening phases from DSI of shape [Nazm' x Nrng'].

    skp_calibration_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg calibration phases from SKP of shape [Nazm' x Nrng'].

    azimuth_axis: npt.NDArray[float] [s]
        The [1 x Nazm] stack's relative azimuth axis.

    range_axis: npt.NDArray[float] [s]
        The [1 x Nrng] stack's relative range axis.

    azimuth_subsampling_indices: npt.NDArray[int]
        The SKP subsampling indices in azimuth. This must match
        the size of the SKP ground phases.

    range_subsampling_indices: npt.NDArray[int]
        The SKP subsampling indices in range. This must match
        the size of the SKP ground phases.

    quality_threshold: float
        The minimum quality for the SKP phase to be corrected.

    dtypes: EstimationDType
        Floating point precision used for the estiamtions.

    num_worker_threads: int = 1
        Number of parallel threads assigned to this task.

    Raises
    ------
    SkpRuntimeError, ValueError

    """
    # Check the input.
    if not (len(skp_flattening_phases) == len(skp_calibration_phases) == len(stack)):
        raise SkpRuntimeError("Incompatible stack and SKP phase interpolators")
    if num_worker_threads < 1:
        raise ValueError("Number of worker threads must be a positive integer")

    # Stack dimensions.
    num_images = len(stack)
    num_polarizations = len(stack[0])

    # Compute the SKP ground phases for calibration by usampling on the
    # reference primary grid.

    # The indices of the last azimuth and range with respect to the original
    # azimuth and range axis.
    last_azimuth = azimuth_subsampling_indices.reshape(-1)[-1] + 1
    last_range = range_subsampling_indices.reshape(-1)[-1] + 1

    # The interpolation axis.
    skp_azimuth_axis = azimuth_axis[azimuth_subsampling_indices]
    skp_range_axis = range_axis[range_subsampling_indices]

    # The portion of the image that can be calibrated.
    cal_azimuth_axis = azimuth_axis[0:last_azimuth]
    cal_range_axis = range_axis[0:last_range]

    skp_removal_phases = []
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Upsample the SKP ground phases to the full grid.
        def upsample_skp_correction_phases_fn(phi_flat, phi_skp):
            return interpolate_on_grid(
                phi_flat,
                axes_in=(skp_azimuth_axis, skp_range_axis),
                axes_out=(cal_azimuth_axis, cal_range_axis),
            ).astype(dtypes.float_dtype) + interpolate_on_grid(
                phi_skp,
                axes_in=(skp_azimuth_axis, skp_range_axis),
                axes_out=(cal_azimuth_axis, cal_range_axis),
                phase_interpolation=True,
            ).astype(dtypes.float_dtype)

        skp_removal_phases = list(
            executor.map(upsample_skp_correction_phases_fn, skp_flattening_phases, skp_calibration_phases)
        )
        assert_list_numeric_types_equal(skp_removal_phases, expected_dtype=dtypes.float_dtype)

    # Remove the SKP phases.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core removel routine.
        def skp_ground_phase_removal_fn(img_pol):
            img, pol = img_pol
            stack[img][pol][0:last_azimuth, 0:last_range] *= np.exp(
                1j * skp_removal_phases[img],
                dtype=dtypes.complex_dtype,
            )

        for _ in executor.map(
            skp_ground_phase_removal_fn,
            product(range(num_images), range(num_polarizations)),
        ):
            pass


def upsample_skp_phases_multithreaded(
    *,
    synth_phases: tuple[npt.NDArray[float], ...],
    skp_calibration_phases: npt.NDArray[float],
    azimuth_axis: npt.NDArray[float],
    range_axis: npt.NDArray[float],
    azimuth_estimation_axis: npt.NDArray[float],
    range_estimation_axis: npt.NDArray[float],
    output_azimuth_subsampling_step: int,
    output_range_subsampling_step: int,
    dsi_azimuth_filter_window_size: int | None = None,
    dsi_range_filter_window_size: int | None = None,
    dtypes: EstimationDType,
    num_worker_threads: int = 1,
) -> tuple[
    tuple[npt.NDArray[float], ...],
    tuple[npt.NDArray[float], ...],
    npt.NDArray[int],
    npt.NDArray[int],
]:
    """
    Compute the SKP ground phases.

    Parameters
    ----------
    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    skp_calibration_phases: npt.NDArray[float] [rad]
        The SKP calibration phases packed in an array of shape
        [Nimg x Nazm' x Nrng'].

    azimuth_axis: npt.NDArray[float] [s]
        The [1 x Nazm] azimuth axis of the stack.

    range_axis: npt.NDArray[float] [s]
        The [1 x Nrng] slant range axis of the stack.

    azimuth_estimation_axis: npt.NDArray[float] [s]
        The [1 x Nazm'] azimuth axis used for estimating the calibration
        phases. This must be consistent with the azimuth axis of the stack.

    range_estimation_axis: npt.NDArray[float] [s]
        The [1 x Nrng'] range axis used for estimating the calibration
        phases. This must be consistent with the range axis of the stack.

    output_azimuth_subsampling_step: int [px]
        Azimuth subsampling step of the output grid.

    output_range_subsampling_step: int [px]
        Range subsampling step of the output grid.

    dsi_azimuth_filter_window_size: int | None = None
        Possibly a window filter size for the DSI in azimuth. If None, the
        SKP estimation window size is used.

    dsi_range_filter_window_size: int | None = None,
        Possibly a window filter size for the DSI in range. If None, the SKP
        estimation window size is used.

    dtypes: EstimationDType
        Floating point precision used for the estiamtions.

    num_worker_threads: int = 1,
        Max number of threads for the job.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    tuple[npt.NDArray[float], ...] [rad]
        The [Nazm'' x Nrng''] SKP calibration phases on the SKP grid.

    tuple[npt.NDArray[float], ...] [rad]
        The [Nazm'' x Nrng''] flattening phases on the SKP grid.

    npt.NDArray[int]
        The [1 x Nazm''] array of indices of the SKP azimuth axis wrt
        the full data axis.

    npt.NDArray[int]
        The [1 x Nrng''] array of indices of the SKP range axis wrt
        the full data axis.

    """
    if len(synth_phases) == 0:
        raise SkpRuntimeError("empty synth phases")
    if len(synth_phases) != len(skp_calibration_phases):
        raise SkpRuntimeError("DSI and calibration phases are inconsistent")
    if synth_phases[0].shape[0] != azimuth_axis.size:
        raise SkpRuntimeError("DSI shape is inconsistent with azimuth axis")
    if synth_phases[0].shape[1] != range_axis.size:
        raise SkpRuntimeError("DSI shape is inconsistent with azimuth axis")
    if skp_calibration_phases[0].shape[0] != azimuth_estimation_axis.size:
        raise SkpRuntimeError("calibration phases are inconsistent with azimuth axis")
    if skp_calibration_phases[0].shape[1] != range_estimation_axis.size:
        raise SkpRuntimeError("calibration phases are inconsistent with range axis")

    # Start with a full matrix.
    azimuth_output_indices = np.arange(0, synth_phases[0].shape[0], output_azimuth_subsampling_step)
    range_output_indices = np.arange(0, synth_phases[0].shape[1], output_range_subsampling_step)

    if dsi_azimuth_filter_window_size is None:
        dsi_azimuth_filter_window_size = _odd(output_azimuth_subsampling_step)
    if dsi_range_filter_window_size is None:
        dsi_range_filter_window_size = _odd(output_range_subsampling_step)

    # Downsample in azimuth the DSI (with some lowering of resolution).
    #
    # NOTE: Uniform filtering on angles requires circular averaging, however
    # DSI are unwrapped, so we can interpolate linearly.
    #
    skp_flattening_phases = synth_phases

    if output_azimuth_subsampling_step > 1:
        (
            azimuth_filter_matrix,
            azimuth_output_indices,
        ) = build_sparse_uniform_filter_matrix(
            input_size=synth_phases[0].shape[0],
            axis=0,
            subsampling_step=output_azimuth_subsampling_step,
            uniform_filter_window_size=dsi_azimuth_filter_window_size,
            border_type=ConvolutionBorderType.ISOLATED,
            dtype=dtypes.float_dtype,
        )
        skp_flattening_phases = tuple(
            azimuth_filter_matrix @ phi.astype(dtypes.float_dtype) for phi in skp_flattening_phases
        )

    # Downsample in range the DSI (with some lowering of resolution).
    if output_range_subsampling_step > 1:
        (
            range_filter_matrix,
            range_output_indices,
        ) = build_sparse_uniform_filter_matrix(
            input_size=synth_phases[0].shape[1],
            axis=1,
            subsampling_step=output_range_subsampling_step,
            uniform_filter_window_size=dsi_range_filter_window_size,
            border_type=ConvolutionBorderType.ISOLATED,
            dtype=dtypes.float_dtype,
        )
        skp_flattening_phases = tuple(
            phi.astype(dtypes.float_dtype) @ range_filter_matrix for phi in skp_flattening_phases
        )

    # Apply the flattening phases to the SKP calibration phases.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The output axes.
        azimuth_output_axis = azimuth_axis[azimuth_output_indices]
        range_output_axis = range_axis[range_output_indices]

        # The core routine.
        def compute_skp_calibration_phases_fn(phi_cal):
            return -interpolate_on_grid(
                phi_cal,
                axes_in=(azimuth_estimation_axis, range_estimation_axis),
                axes_out=(azimuth_output_axis, range_output_axis),
                phase_interpolation=True,
            ).astype(dtypes.float_dtype)

        return (
            tuple(executor.map(compute_skp_calibration_phases_fn, skp_calibration_phases)),
            skp_flattening_phases,
            azimuth_output_indices,
            range_output_indices,
        )


def _odd(n: int) -> int:
    """Get the closest odd number"""
    return 2 * (floor(n / 2) * 2) - 1
