# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Slow-varying Ionosphere Removal's Preprocessor
----------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.configuration import StackDataSpecs
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.iob.utils import IobRuntimeError
from bps.stack_cal_processor.core.signal_processing import (
    prepare_spectral_window_removal,
    spectral_window_removal,
)


def preprocess_input_stack_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[complex], ...],
    reference_polarization_index: int,
    stack_specs: StackDataSpecs,
    estimation_dtypes: EstimationDType,
    num_worker_threads: int,
) -> tuple[npt.NDArray[float], ...]:
    """
    Preprocess the stack images for a selected polarization by removing
    the range spectral window (spectral whitening) and by compensating
    the synthetic phases from DEM.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images
        of shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    reference_polarization_index: int
        The index of the reference polarization used for the slow-varying
        ionosphere estimation.

    stack_specs: StackCalConf.StackDataSpecs
        The stack specs object.

    estimation_dtypes: EstimationDType
        The floating point precision to be used for the estimation.

    num_worker_threads: int
        Number of threads assigned to the job.

    Return
    ------
    tuple[npt.NDArray[complex], ...]
        The Nimg preprocessed images for the selected polazation.

    """
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The preprocessing core routine. This executes both the range
        # spectral window removal and the synthtetic phase removal.
        def preprocessing_fn(image, synth, win_param, win_band):
            return preprocess_stack_image(
                image=image[reference_polarization_index],
                synth=synth,
                range_compression_window_parameter=win_param,
                range_compression_window_band=win_band,
                range_sampling_step=stack_specs.range_sampling_step,
                dtypes=estimation_dtypes,
            )

        # Spawn the preprocessing step by distributing the computation over
        # the images (but keeping fixed the polarization).
        return tuple(
            executor.map(
                preprocessing_fn,
                stack_images,
                synth_phases,
                stack_specs.range_compression_window_parameters,
                stack_specs.range_compression_window_bands,
            )
        )


def preprocess_stack_image(
    *,
    image: npt.NDArray[complex],
    synth: npt.NDArray[float],
    range_compression_window_parameter: float,
    range_compression_window_band: float,
    range_sampling_step: float,
    dtypes: EstimationDType,
) -> npt.NDArray[complex]:
    """
    Preprocess the a stack image by sequentially executing:

      - Whitening of the range spectral window
      - Compensation of the synthetic phases from DEM.

    Parameters
    ----------
    image: npt.NDArray[complex]
        A [Nazm x Nrng] stack image.

    synth: npt.NDArray[float] [rad]
        The corresponding [Nazm x Nrng] synthetic phases from DEM.

    range_compression_window_parameter: float [%]
        The range preprocessing window parameter.

    range_compression_window_band: float [%]
        The range preprocessing bandwidth.

    range_sampling_step: float [Hz]
        The range sampling step.

    dtypes: EstimationDType
        The estimation floating point accuracy.

    Return
    ------
    npt.NDArray[complex]
        The processed image data.

    """
    # Filter parameters to remove the preprocessing window.
    (
        sampling_frequencies,
        cutoff_sampling_frequency,
        spectral_window,
    ) = prepare_spectral_window_removal(
        compression_window_parameter=range_compression_window_parameter,
        compression_window_band=range_compression_window_band,
        sampling_step=range_sampling_step,
        num_samples=image.shape[1],
    )
    if sampling_frequencies is None or cutoff_sampling_frequency is None or spectral_window is None:
        raise IobRuntimeError("Cannot compute spectral whitening window")

    # Apply the range spectral window removal and store the range indices
    # associated to the frequencies below the cutoff value and compensated the
    # synthetic phases from DEM.
    return spectral_window_removal(
        image=image,
        sampling_frequencies=sampling_frequencies,
        spectral_window=spectral_window,
        cutoff_sampling_frequency=cutoff_sampling_frequency,
        axis=1,  # Range direction.
        dtype=dtypes.complex_dtype,
    ) * np.exp(1j * synth, dtype=dtypes.complex_dtype)
