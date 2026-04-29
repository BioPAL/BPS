# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Phase Correction for the Multi-Squint Calibration Module
--------------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.msc.ionoestimation import IonosphereEstimationOutput
from bps.stack_cal_processor.core.msc.utils import MscRuntimeError
from bps.stack_cal_processor.core.utils import interpolate_on_grid, percentage_completed_msg


def ionosphere_inplace_correction_multithreaded(
    *,
    mpol_image: tuple[npt.NDArray[complex], ...],
    ionosphere_estimation_output: IonosphereEstimationOutput,
    azimuth_space_resolution: float,
    azimuth_space_axis: npt.NDArray[float],
    ms_azimuth_indices: npt.NDArray[int],
    slant_range_space_axis: npt.NDArray[float],
    ms_range_indices: npt.NDArray[int],
    subband_axis: npt.NDArray[float],
    subband_azimuth_resolution: float,
    wavelength: float,
    dtypes: EstimationDType,
    aperture_inflation: float = 1.1,
    max_num_threads: int = 1,
):
    """Correct in-place the ionosphere.

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[complex], ...]
        A tuple of [Naz x Nrg] arrays that stores a multi-polarimetric
        single-look complex.

    ionosphere_estimation_output: IonosphereEstimationOutput
        The output of the ionosphere estimation step.

    azimuth_space_resolution: float  [m]
        The spatial resolution in along-track direction.

    azimuth_space_axis: npt.NDArray[float]  [m]
        Array of shape [1 x Naz] that stores the space axis in along-track
        direction.

    ms_azimuth_indices: npt.NDArray[int]
        Array of shape [1 x Naz'] that stores the indices associated the the
        along-track axis of the Multi-Squint coherence matrix.

    slant_range_space_axis: npt.NDArray[float]
        Array of shape [1 x Nrg] that stores the space axis in slant-range
        direction.

    ms_range_indices: npt.NDArray[int]
        Array of shape [1 x Naz'] that stores the indices associated the the
        range axis of the Multi-Squint coherence matrix.

    subband_axis: npt.NDArray[float]  [1/m]
        Center frequencies of the Multi-Squint coherence matrix's sub-bands.

    subband_azimuth_resolution: float  [m]
        Sub-band azimuth resolution.

    wavelength: float  [m]
        The RADAR wavelength.

    dtypes: EstimationDType
        The floating-point precision used for the computations.

    aperture_inflation: float = 1.1
        An inflation coefficient applied to the squint aperture that mitigates
        the ripples during focusing/refocusing. Recommended value is 1.1 (as
        per experimental evindence).

    max_num_threads: int = 1
        Number of threads assigned to the job.

    Raises
    ------
    MscRuntimeError

    """
    num_iono_layers = len(ionosphere_estimation_output.ionosphere_layer_ranges)
    if num_iono_layers == 1:
        bps_logger.info("Compensate ionosphere via single-layer correction")
        single_layer_ionosphere_inplace_correction_multithreaded(
            mpol_image=mpol_image,
            azimuth_space_axis=azimuth_space_axis,
            azimuth_iono_space_axis=ionosphere_estimation_output.azimuth_iono_space_axis,
            azimuth_space_resolution=azimuth_space_resolution,
            slant_range_space_axis=slant_range_space_axis,
            slant_range_ms_space_axis=slant_range_space_axis[ms_range_indices],
            ionosphere_phase_screen=np.squeeze(ionosphere_estimation_output.iono_layer_phase_screens),
            ionosphere_range=ionosphere_estimation_output.ionosphere_layer_ranges[0],
            wavelength=wavelength,
            dtypes=dtypes,
            aperture_inflation=aperture_inflation,
            max_num_threads=max_num_threads,
        )
    elif num_iono_layers > 1:
        bps_logger.info("Compensate ionosphere via sub-band refocusing")
        subband_inplace_refocusing_multithreaded(
            mpol_image=mpol_image,
            ms_ionosphere_phases=ionosphere_estimation_output.ms_iono_phases,
            azimuth_space_axis=azimuth_space_axis,
            slant_range_space_axis=slant_range_space_axis,
            ms_azimuth_indices=ms_azimuth_indices,
            ms_range_indices=ms_range_indices,
            subband_axis=subband_axis,
            subband_azimuth_resolution=subband_azimuth_resolution,
            dtypes=dtypes,
            max_num_threads=max_num_threads,
        )
    else:
        raise MscRuntimeError(f"Invalid {num_iono_layers=}")


def subband_inplace_refocusing_multithreaded(
    *,
    mpol_image: tuple[npt.NDArray[complex], ...],
    ms_ionosphere_phases: npt.NDArray[float],
    azimuth_space_axis: npt.NDArray[float],
    slant_range_space_axis: npt.NDArray[float],
    ms_azimuth_indices: npt.NDArray[int],
    ms_range_indices: npt.NDArray[int],
    subband_axis: npt.NDArray[float],
    subband_azimuth_resolution: float,
    dtypes: EstimationDType,
    fft_num_threads: int = 3,
    max_num_threads: int = 7,
):
    """
    Compensate in-place for ionospheric phase in the squint-frequency domain

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[complex], ...]
        A tuple of [Naz x Nrg] arrays that stores a multi-polarimetric
        single-look complex.

    ms_ionosphere_phases: npt.NDArray[float]  [rad]
        Array of shape [Naz' x Nrg' x Nf] that stores the ionospheric phase
        screen.

    azimuth_space_axis: npt.NDArray[float]  [m]
        Array of shape [1 x Naz] that stores the space axis in along-track
        direction.

    slant_range_space_axis: npt.NDArray[float]
        Array of shape [1 x Nrg] that stores the space axis in slant-range
        direction.

    ms_azimuth_indices: npt.NDArray[int]
        Array of shape [1 x Naz'] that stores the indices associated the the
        along-track axis of the Multi-Squint coherence matrix.

    ms_range_indices: npt.NDArray[int]
        Array of shape [1 x Naz'] that stores the indices associated the the
        range axis of the Multi-Squint coherence matrix.

    subband_axis: npt.NDArray[float]  [1/m]
        Center frequencies of the Multi-Squint coherence matrix's sub-bands.

    subband_azimuth_resolution: float  [m]
        Sub-band azimuth resolution.

    dtypes: EstimationDType
        The floating-point precision used for the computations.

    fft_num_threads: int = 3
        Number of threads used by the FFT.

    max_num_threads: int = 7
        Total number of threads assigned to the job.

    """
    # Convert once the ionospheric phase screens into phasors.
    ms_ionosphere_phasors = np.exp(1j * ms_ionosphere_phases, dtype=dtypes.complex_dtype)

    # Initialize the refocusing.
    num_azimuths = azimuth_space_axis.size
    azimuth_space_step = np.diff(azimuth_space_axis[:2])[0]

    num_freqs = int(2 ** np.ceil(np.log2(azimuth_space_axis.size)))
    freq_axis = (sp.fft.fftshift(sp.fft.fftfreq(num_freqs)) / azimuth_space_step).astype(dtypes.float_dtype)

    # Build the SAR grid points for interpolation.
    slant_range_space_points, azimuth_space_points = np.meshgrid(
        slant_range_space_axis, azimuth_space_axis, indexing="ij"
    )
    sar_grid = np.stack(
        [slant_range_space_points.ravel(), azimuth_space_points.ravel()],
        axis=-1,
    )

    # Build the conversion from MS to sar grid.
    ms_azimuth_space_axis = azimuth_space_axis[ms_azimuth_indices]
    ms_slant_range_space_axis = slant_range_space_axis[ms_range_indices]
    ms_to_sar_grid = np.stack(
        [
            (sar_grid[:, 0] - ms_slant_range_space_axis[0]) / np.diff(ms_slant_range_space_axis[:2])[0],
            (sar_grid[:, 1] - ms_azimuth_space_axis[1]) / np.diff(ms_azimuth_space_axis[:2])[0],
        ],
        axis=0,
    )

    # Prepare refocusing. We place the azimuth axis as last to speed up the FFT.
    mpol_image_refoc_fft = sp.fft.fft(
        np.moveaxis(mpol_image, (0, 1, 2), (0, 2, 1)),
        n=num_freqs,
        axis=-1,
        workers=fft_num_threads,
    )

    with ThreadPoolExecutor(max_workers=max_num_threads // fft_num_threads) as executor:
        # Prepare the output.
        for image in mpol_image:
            image[...] = np.zeros_like(image)

        # Pre-compute the filter normalization that preserve the data energy.
        filter_normalization = _compute_filter_normalization(
            freq_axis=freq_axis,
            subband_axis=subband_axis,
            subband_azimuth_resolution=subband_azimuth_resolution,
            dtype=dtypes.float_dtype,
        )

        # The core refocusing method.
        def refocus_fn(band):
            mpol_image_subband = sp.fft.ifft(
                mpol_image_refoc_fft
                * (
                    _rectpulse_ifftshift(
                        (freq_axis - subband_axis[band]) * subband_azimuth_resolution,
                        dtype=dtypes.float_dtype,
                    )
                    / filter_normalization
                ),
                n=num_freqs,
                axis=-1,
                workers=fft_num_threads,
            )[..., :num_azimuths]
            iono_band_phi = _interp2d(
                ms_ionosphere_phasors[..., band].T,
                ms_to_sar_grid=ms_to_sar_grid,
                dtype=dtypes.complex_dtype,
            ).reshape(slant_range_space_axis.size, azimuth_space_axis.size)
            return mpol_image_subband * np.conj(iono_band_phi)

        for completed, mpol_image_subband_refoc in enumerate(
            executor.map(refocus_fn, range(subband_axis.size)),
            start=1,
        ):
            for image, image_subband_refoc in zip(mpol_image, mpol_image_subband_refoc):
                image[...] += image_subband_refoc.T

            if completed % round(subband_axis.size / 10) == 0:
                bps_logger.info(percentage_completed_msg(completed, total=subband_axis.size))

        bps_logger.info(percentage_completed_msg(subband_axis.size, total=subband_axis.size))


def single_layer_ionosphere_inplace_correction_multithreaded(
    *,
    mpol_image: tuple[npt.NDArray[complex], ...],
    azimuth_space_axis: npt.NDArray[float],
    azimuth_iono_space_axis: npt.NDArray[float],
    azimuth_space_resolution: float,
    slant_range_space_axis: npt.NDArray[float],
    slant_range_ms_space_axis: npt.NDArray[float],
    ionosphere_phase_screen: npt.NDArray[float],
    ionosphere_range: float,
    wavelength: float,
    dtypes: EstimationDType,
    aperture_inflation: float = 1.1,
    max_num_threads: int = 1,
):
    """
    Compensate (in-place) the ionospheric phase screen via defocusing/refocusing.

    This function compensates for ionospheric distortion by:
    1. Project the focused SAR image to the ionospheric altitude (ri).
    2. Upsample the estimated ionospheric phase screen to the image grid.
    3. Compensate the phase in the celestial (defocused) domain.
    4. Re-focus the corrected signal back to the ground-range geometry.

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[complex], ...]
        A tuple of [Naz x Nrg] arrays that stores a multi-polarimetric
        single-look complex.

    azimuth_space_axis: npt.NDArray[float]  [m]
        Array of shape [1 x Naz] that stores the space axis in along-track
        direction.

    azimuth_iono_space_axis: npt.NDArray[float]
        Array of shape [1 x Niaz] that stores the space axis in along-track
        direction at the ionosphere altitude.

    azimuth_space_resolution: float  [m]
        Spatial resolution in along-track direction.

    slant_range_space_axis: npt.NDArray[float]
        Array of shape [1 x Nrg] that stores the space axis in slant-range
        direction.

    slant_range_ms_space_axis: npt.NDArray[float]
        Array of shape [1 x Nrg'] that stores the space axis in slant-range
        direction associated to the Multi-Squint coherence matrix.

    ionosphere_phase_screen: npt.NDArray[float]  [rad]
        Array of shape [Niaz x Nrg'] that stores the ionospheric phase
        screen.

    ionosphere_range: float  [m]
        Height of the ionosphere (single) layer.

    wavelength: float  [m]
        The RADAR wavelength.

    dtypes: EstimationDType
        The floating-point precision used for the computations.

     aperture_inflation: float = 1.1
        An inflation coefficient applied to the squint aperture that mitigates
        the ripples during focusing/refocusing. Recommended value is 1.1 (as
        per experimental evindence).

    max_num_threads: int = 1
        Number of threads assigned to the job.

    """
    # Step 1. Defocus the image to the ionosphere height.
    bps_logger.info("Defocus the data to the ionospheric height")
    mpol_defoc_image, iono_filter, azimuth_iono_defoc_space_axis = _defocus_to_ionosphere_multithreaded(
        mpol_image=mpol_image,
        azimuth_space_axis=azimuth_space_axis,
        azimuth_space_resolution=azimuth_space_resolution,
        ionosphere_range=ionosphere_range,
        wavelength=wavelength,
        dtypes=dtypes,
        aperture_inflation=aperture_inflation,
        max_num_threads=max_num_threads,
    )

    # Step 2. Compensate the ionosphere phase screen (no need of multi-threading).
    bps_logger.info("Compensate ionosphere phase screen")
    iono_phasors = np.exp(
        -1j
        * interpolate_on_grid(
            np.squeeze(ionosphere_phase_screen),
            axes_in=(azimuth_iono_space_axis, slant_range_ms_space_axis),
            axes_out=(azimuth_iono_defoc_space_axis, slant_range_space_axis),
            phase_interpolation=True,
            degree_x=1,
            degree_y=1,
        ),
        dtype=dtypes.complex_dtype,
    )
    for image in mpol_defoc_image:
        image *= iono_phasors.T

    # Step 3. Refocus to the ground (in place).
    bps_logger.info("Refocus the data to ground level")
    _refocus_to_ground_multithreaded(
        mpol_defoc_image=mpol_defoc_image,
        out_refoc_image=mpol_image,
        azimuth_space_axis=azimuth_space_axis,
        azimuth_space_resolution=azimuth_space_resolution,
        ionosphere_filter=iono_filter,
        ionosphere_range=ionosphere_range,
        wavelength=wavelength,
        aperture_inflation=aperture_inflation,
        max_num_threads=max_num_threads,
    )


def _refocus_to_ground_multithreaded(
    *,
    out_refoc_image: tuple[npt.NDArray[complex], ...],
    mpol_defoc_image: tuple[npt.NDArray[complex], ...],
    azimuth_space_axis: npt.NDArray[float],
    azimuth_space_resolution: float,
    ionosphere_filter: npt.NDArray[float],
    ionosphere_range: float,
    wavelength: float,
    aperture_inflation: float,
    filter_size_normalize: int = 100,
    max_num_threads: int = 1,
) -> tuple[npt.NDArray[complex], ...]:
    """Projects data from the ionospheric grid back to the ground."""
    # Synthetic aperture size at ionospheric height.
    azimuth_space_step = np.diff(azimuth_space_axis[:2])[0]
    num_azimuths = azimuth_space_axis.size
    aperture_size = 0.5 * aperture_inflation * wavelength * ionosphere_range / azimuth_space_resolution
    filter_half_length_samples = round(0.5 * aperture_size / azimuth_space_step)

    with ThreadPoolExecutor(max_workers=max_num_threads) as executor:
        # Frequency-domain convolution.
        num_ionosphere_samples = num_azimuths + 2 * filter_half_length_samples
        fft_length = int(2 ** np.ceil(np.log2(num_ionosphere_samples + ionosphere_filter.size - 1)))
        refoc_filter = sp.fft.fft(np.conj(ionosphere_filter), n=fft_length)
        refoc_filter /= np.mean(np.abs(refoc_filter[0:filter_size_normalize]))
        refoc_filter = refoc_filter[np.newaxis, :]

        # Bounds for the delay induce by the defocusing step.
        aligned_begin = filter_half_length_samples
        aligned_end = aligned_begin + num_ionosphere_samples

        # Bounds for extracting the original image size at the ground.
        ground_begin = filter_half_length_samples
        ground_end = ground_begin + num_azimuths

        def refocus_fn(image, out_image):
            refoc_image = sp.fft.ifft(
                sp.fft.fft(image, n=fft_length, axis=1) * refoc_filter,
                axis=1,
            )
            refoc_image = refoc_image[:, : num_ionosphere_samples + ionosphere_filter.size]
            refoc_image = refoc_image[:, aligned_begin:aligned_end]
            out_image[...] = refoc_image[:, ground_begin:ground_end].T

        for _ in executor.map(refocus_fn, mpol_defoc_image, out_refoc_image):
            pass


def _defocus_to_ionosphere_multithreaded(
    *,
    mpol_image: tuple[npt.NDArray[complex], ...],
    azimuth_space_axis: npt.NDArray[float],
    azimuth_space_resolution: float,
    ionosphere_range: float,
    wavelength: float,
    dtypes: EstimationDType,
    aperture_inflation: float,
    filter_size_normalize: int = 100,
    max_num_threads: int = 1,
) -> tuple[
    tuple[npt.NDArray[complex], ...],
    npt.NDArray[float],
    npt.NDArray[float],
]:
    """Projects ground-range SAR data to an ionospheric altitude."""
    # Synthetic aperture size at ionospheric height.
    azimuth_space_step = np.diff(azimuth_space_axis[:2])[0]
    num_azimuths = azimuth_space_axis.size

    aperture_size = 0.5 * aperture_inflation * wavelength * ionosphere_range / azimuth_space_resolution
    filter_half_length_samples = round(0.5 * aperture_size / azimuth_space_step)

    # Compute the filter.
    filter_space_axis = azimuth_space_step * np.arange(
        -filter_half_length_samples, filter_half_length_samples + 1, dtype=dtypes.float_dtype
    )

    # Extended ionosphere azimuth axis.
    ionosphere_azimuth_axis = np.mean(azimuth_space_axis) + np.linspace(
        -0.5 * (filter_space_axis.size + num_azimuths - 1) * azimuth_space_step,
        0.5 * (filter_space_axis.size + num_azimuths - 1) * azimuth_space_step,
        filter_space_axis.size + num_azimuths - 1,
        dtype=dtypes.float_dtype,
    )

    # Ionosphere impulse response (spherical wave phase).
    ionosphere_filter = np.exp(
        -4j * np.pi / wavelength * np.sqrt(ionosphere_range**2 + filter_space_axis**2),
        dtype=dtypes.complex_dtype,
    )

    with ThreadPoolExecutor(max_workers=max_num_threads) as executor:
        # Frequency-domain convolution.
        num_ionosphere_samples = num_azimuths + 2 * filter_half_length_samples
        fft_length = int(2 ** np.ceil(np.log2(num_ionosphere_samples + filter_space_axis.size - 1)))
        defoc_filter = sp.fft.fft(ionosphere_filter, n=fft_length)
        defoc_filter /= np.mean(np.abs(defoc_filter[0:filter_size_normalize]))
        defoc_filter = defoc_filter[np.newaxis, :]

        # The core defocusing function.
        def defocus_fn(image):
            return sp.fft.ifft(
                sp.fft.fft(image.T, n=fft_length, axis=1) * defoc_filter,
                axis=1,
            )[:, :num_ionosphere_samples]

        return (
            tuple(executor.map(defocus_fn, mpol_image)),
            ionosphere_filter,
            ionosphere_azimuth_axis,
        )


def _compute_filter_normalization(
    freq_axis: npt.NDArray[float],
    subband_axis: npt.NDArray[float],
    subband_azimuth_resolution: float,
    dtype: np.dtype,
) -> npt.NDArray[float]:
    """Compute the subband refocusing filter's normalization term."""
    # NOTE: No need of running this multi-threaded since it's quite fast.
    filter_normalization = np.zeros(freq_axis.shape, dtype=dtype)
    for subband in subband_axis:
        filter_normalization += _rectpulse_ifftshift(
            (freq_axis - subband) * subband_azimuth_resolution,
            dtype=dtype,
        )
    filter_normalization[np.abs(filter_normalization) < np.finfo(dtype).tiny] = 1.0
    return filter_normalization


def _interp2d(
    ms_iono_phasor: npt.NDArray[complex],
    ms_to_sar_grid: npt.NDArray[float],
    dtype: np.dtype,
) -> npt.NDArray[complex]:
    """Apply linear interpolation"""
    # Linear interpolation with out-of-bounds as 0
    return sp.ndimage.map_coordinates(ms_iono_phasor, ms_to_sar_grid, order=1, mode="constant", cval=0).astype(dtype)


def _rectpulse_ifftshift(values: npt.NDArray[float], dtype: np.dtype) -> npt.NDArray[float]:
    """The iFFT-shift of a rectangular pulse kernel."""
    return sp.fft.ifftshift((np.abs(values) <= 0.5).astype(dtype))
