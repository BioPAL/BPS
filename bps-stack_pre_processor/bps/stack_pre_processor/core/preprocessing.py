# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Stack Pre-processor Module
------------------------------
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.io.metadata import (
    EPolarization,
    ESideLooking,
    MetaDataElement,
    StateVectors,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common
from bps.stack_pre_processor.configuration import PrimaryImageSelectionConf
from bps.stack_pre_processor.core.utils import (
    StackPreProcessorRuntimeError,
    compute_faraday_index,
    compute_interferometric_baselines,
    compute_rfi_indices,
    sort_from_pivot,
)


def compute_critical_baseline(
    *,
    absolute_distance: float,
    incidence_angle: float,
    central_frequency: float,
    range_bandwidth: float,
) -> float:
    """
    Compute the critical baseline as

        0.5 * c/f0 * R/dR * tan(a)

    with

        c: speed of light
        f0: central frequency,
        R: slant-range distance,
        dR: slant-range resolution,
        a: incidence angle.

    Parameters
    ----------
    absolute_ditance: float [m]
        Slant-range distance from the sensor to the target on ground,
        e.g. center of the scene.

    incidence_angle: float [rad]
        Incidence angle to the target on ground, e.g. center of
        the scene.

    central_frequency: float [m]
        The RADAR's central (carrier) frequency.

    range_bandwidth: float [Hz]
        The bandwidth in range direction.

    Raises
    ------
    ValueError

    Return
    ------
    float [m]
        The critical baseline.

    """
    if absolute_distance <= 0:
        raise ValueError("Absolute slant-range distance must be positive")
    if incidence_angle <= 0:
        raise ValueError("Incidence angle must be positive")
    if central_frequency == 0:
        raise ValueError("Central frequency can't be zero")
    if range_bandwidth <= 0:
        raise ValueError("Bandwidth in range direction")

    range_resolution = 0.5 * sp.constants.speed_of_light / range_bandwidth
    wavelength = sp.constants.speed_of_light / central_frequency
    return 0.5 * wavelength * absolute_distance * np.tan(incidence_angle) / range_resolution


def compute_spatial_baselines(
    *,
    stack_state_vectors: tuple[StateVectors, ...],
    reference_state_vectors: StateVectors,
    reference_azimuth_time: PreciseDateTime,
    reference_range_time: float,
    satellite_side_looking: ESideLooking = ESideLooking.left_looking,
    pivot_fn: Callable = np.median,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """
    Compute the stack spatial baselines (normal).

    Parameters
    ----------
    stack_state_vectors: tuple[StateVectors, ...]
        State vectors of the stack image.

    reference_state_vectors: StateVectors
        State vectors of a reference stack image. The reference stack
        image can be selected arbitrarily.

    reference_azimuth_time: PreciseDateTime [UTC]
        A reference azimuth absolute time to compute the normal
        baselines.

    reference_range_time: float [s]
        A reference relative slant range time to compute the normal
        baselines.

    satellite_side_looking: ESideLooking
        The side-looking of the satellite. Defaulted to LEFT.

    pivot_fn: Callable
        A callable that returns the pivot wrt which the sorting
        is executed. Defaulted to np.median.

    Raises
    ------
    ValueError

    Return
    ------
    tuple[float, ...] [m]
        The spatial baseline of the stack (in normal direction).

    tuple[int, ...]
        The ordering permutation wrt to the median value.

    """
    # NOTE: 0 refers to the normal component.
    spatial_baselines = tuple(
        compute_interferometric_baselines(
            state_vectors_primary=reference_state_vectors,
            state_vectors_secondary=current_state_vectors,
            azimuth_time_primary=reference_azimuth_time,
            range_time_primary=reference_range_time,
            look_direction=satellite_side_looking.value,
        )[0]
        for current_state_vectors in stack_state_vectors
    )

    return (
        spatial_baselines,
        sort_from_pivot(spatial_baselines, pivot_fn=pivot_fn),
    )


def compute_temporal_baselines(
    *,
    stack_start_times: tuple[PreciseDateTime, ...],
    reference_start_time: PreciseDateTime,
    pivot_fn: Callable = np.median,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """
    Compute the temporal baselines of the stack.

    Parameters
    ----------
    start_times: tuple[PreciseDateTime, ...]
        The start acquisition time of the stack images.

    reference_start_time: PreciseDateTime
        The start time of the reference image of the stack. The
        reference image can be chosen arbitrarily.

    pivot_fn: Callable
        A callable that returns the pivot wrt which the sorting
        is executed. Defaulted to np.median.

    Return
    ------
    tuple[float, ...] [s]
        The temporal baselines.

    tuple[int, ...]
        The permutation that increasingly orders the temporal
        baselines wrt the median baseline.

    """
    temporal_baselines = tuple(np.float64(start_time - reference_start_time) for start_time in stack_start_times)
    return (
        temporal_baselines,
        sort_from_pivot(temporal_baselines, pivot_fn=pivot_fn),
    )


def baseline_ordering(
    baselines: npt.NDArray,
) -> npt.NDArray[int]:
    """
    Compute the baseline ordering index given the array baselines.

    Example
    -------

    Critical baselines: 0% = coreg primary.

         [-15%, 0%, -45%, -30%, 45%, 15%, 30%]

    Baseline ordering indices:

         [2, 3, 0, 1, 6, 4, 5]

    Parameters
    ----------
    baselines: npt.NDArray
        The baseline values (e.g. spatial, temporal etc.)

    Return
    ------
    npt.NDArray[int]
        A an array containing the baseline ordering indices.

    """
    return np.argsort(np.argsort(baselines))


def compute_coreg_primary_image_index(
    *,
    job_order_input_stack: tuple[Path, ...],
    job_order_primary_image: Path | None,
    config: PrimaryImageSelectionConf,
    stack_spatial_ordering: tuple[int, ...],
    stack_temporal_ordering: tuple[int, ...],
    reference_polarization: EPolarization,
    rfi_coherence_degradation_indices: tuple[dict[EPolarization, float]],
    faraday_decorrelation_indices: tuple[float, ...],
) -> tuple[int, common.PrimaryImageSelectionInformationType]:
    """
    Compute the index of the primary image.

    Parameters
    ----------
    job_order_primary_image: Optional[Path]
        The primary image optionally specified in the job order.

    job_order_input_stack: tuple[Path, ...]
        The input stack paths.

    config: PrimaryImageSelectionConf
        The criterion specs for selecting the primary image index.

    stack_spatial_ordering: tuple[int, ...]
        The index permutation that increasingly orders the stack
        images wrt the spatial (normal) baselines.

    stack_temporal_ordering: tuple[int, ...]
        The index permutation that increasingly orders the stack
        images wrt the temporal baselines.

    reference_polarization: EPolarization
        The reference polarization for coregistration.

    rfi_coherence_degradation_indices: tuple[dict[EPolarization, float]]
        The RFI indices. To disable set all to 1.

    faraday_decorrelation_indices: tuple[int, ...]
        The Faraday decorrelation indices. To disable, set all to 1.

    Raises
    ------
    StackPreProcessorRuntimeError

    Return
    ------
    int
        Index of the coregistation primary image.

    PrimaryImageInformation
        The actualize selection method.

    """
    # Check if the primary is specified in the job order (if it is None, it
    # won't be in the JobOrder input stack.
    if job_order_primary_image in job_order_input_stack:
        coreg_primary_image_index = job_order_input_stack.index(job_order_primary_image)
        bps_logger.warning(
            "Selected coreg primary image %s from job order (index=%d). "
            "Coregistration primary selection parameters from AUX-PPS will be ignored.",
            job_order_primary_image.name,
            coreg_primary_image_index,
        )
        return (
            coreg_primary_image_index,
            config.primary_image_selection_information,
        )

    if job_order_primary_image is not None:
        raise StackPreProcessorRuntimeError("Selected coreg primary image from job order %s is not in the stack")

    # Use the geometric selection criterion as first choice.
    if config.primary_image_selection_information is common.PrimaryImageSelectionInformationType.GEOMETRY:
        coreg_primary_image_index = stack_spatial_ordering[0]
        bps_logger.info(
            "Selected coreg primary image index %s using geometry (index=%d)",
            job_order_input_stack[coreg_primary_image_index].name,
            coreg_primary_image_index,
        )
        return (
            coreg_primary_image_index,
            config.primary_image_selection_information,
        )

    # Use temporal baseline as secondary option.
    if config.primary_image_selection_information is common.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE:
        coreg_primary_image_index = stack_temporal_ordering[0]
        bps_logger.info(
            "Selected coreg primary image index %s using temporal baseline (index=%d)",
            job_order_input_stack[coreg_primary_image_index].name,
            coreg_primary_image_index,
        )
        return (
            coreg_primary_image_index,
            config.primary_image_selection_information,
        )

    # Finally try using the RFI and/or FR.
    faraday_flag = config.primary_image_selection_information in (
        common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
        common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
    )

    faraday_validity = np.full((len(faraday_decorrelation_indices),), True)
    if faraday_flag:
        faraday_validity = np.asarray(faraday_decorrelation_indices) >= config.faraday_decorrelation_threshold
        bps_logger.info(
            "Valid images according to Faraday Rotation quality (threshold: %f): %s",
            config.faraday_decorrelation_threshold,
            faraday_validity.astype(np.int16).tolist(),
        )

    rfi_flag = config.primary_image_selection_information in (
        common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
        common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
    )
    rfi_validity = np.full((len(rfi_coherence_degradation_indices),), True)
    if rfi_flag:
        rfi_validity = (
            np.asarray([rfi[reference_polarization] for rfi in rfi_coherence_degradation_indices])
            >= config.rfi_decorrelation_threshold
        )
        bps_logger.info(
            "Valid images according to RFI degradation (threshold: %f): %s",
            config.rfi_decorrelation_threshold,
            rfi_validity.astype(np.int16).tolist(),
        )

    # If none is valid. Fall back to spatial baseline.
    combined_validity = faraday_validity & rfi_validity
    if not np.any(combined_validity):
        coreg_primary_image_index = stack_spatial_ordering[0]
        bps_logger.info(
            "Combined RFI/FR are all invalid. Selected coreg primary image %s using geometry (index=%d)",
            job_order_input_stack[coreg_primary_image_index].name,
            coreg_primary_image_index,
        )
        return (
            coreg_primary_image_index,
            common.PrimaryImageSelectionInformationType.GEOMETRY,
        )

    bps_logger.info(
        "Valid images according to FR and/or RFI: %s",
        combined_validity.astype(np.int16).tolist(),
    )

    # We take the spatially best that is also valid.
    valid_image_indices = [index for index in stack_spatial_ordering if combined_validity[index]]
    if len(valid_image_indices) == 0:
        raise StackPreProcessorRuntimeError("Image selection failed")

    coreg_primary_image_index = valid_image_indices[0]
    bps_logger.info(
        "Selected coreg primary image %s using RFI and/or FR and geometry (index=%d)",
        job_order_input_stack[coreg_primary_image_index].name,
        coreg_primary_image_index,
    )

    return (
        coreg_primary_image_index,
        config.primary_image_selection_information,
    )


def prepare_stack_data(
    *,
    common_polarizations: tuple[EPolarization, ...],
    data_list: list[npt.NDArray[complex]],
    polarization_list: list[str],
    metadata_list: list[list[MetaDataElement]],
):
    """
    Pack the stack data for processing. Downstream of this function, data and
    metadata will be polarization aligned and containing only the common
    polarizations.

    Parameters
    ----------
    common_polarizations: tuple[EPolarization, ...]
        The polarizations available/usable in the stack.

    data_list: list[npt.NDArray[complex]]
        The list of data in the product ordered by polarization.

    polarization_list: list[str]
        The polarization available in the product.

    metadata_list: list[list[MetaDataElement]]
        Other L1a metadata products (raster info, dataset info etc.).

    Raises
    ------
    StackPreProcessorRuntimeError

    """
    if any(p.value not in polarization_list for p in common_polarizations):
        raise StackPreProcessorRuntimeError(f"{common_polarizations} are not all available in the stack")
    if len(data_list) > 0 and len(data_list) != len(polarization_list):
        raise StackPreProcessorRuntimeError("Data and polarizations are inconsistent")
    if any(len(m) != len(polarization_list) for m in metadata_list):
        raise StackPreProcessorRuntimeError("Metadata and polarizations are inconsistent")

    # The permutation that reorders the stack.
    reordering = [polarization_list.index(p.value) for p in common_polarizations]
    for lst in [data_list, polarization_list, *metadata_list]:
        if len(lst) > 0:
            lst[:] = [lst[i] for i in reordering]


def compute_rfi_degradation_indices(
    lut_data: dict[str, npt.NDArray[float]],
    polarizations: tuple[EPolarization, ...],
) -> dict[EPolarization, float]:
    """
    Compute the RFI degradation indices.

    Parameters
    ----------
    lut_data: dict[str, npt.NDArray[float]]
        The L1a product LUTs.

    polarizations: tuple[EPolarization]
        The selected polarization. If not available in LUTs, 1
        will be used.

    Return
    ------
    dict[EPolarization, float]
        Polarization/RFI map.

    """
    return {
        pol: compute_rfi_indices(
            # Use RFI time mask, if available.
            lut_data.get(
                "rfiTimeMask{}".format(pol.value.replace("/", "")),
                # If not, use the RFI frequency mask.
                lut_data.get(
                    "rfiFreqMask{}".format(pol.value.replace("/", "")),
                    # If the freq mask is also missing, we ignore the RFI
                    # (setting to False means no RFI whatsoever).
                    np.array([False]),
                ),
            )
        )
        for pol in polarizations
    }


def compute_faraday_decorrelation_index(
    lut_data: dict[str, npt.NDArray[float]],
) -> float:
    """Compute the Faraday decorrelation index."""
    return compute_faraday_index(
        lut_data.get(
            "faradayRotation",
            np.ones(1),  # Will result in FR=1, so ignored.
        ),
    )
