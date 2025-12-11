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
from itertools import product
from pathlib import Path
from typing import Any
from warnings import catch_warnings, simplefilter

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.io.metadata import (
    EPolarization,
    ESideLooking,
    MetaDataElement,
    StateVectors,
    SwathInfo,
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


def cross_pol_merging(
    *,
    data_list: list[npt.NDArray[complex]],
    swath_info_list: list[SwathInfo],
    polarization_list: list[str],
    lut_data: dict[str, npt.NDArray[float]],
    lut_axes: tuple[dict[str, npt.NDArray[PreciseDateTime]], dict[str, npt.NDArray[float]]],
    metadata_list: list[list[MetaDataElement]],
    xpol_merging_method: common.PolarisationCombinationMethodType | None,
) -> int:
    """
    Perform the cross-pol merging.

    Cross-pol merging executes the following operations:
       - HV: Drop V/H and keep only H/V as single cross-pol,
       - VH: Drop H/V and keep only V/H as single cross-pol,
       - AVERAGE: Create X/X := (H/V + V/H) / 2,
       - NONE: Keep everything as is.

    Note that the input arguments are modified.

    Parameters
    ----------
    data_list: list[npt.NDArray[complex]]
        The actual product data. These will be updated according to the
        cross-pol merging method.

    swath_info_list: list[SwathInfo]
        The swath infos. These will ne updated according to the
        cross-pol merging method.

    polarization_list: list[EPolarization]
        The available polarizations. This will be updated according to
        the cross-pol method.

    lut_data: dict[str, npt.NDArray[float]]
        The product LUTs. These will be updated according to the
        cross-pol merging method.

    lut_axes: tuple[dict[str, npt.NDArray[PreciseDateTime]], dict[str, npt.NDArray[float]]]
        THe LUT absolute axes, ordered as azimuth [UTC} and range [s].

    metadata_list: list[list[MetaDataElement]]
        Other L1a product metadata that are not image data. These
        metadata will be updated according to the cross-pol method.

    xpol_merging_method: PolarisationCombinationMethodType | None
        The cross-pol merging method.

    Raises
    ------
    StackPreProcessorRuntimeError

    Return
    ------
    int
        The number of remaining channels.

    """
    # If no cross-pol merging at all. We can stop here.
    if xpol_merging_method is None:
        return len(polarization_list)

    # The relevant LUT prefixes.
    xpol_luts = ["denoising"]
    if any(lut_name.startswith("rfiTimeMask") for lut_name in lut_data.keys()):
        xpol_luts.append("rfiTimeMask")
    if any(lut_name.startswith("rfiFreqMask") for lut_name in lut_data.keys()):
        xpol_luts.append("rfiFreqMask")

    # If we need to keep H/V, we need to have it and possibly we will drop V/H.
    if xpol_merging_method is common.PolarisationCombinationMethodType.HV:
        if EPolarization.hv.value not in polarization_list:
            raise StackPreProcessorRuntimeError("HV data not available but HV selected for polarization combination")
        if any(f"{lut}HV" not in lut_data for lut in xpol_luts):
            raise StackPreProcessorRuntimeError(
                "HV LUT data not available but HV selected for polarization combination"
            )

        # Drop the V/H index.
        if EPolarization.vh.value in polarization_list:
            vh_index = polarization_list.index(EPolarization.vh.value)
            polarization_list.pop(vh_index)
            _try_pop(data_list, index=vh_index)
            swath_info_list.pop(vh_index)
            for metadata in metadata_list:
                metadata.pop(vh_index)
        for lut in xpol_luts:
            lut_data.pop(f"{lut}HV", None)
            lut_axes[0].pop(f"{lut}HV", None)
            lut_axes[1].pop(f"{lut}HV", None)

    # If we need to keep V/H, we need to have it and possibly we will drop H/V.
    if xpol_merging_method is common.PolarisationCombinationMethodType.VH:
        if EPolarization.vh.value not in polarization_list:
            raise StackPreProcessorRuntimeError("V/H not available but selected for polarization combination")
        if any(f"{lut}VH" not in lut_data for lut in xpol_luts):
            raise StackPreProcessorRuntimeError(
                "VH LUT data not available but VH selected for polarization combination"
            )

        # Drop the H/V index.
        if EPolarization.hv.value in polarization_list:
            hv_index = polarization_list.index(EPolarization.hv.value)
            polarization_list.pop(hv_index)
            _try_pop(data_list, index=hv_index)
            swath_info_list.pop(hv_index)
            for metadata in metadata_list:
                metadata.pop(hv_index)
        for lut in xpol_luts:
            lut_data.pop(f"{lut}VH", None)
            lut_axes[0].pop(f"{lut}VH", None)
            lut_axes[1].pop(f"{lut}VH", None)

    # If we need to merge the cross-pols, we need to have them both.
    if xpol_merging_method is common.PolarisationCombinationMethodType.AVERAGE:
        if EPolarization.vh.value not in polarization_list or EPolarization.hv.value not in polarization_list:
            raise StackPreProcessorRuntimeError("V/H and H/V are both required when merging cross-polarizations")
        if any(f"{lut}{p}" not in lut_data for lut, p in product(xpol_luts, ("HV", "VH"))):
            raise StackPreProcessorRuntimeError("V/H and H/V LUTs are both required when merging cross-polarizations")

        # The cross-pol indices.
        hv_index = polarization_list.index(EPolarization.hv.value)
        vh_index = polarization_list.index(EPolarization.vh.value)

        # Substitute the H/V index with X/X.
        polarization_list[hv_index] = EPolarization.xx.value
        with catch_warnings():
            simplefilter("ignore")
            if len(data_list) > 0:
                if len(data_list) < max(hv_index, vh_index):
                    raise StackPreProcessorRuntimeError(
                        "Data stack ill-formed. Cannot access H/V and/or V/H polarization"
                    )
                data_list[hv_index] = (data_list[hv_index] + data_list[vh_index]) / 2
                data_list[hv_index][_invalid_data(data_list[hv_index])] = 0

        swath_info_list[hv_index].polarization = EPolarization.xx
        lut_data["denoisingXX"] = (lut_data["denoisingHV"] + lut_data["denoisingVH"]) / 4
        if "rfiTimeMask" in xpol_luts:
            lut_data["rfiTimeMaskXX"] = lut_data["rfiTimeMaskHV"] | lut_data["rfiTimeMaskVH"]
        if "rfiFreqMask" in xpol_luts:
            lut_data["rfiFreqMaskXX"] = lut_data["rfiFreqMaskHV"] | lut_data["rfiFreqMaskVH"]

        # Update the LUT axes.
        lut_axes[0]["denoisingXX"] = lut_axes[0]["denoisingHV"]
        lut_axes[1]["denoisingXX"] = lut_axes[1]["denoisingHV"]
        if "rfiTimeMask" in xpol_luts:
            lut_axes[0]["rfiTimeMaskXX"] = lut_axes[0]["rfiTimeMaskHV"]
            lut_axes[1]["rfiTimeMaskXX"] = lut_axes[1]["rfiTimeMaskHV"]
        if "rfiFreqMask" in xpol_luts:
            lut_axes[0]["rfiFreqMaskXX"] = lut_axes[0]["rfiFreqMaskHV"]
            lut_axes[1]["rfiFreqMaskXX"] = lut_axes[1]["rfiFreqMaskHV"]

        # Drop V/H index from metadata.
        polarization_list.pop(vh_index)
        _try_pop(data_list, index=vh_index)
        swath_info_list.pop(vh_index)
        for metadata in metadata_list:
            metadata.pop(vh_index)
        for lut, p in product(xpol_luts, ("HV", "VH")):
            lut_data.pop(f"{lut}{p}", None)
            lut_axes[0].pop(f"{lut}{p}", None)
            lut_axes[1].pop(f"{lut}{p}", None)

    return len(polarization_list)


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


def _invalid_data(data: npt.NDArray[complex]) -> npt.NDArray[bool]:
    """Mask of possibly problematic data."""
    return np.isnan(data) | (np.abs(data) == np.inf)


def _try_pop(data: list[npt.NDArray[complex]], *, index: int, default_value: Any = None):
    """Pop from list if possible, otherwise return default value."""
    try:
        return data.pop(index)
    except IndexError:
        return default_value
