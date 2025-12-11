# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Memory Checks for the Stack Processor
-------------------------------------
"""

from pathlib import Path

import numpy as np
from bps.stack_cal_processor.configuration import (
    AZF_NAME,
    SKP_NAME,
)
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.translate_job_order import (
    InvalidStackJobOrder,
)
from bps.transcoder.utils.product_name import (
    InvalidBIOMASSProductName,
    parse_l1product_name,
)

# We need to suffix _MB in this module, so we suppress warnings.
# pylint: disable=invalid-name

# Some extra amount of RAM usage to account for deviations from the reported
# values, and to be on the safe side.
_STANDARD_SAFETY_OVERHEAD_MB = 1500  # 1.5 GB.

# Nominal and maximum swath duration expected as input for the stack.
NOMINAL_SWATH_DURATION = 21.0  # [s]
CRITICAL_SWATH_DURATION = 29.0  # [s].

# Some required extra disk space to make sure that we do not run out of memory
# when we forecast the required space.
_WORKING_DIR_SAFETY_OVERHEAD_MB = 250  # [MB].
_L1A_PRODUCT_SAFETY_OVERHEAD_MB = 100  # [MB].
_L1C_PRODUCT_SAFETY_OVERHEAD_MB = 100  # [MB].


class InsufficientMemoryAllocationError(InvalidStackJobOrder):
    """Handle insufficient RAM allocated in the JobOrder."""

    def __init__(
        self,
        *,
        allocated_memory: float,
        unsupported_modules: dict[str, float],
    ):
        """Initialize the error message."""
        super().__init__(
            "Allocated RAM memory [MB] {} not sufficient for {}".format(
                allocated_memory,
                {module: f"Forecasted load [MB]: {memory}" for module, memory in unsupported_modules.items()},
            )
        )


class InsufficientDiskSpaceAllocationError(InvalidStackJobOrder):
    """It's triggered when the JobOrder does not allocate enough disk space."""


def raise_if_disk_memory_allocated_is_insufficient(job_order: StackJobOrder):
    """
    Forecast the (maximum) disk space used by the working directory, L1a input
    products and L1c output products. The value is obtained by the actual sizes
    of a TOM STA_P executions.

    Parameters
    ----------
    job_order: StackJobOrder
         The STA_P JobOrder object.

    Raises
    ------
    InsufficientDiskSpaceAllocationError

    """
    num_frames = len(job_order.input_stack)
    mean_swath_duration = _get_mean_swath_duration(job_order.input_stack)
    # fmt:off
    forecast_MB = (
        5142.8571 * num_frames + _WORKING_DIR_SAFETY_OVERHEAD_MB  # Working dir.
        + (1234.3334 + _L1A_PRODUCT_SAFETY_OVERHEAD_MB) * num_frames  # Input L1a products.
        + (1233.3334 + _L1C_PRODUCT_SAFETY_OVERHEAD_MB) * num_frames  # Output L1c products.
    ) * mean_swath_duration / CRITICAL_SWATH_DURATION
    # fmt:on
    if job_order.device_resources.available_space < forecast_MB:
        raise InsufficientDiskSpaceAllocationError(
            "Expected disk space footprint [GB]: {:.2f}. Allocated [GB]: {:.2f}".format(
                forecast_MB / 1e3,
                job_order.device_resources.available_space / 1e3,
            )
        )


def raise_if_memory_allocated_is_insufficient(
    job_order: StackJobOrder,
    aux_pps: AuxiliaryStaprocessingParameters,
):
    """
    Raise an error if the memory allocated by the JobOrder is expected
    to be insufficient for the STA_P task specified by JobOrder/AuxPPS.

    Parameters
    ----------
    job_order: StackJobOrder
        The STA_P job order object.

    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS object.

    Raises
    ------
    InsufficientMemoryAllocationError

    """
    # The stack size and the allocated memory.
    num_frames = len(job_order.input_stack)
    allocated_memory_MB = job_order.device_resources.available_ram

    # The mean swath duration.
    mean_swath_duration = _get_mean_swath_duration(job_order.input_stack)

    # Check that all processes will have enough memory.
    unsupported_processes = {}

    # Check if we have enough memory allocated for the pre-processor.
    preproc_forecast_load_MB = preproc_memory_usage_forecast_MB(
        num_frames,
        swath_duration=mean_swath_duration,
    )
    if preproc_forecast_load_MB > allocated_memory_MB:
        unsupported_processes["Preprocessor"] = preproc_forecast_load_MB

    # Check if we have enough memory allocated for coregistration.
    coreg_forecast_load_MB = coreg_memory_usage_forecast_MB(
        num_frames,
        swath_duration=mean_swath_duration,
    )
    if coreg_forecast_load_MB > allocated_memory_MB:
        unsupported_processes["Coregistration"] = coreg_forecast_load_MB

    # Check if we have enough memory allocated for the AZF.
    azf_forecast_load_MB = azf_memory_usage_forecast_MB(
        num_frames,
        swath_duration=mean_swath_duration,
        use_32bit_flag=aux_pps.azimuth_spectral_filtering.use_32bit_flag,
    )
    if (
        aux_pps.azimuth_spectral_filtering.azimuth_spectral_filtering_flag
        and azf_forecast_load_MB > allocated_memory_MB
    ):
        unsupported_processes[AZF_NAME] = azf_forecast_load_MB

    # Check if we have enough memory allocated for the SKP.
    skp_forecast_load_MB = skp_memory_usage_forecast_MB(
        num_frames,
        swath_duration=mean_swath_duration,
        use_32bit_flag=aux_pps.skp_phase_calibration.use_32bit_flag,
    )
    if aux_pps.skp_phase_calibration.skp_phase_estimation_flag and skp_forecast_load_MB > allocated_memory_MB:
        unsupported_processes[SKP_NAME] = skp_forecast_load_MB

    # Check if we have enough memory allocated for exporting the products.
    l1c_forecast_load_MB = l1c_memory_usage_forecast_MB(
        num_frames,
        swath_duration=mean_swath_duration,
    )
    if l1c_forecast_load_MB > allocated_memory_MB:
        unsupported_processes["L1c-export"] = l1c_forecast_load_MB

    # Raise if there are unsupported processes.
    if len(unsupported_processes) > 0:
        raise InsufficientMemoryAllocationError(
            allocated_memory=allocated_memory_MB,
            unsupported_modules=unsupported_processes,
        )


def l1c_memory_usage_forecast_MB(
    num_frames: int,
    *,
    swath_duration: float,
    safety_overhead_MB: float = _STANDARD_SAFETY_OVERHEAD_MB,
) -> float:
    """
    The expected usage during L1c export. This was obtained via polynomial
    regression over 29s swath frames of the memory load on a 59GB RAM
    machine with an Intel Xeon(R) CPU E5-1650 v3 @ 3.50GHz (6 core,
    12 threads), under default STA_P configuration and 7 worker threads.

    Parameters
    ----------
    num_frames: int
        Number of stack input frames.

    swath_duration: float [s]
         Duration of the coreg reference frame (approximate).

    safety_overhead_MB: float [MB]
        An extra margin to be safe.

    Raises
    ------
    ValueError

    Return
    ------
    float [MB]
        Number of expected RAM usage in Megabytes.

    """
    if num_frames < 2:
        raise ValueError("Invalid number of input frames")
    if safety_overhead_MB < 0.0:
        raise ValueError("Invalid safety value")

    forecast_MB = 2719.6429 * num_frames + 5381.7857
    return safety_overhead_MB + forecast_MB * swath_duration / CRITICAL_SWATH_DURATION


def skp_memory_usage_forecast_MB(
    num_frames: int,
    *,
    swath_duration: float,
    use_32bit_flag: bool,
    safety_overhead_MB: float = _STANDARD_SAFETY_OVERHEAD_MB,
) -> float:
    """
    The expected usage of the SKP. This was obtained via polynomial
    regression over 29s swath frames of the memory load on a 59GB RAM
    machine with an Intel Xeon(R) CPU E5-1650 v3 @ 3.50GHz (6 core,
    12 threads), under default STA_P configuration and 7 worker threads.

    Parameters
    ----------
    num_frames: int
        Number of stack input frames.

    swath_duration: float [s]
         Duration of the coreg reference frame (approximate).

    use_32bit_flag: bool
        True if 32 bit precision is used for estimation.

    safety_overhead_MB: float [MB]
        An extra margin to be safe.

    Raises
    ------
    ValueError

    Return
    ------
    float [MB]
        Number of expected RAM usage in Megabytes.

    """
    if num_frames < 2:
        raise ValueError("Invalid number of input frames")
    if safety_overhead_MB < 0.0:
        raise ValueError("Invalid safety value")

    if use_32bit_flag:
        forecast_MB = 4895.0 * num_frames + 6267.8571
    else:
        forecast_MB = 7325.0 * num_frames + 8570.0
    return safety_overhead_MB + forecast_MB * swath_duration / CRITICAL_SWATH_DURATION


def azf_memory_usage_forecast_MB(
    num_frames: int,
    *,
    swath_duration: float,
    use_32bit_flag: bool,
    safety_overhead_MB: float = _STANDARD_SAFETY_OVERHEAD_MB,
) -> float:
    """
    The expected usage of the AZF. This was obtained via polynomial
    regression over 29s swath frames of the memory load on a 59GB RAM
    machine with an Intel Xeon(R) CPU E5-1650 v3 @ 3.50GHz (6 core,
    12 threads), under default STA_P configuration and 7 worker threads.

    Parameters
    ----------
    num_frames: int
        Number of stack input frames.

    swath_duration: float [s]
         Duration of the coreg reference frame (approximate).

    use_32bit_flag: bool
        True if 32 bit precision is used for estimation.

    safety_overhead_MB: float [MB]
        An extra margin to be safe.

    Raises
    ------
    ValueError

    Return
    ------
    float [MB]
        Number of expected RAM usage in Megabytes.

    """
    if num_frames < 2:
        raise ValueError("Invalid number of input frames")
    if safety_overhead_MB < 0.0:
        raise ValueError("Invalid safety value")

    if use_32bit_flag:
        forecast_MB = 2423.5714 * num_frames + 10229.2857
    else:
        forecast_MB = 3192.0 * num_frames + 13468.0
    return safety_overhead_MB + forecast_MB * swath_duration / CRITICAL_SWATH_DURATION


def coreg_memory_usage_forecast_MB(
    num_frames: int,
    *,
    swath_duration: float,
    safety_overhead_MB: float = _STANDARD_SAFETY_OVERHEAD_MB,
) -> float:
    """
    The expected usage of the Coreg processor. This was obtained via polynomial
    regression over 29s swath frames of the memory load on a 59GB RAM
    machine with an Intel Xeon(R) CPU E5-1650 v3 @ 3.50GHz (6 core,
    12 threads), under default STA_P configuration and 7 worker threads.

    Parameters
    ----------
    num_frames: int
        Number of stack input frames.

    swath_duration: float [s]
         Duration of the coreg reference frame (approximate).

    safety_overhead_MB: float [MB]
        An extra margin to be safe.

    Raises
    ------
    ValueError

    Return
    ------
    float [MB]
        Number of expected RAM usage in Megabytes.

    """
    if num_frames < 2:
        raise ValueError("Invalid number of input frames")
    if safety_overhead_MB < 0.0:
        raise ValueError("Invalid safety value")

    forecast_MB = 437.5 * num_frames + 7923.9286
    return safety_overhead_MB + forecast_MB * swath_duration / CRITICAL_SWATH_DURATION


def preproc_memory_usage_forecast_MB(
    num_frames: int,
    *,
    swath_duration: float,
    safety_overhead_MB: float = _STANDARD_SAFETY_OVERHEAD_MB,
) -> float:
    """
    The expected usage of the Preprocessor. This was obtained via polynomial
    regression over 29s swath frames of the memory load on a 59GB RAM
    machine with an Intel Xeon(R) CPU E5-1650 v3 @ 3.50GHz (6 core,
    12 threads), under default STA_P configuration and 7 worker threads.

    Parameters
    ----------
    num_frames: int
        Number of stack input frames.

    swath_duration: float [s]
         Duration of the coreg reference frame (approximate).

    safety_overhead_MB: float [MB]
        An extra margin to be safe.

    Raises
    ------
    ValueError

    Return
    ------
    float [MB]
        Number of expected RAM usage in Megabytes.

    """
    if num_frames < 2:
        raise ValueError("Invalid number of input frames")
    if safety_overhead_MB < 0.0:
        raise ValueError("Invalid safety value")

    forecast_MB = 2599.2857 * num_frames + 2747.8571
    return safety_overhead_MB + forecast_MB * swath_duration / CRITICAL_SWATH_DURATION


def _get_mean_swath_duration(l1a_product_paths: list[Path]) -> float:
    """Get mean swath time in seconds."""
    if len(l1a_product_paths) == 0:
        return 0.0

    swath_durations = []
    for l1a_product_path in l1a_product_paths:
        try:
            parsed_l1a_product_name = parse_l1product_name(l1a_product_path.name)
            swath_durations.append(parsed_l1a_product_name.utc_stop_time - parsed_l1a_product_name.utc_start_time)
        except InvalidBIOMASSProductName:
            swath_durations.append(NOMINAL_SWATH_DURATION)
    return np.mean(swath_durations)
