# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS utils
----------
"""

import logging
from enum import Enum
from itertools import product
from typing import Any
from warnings import catch_warnings, simplefilter

import numpy as np
import numpy.typing as npt
from arepytools.io.metadata import EPolarization, MetaDataElement, SwathInfo
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common


class EarthModel(Enum):
    """Earth models"""

    WGS84 = "WGS84"
    GETASSE = "GETASSE"
    SRTM = "SRTM"
    COPERNICUS = "COPERNICUS"


class ProductFormat(Enum):
    """Aresys products format"""

    BIN = "BIN+XML"
    TIFF = "TIFF+XML"


class LogLevel(Enum):
    """Logging levels"""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    DEBUG = "DEBUG"


def floating_point_error_handler(
    loglevel: int,
    fn_name: str,
    error_type: str,
    *args,
):
    """
    Floating point warning callback function. This is used by
    numpy to report floating point errors such as floating point
    under- or overflows.

    Example
    -------
    Possible usage within the stack (see for instance
    https://numpy.org/devdocs/reference/generated/numpy.seterr.html)

        np.seterr(all="call")
        np.seterrcall(
            functools.partial(floating_point_error_handler, loglevel, fn_name)
        )

    or

        np.seterr(all="call")
        np.seterrcall(
            lambda t, _: floating_point_error_handler(loglevel, fn_name, t, _)
        )

    Parameters
    ----------
    loglevel: int
        The logger level (i.e. DEBUG, INFO, etc.)

    fn_name: str
        The function throwing the warning.

    error_type: str
        The error type (e.g. divide by zero).

    Raises
    ------
    ValueError

    """
    if loglevel == logging.DEBUG:
        logger_fn = bps_logger.debug
    elif loglevel == logging.INFO:
        logger_fn = bps_logger.info
    elif loglevel == logging.WARNING:
        logger_fn = bps_logger.warning
    elif loglevel == logging.ERROR:
        logger_fn = bps_logger.error
    elif loglevel == logging.CRITICAL:
        logger_fn = bps_logger.critical
    else:
        raise ValueError(f"Unsupported logging level '{loglevel}'")

    logger_fn(
        "Floating point error %s ecountered in %s",
        error_type,
        fn_name,
    )


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
    Perform the cross-pol merging (Used in Stack PreProcessor or L2A Processor).

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
    RuntimeError

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
            raise RuntimeError("HV data not available but HV selected for polarization combination")
        if any(f"{lut}HV" not in lut_data for lut in xpol_luts):
            raise RuntimeError("HV LUT data not available but HV selected for polarization combination")

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
            raise RuntimeError("V/H not available but selected for polarization combination")
        if any(f"{lut}VH" not in lut_data for lut in xpol_luts):
            raise RuntimeError("VH LUT data not available but VH selected for polarization combination")

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
            raise RuntimeError("V/H and H/V are both required when merging cross-polarizations")
        if any(f"{lut}{p}" not in lut_data for lut, p in product(xpol_luts, ("HV", "VH"))):
            raise RuntimeError("V/H and H/V LUTs are both required when merging cross-polarizations")

        # The cross-pol indices.
        hv_index = polarization_list.index(EPolarization.hv.value)
        vh_index = polarization_list.index(EPolarization.vh.value)

        # Substitute the H/V index with X/X.
        polarization_list[hv_index] = EPolarization.xx.value
        with catch_warnings():
            simplefilter("ignore")
            if len(data_list) > 0:
                if len(data_list) < max(hv_index, vh_index):
                    raise RuntimeError("Data stack ill-formed. Cannot access H/V and/or V/H polarization")
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


def _invalid_data(data: npt.NDArray[complex]) -> npt.NDArray[bool]:
    """Mask of possibly problematic data."""
    return np.isnan(data) | (np.abs(data) == np.inf)


def _try_pop(data: list[npt.NDArray[complex]], *, index: int, default_value: Any = None):
    """Pop from list if possible, otherwise return default value."""
    try:
        return data.pop(index)
    except IndexError:
        return default_value
