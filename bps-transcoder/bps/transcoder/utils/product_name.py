# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BIOMASS L1 Product Naming Convention
------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.utils.production_model_utils import decode_product_name_id_value

# Regular expression that matches a BIOMASS L1 product name.
BPS_L1_PRODUCT_REGEX = (
    "^BIO_S[1-3]"
    + "_[A-Z]+"
    + "[_12XY]_1[MS]"
    + "_[0-9]{8}T[0-9]{6}"
    + "_[0-9]{8}T[0-9]{6}"
    + "_[A-Z]"
    + "_G([0-9]{2}|_{2})"
    + "_M([0-9]{2}|_{2})"
    + "_C([0-9]{2}|(DR)|_{2})"
    + "_T([0-9]{3}|_{3})"
    + "_F([0-9]{3}|_{3})"
    + "_[0-9]{2}"
    + "_[0-9A-Z]{6}$"
)

# Regular expression that matches a BIOMASS L2A product name.
BPS_L2A_PRODUCT_REGEX = (
    "^BIO_FP"
    + "_[A-Z]{2}"
    + "[_A-Z]{1}_L2A"
    + "_[0-9]{8}T[0-9]{6}"
    + "_[0-9]{8}T[0-9]{6}"
    + "_[A-Z]"
    + "_G([0-9]{2}|_{2})"
    + "_M([0-9]{2}|_{2})"
    + "_C([0-9]{2}|(DR)|_{2})"
    + "_T([0-9]{3}|_{3})"
    + "_F([0-9]{3}|_{3})"
    + "_[0-9]{2}"
    + "_[0-9A-Z]{6}$"
)


# Regular expression that matches a BIOMASS L2B product name.
BPS_L2B_PRODUCT_REGEX = (
    "^BIO_FP"
    + "_[A-Z]{2}"
    + "[B_]"
    + "_L2B"
    + "_[A-Z]"
    + "_G([0-9]{2}|_{2})"
    + "_T[SN]{1}[0-9]{2}"
    + "[EW]{1}[0-9]{3}"
    + "_B[0-9]{3}"
    + "_[0-9]{2}"
    + "_[0-9A-Z]{6}$"
)


class InvalidBIOMASSProductName(ValueError):
    """Handle invalid BIOMASS product names."""


@dataclass(frozen=True)
class ParsedBIOMASSL1ProductName:
    """
    Store the parsed BIOMASS L1 product name.

    Note
    ----
    global_coverage_id, major_cycle_id, repeat_cycle_id, track_number, and
    frame_number are stored as integer in internal encoding in order to
    account for NA and DR.

    See Also
    --------
    bps.transcoder.production_model_utils

    """

    satellite_id: str
    stripmap_id: str
    processor_id: str
    is_monitoring: bool
    utc_start_time: PreciseDateTime | str
    utc_stop_time: PreciseDateTime | str
    mission_id: str
    coverage: int
    major_cycle: int
    repeat_cycle: int
    track_number: int
    frame_number: int
    baseline: int
    creation_stamp: str | None = None

    @property
    def product_type(self) -> str:
        """Return the product type, eg. S1_SCS__1S."""
        return f"{self.stripmap_id}_{self.processor_id.ljust(5, '_')}{'1M' if self.is_monitoring else '1S'}"


@dataclass(frozen=True)
class ParsedBIOMASSL2AProductName:
    """
    Store the parsed BIOMASS L2A product name.

    Note
    ----
    global_coverage_id, major_cycle_id, repeat_cycle_id, track_number, and
    frame_number are stored as integer in internal encoding in order to
    account for NA and DR.

    See Also
    --------
    bps.transcoder.production_model_utils

    """

    satellite_id: str | None = None
    product_type: str | None = None
    utc_start_time: PreciseDateTime | str | None = None
    utc_stop_time: PreciseDateTime | str | None = None
    mission_id: str | None = None
    coverage: int | None = None
    major_cycle: int | None = None
    repeat_cycle: int | None = None
    track_number: int | None = None
    frame_number: int | None = None
    baseline: int | None = None
    creation_stamp: str | None = None


@dataclass(frozen=True)
class ParsedBIOMASSL2BProductName:
    """
    Store the parsed BIOMASS L2B product name.

    Note
    ----
    global_coverage_id is stored as integer in internal encoding in order to
    account for NA and DR.

    See Also
    --------
    bps.transcoder.production_model_utils

    """

    satellite_id: str | None = None
    product_type: str | None = None
    mission_id: str | None = None
    coverage: int | None = None
    tile_id: str | None = None
    basin_id: int | None = None
    baseline: int | None = None
    creation_stamp: str | None = None


def parse_l1product_name(
    product_name: str,
    *,
    time_format: Literal["str", "datetime"] = "datetime",
    store_creation_stamp: bool = True,
) -> ParsedBIOMASSL1ProductName:
    """
    Parse a BIOMASS L1 product name and extract core information.

    Parameters
    ----------
    product_name: str
        A BIOMASS L1 product name.

    time_format: Literal["str", "datetime"] = "datetime"
        How to encode the time format: string or PreciseDateTime. If
        set to "str" the class is hashable.

    store_creation_stamp: bool = False
        If the creation stamp suffix (eg. _CYA64H) is to stored or not.

    Raises
    ------
    InvalidBIOMASSProductName, InvalidUtcString, TypeError

    Return
    ------
    ParsedBIOMASSL1ProductName
        The parsed product name.

    """
    if time_format not in {"str", "datetime"}:
        raise ValueError("Only 'str' and 'datetime' are supported")

    if not re.match(BPS_L1_PRODUCT_REGEX, product_name):
        raise InvalidBIOMASSProductName(f"'{product_name}' is not a valid name for a BIOMASS product")

    satellite_id = product_name[0:3]
    product_type = product_name[4:14]  # S1_SCS__1S
    utc_start_time_str = product_name[15:30]
    utc_stop_time_str = product_name[31:46]
    mission_id = product_name[47]
    coverage_str = product_name[50:52]
    major_cycle_str = product_name[54:56]
    repeat_cycle_str = product_name[58:60]
    track_number_str = product_name[62:65]
    frame_number_str = product_name[67:70]
    baseline_str = product_name[71:73]
    creation_stamp = product_name[74:80]

    stripmap_id, remainder = product_type.split("_", maxsplit=1)
    processor_id, product_level = remainder.replace("__", "_").split("_", maxsplit=1)

    if time_format == "datetime":
        utc_start_time_str = PreciseDateTime.from_utc_string(f"{utc_start_time_str}.0")
        utc_stop_time_str = PreciseDateTime.from_utc_string(f"{utc_stop_time_str}.0")

    if not store_creation_stamp:
        creation_stamp = None

    return ParsedBIOMASSL1ProductName(
        satellite_id=satellite_id,
        stripmap_id=stripmap_id,
        processor_id=processor_id,
        is_monitoring=product_level.endswith("M"),
        utc_start_time=utc_start_time_str,
        utc_stop_time=utc_stop_time_str,
        mission_id=mission_id,
        coverage=decode_product_name_id_value(coverage_str),
        major_cycle=decode_product_name_id_value(major_cycle_str),
        repeat_cycle=decode_product_name_id_value(repeat_cycle_str),
        track_number=decode_product_name_id_value(track_number_str),
        frame_number=decode_product_name_id_value(frame_number_str),
        baseline=int(baseline_str),
        creation_stamp=creation_stamp,
    )


def parse_l2aproduct_name(
    product_name: str,
    *,
    time_format: Literal["str", "datetime"] = "datetime",
    store_creation_stamp: bool = True,
) -> ParsedBIOMASSL2AProductName:
    """
    Parse a BIOMASS L2A product name and extract core information.

    Parameters
    ----------
    product_name: str
        A BIOMASS L2A product name.

    time_format: Literal["str", "datetime"] = "datetime"
        How to encode the time format: string or PreciseDateTime. If
        set to "str" the class is hashable.

    store_creation_stamp: bool = False
        If the creation stamp suffix (eg. _CYA64H) is to stored or not.

    Raises
    ------
    InvalidBIOMASSProductName, InvalidUtcString, TypeError

    Return
    ------
    ParsedBIOMASSL2AProductName
        The parsed product name.

    """
    if time_format not in {"str", "datetime"}:
        raise ValueError("Only 'str' and 'datetime' are supported")

    if not re.match(BPS_L2A_PRODUCT_REGEX, product_name):
        raise InvalidBIOMASSProductName(f"'{product_name}' is not a valid name for a BIOMASS product")

    satellite_id = product_name[0:3]
    product_type = product_name[4:14]  # FP_XX__L2A
    utc_start_time_str = product_name[15:30]
    utc_stop_time_str = product_name[31:46]
    mission_id = product_name[47]
    coverage_str = product_name[50:52]
    major_cycle_str = product_name[54:56]
    repeat_cycle_str = product_name[58:60]
    track_number_str = product_name[62:65]
    frame_number_str = product_name[67:70]
    baseline_str = product_name[71:73]
    creation_stamp = product_name[74:80]

    if time_format == "datetime":
        utc_start_time_str = PreciseDateTime.from_utc_string(f"{utc_start_time_str}.0")
        utc_stop_time_str = PreciseDateTime.from_utc_string(f"{utc_stop_time_str}.0")

    if not store_creation_stamp:
        creation_stamp = None

    return ParsedBIOMASSL2AProductName(
        satellite_id=satellite_id,
        product_type=product_type,
        utc_start_time=utc_start_time_str,
        utc_stop_time=utc_stop_time_str,
        mission_id=mission_id,
        coverage=decode_product_name_id_value(coverage_str),
        major_cycle=decode_product_name_id_value(major_cycle_str),
        repeat_cycle=decode_product_name_id_value(repeat_cycle_str),
        track_number=decode_product_name_id_value(track_number_str),
        frame_number=decode_product_name_id_value(frame_number_str),
        baseline=int(baseline_str),
        creation_stamp=creation_stamp,
    )


def parse_l2bproduct_name(
    product_name: str,
    store_creation_stamp: bool = True,
) -> ParsedBIOMASSL2BProductName:
    """
    Parse a BIOMASS L2B product name and extract core information.

    Parameters
    ----------
    product_name: str
        A BIOMASS L2B product name.

    store_creation_stamp: bool = False
        If the creation stamp suffix (eg. _CYA64H) is to stored or not.

    Raises
    ------
    InvalidBIOMASSProductName, InvalidUtcString, TypeError

    Return
    ------
    ParsedBIOMASSL2BProductName
        The parsed product name.

    """
    if not re.match(BPS_L2B_PRODUCT_REGEX, product_name):
        raise InvalidBIOMASSProductName(f"'{product_name}' is not a valid name for a BIOMASS product")

    satellite_id = product_name[0:3]
    product_type = product_name[4:14]  # FP_XXX_L2B
    mission_id = product_name[15]
    coverage_str = product_name[18:20]
    tile_id = product_name[22:29]
    basin__id = int(product_name[31:34])
    baseline_id = int(product_name[35:37])
    creation_stamp = product_name[38:44]

    if not store_creation_stamp:
        creation_stamp = None

    return ParsedBIOMASSL2BProductName(
        satellite_id=satellite_id,
        product_type=product_type,
        mission_id=mission_id,
        coverage=decode_product_name_id_value(coverage_str),
        tile_id=tile_id,
        basin_id=basin__id,
        baseline=baseline_id,
        creation_stamp=creation_stamp,
    )


def is_l1_product_name_valid(product_name: str) -> bool:
    """Test if product name is valid."""
    try:
        parse_l1product_name(product_name)
        return True
    except InvalidBIOMASSProductName:
        return False


def is_l2a_product_name_valid(product_name: str) -> bool:
    """Test if product name is valid."""
    try:
        parse_l2aproduct_name(product_name)
        return True
    except InvalidBIOMASSProductName:
        return False


def is_l2b_product_name_valid(product_name: str) -> bool:
    """Test if product name is valid."""
    try:
        parse_l2bproduct_name(product_name)
        return True
    except InvalidBIOMASSProductName:
        return False
