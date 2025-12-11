# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Production Model utils
"""

# NOTE: Product names and MPH have slightly different conventions, e.g., a "Not
# Applicable" value will be translated to "NA" in MPH and "__" or "___" in
# product names.


def decode_mph_id_value(string: str) -> int:
    """
    Convert a production model ID string as encoded in the MPH files
    to an optional integer for internal use only.

    The production model will follow the following convention:
    - global coverage ID: "1", "2", ..., "6", "NA"
    - major cycle ID: "1", "2", ..., "7", "NA"
    - repeat cycle ID: "1", "2", ..., "8", "NA", "DR"

    This function handles 0-padded integers.

    Parameters
    ----------
    string: str
        The string ID.

    Raises
    ------
    ValueError when the input ID is invalid.

    Return
    ------
    int
        The value for internal usage..

    """
    if string.isdigit():  # We stay general here.
        return int(string)
    if string == "NA":
        return -1
    if string == "DR":
        return -2
    raise ValueError(f"Value '{string}' is not supported by the production model")


def encode_mph_id_value(value: int) -> str:
    """
    Convert the production model ID from the value used internally
    in the code to the value needed by MPH.

    The production model will follow the following convention:
    - global coverage ID: 1, 2, ..., 6, -1 (=NA)
    - major cycle ID: 1, 2, ..., 7, -1 (=NA)
    - repeat cycle ID: 1, 2, ..., 8, -1 (=NA), -2 (=DR).

    None is converted to "__"

    Parameters
    ----------
    value: int
        The encoded value.

    Raises
    ------
    ValueError if the input encoded value is invalid.

    Return
    ------
    str
        The string ID. Integers are not 0-padded.

    """
    if value >= 0:  # We stay general here.
        return str(value)
    if value == -1:
        return "NA"
    if value == -2:
        return "DR"
    raise ValueError(f"Value '{value}' is not supported by the production model")


def decode_product_name_id_value(string: str) -> int:
    """
    Convert a production model ID string as encoded in the product
    name to an integer for internal use only.

    The production model will follow the following convention:
    - global coverage ID: "01", "02", ..., "06", "__"
    - major cycle ID: "01", "02", ..., "07", "__"
    - repeat cycle ID: "01", "02", ..., "08", "__", "DR"

    Parameters
    ----------
    string: str
        The string ID.

    Raises
    ------
    ValueError when the input ID is invalid.

    Return
    ------
    int
        The value for internal usage..

    """
    if string.isdigit():
        return int(string)
    if all(c == "_" for c in string):
        return -1
    if string == "DR":
        return -2
    raise ValueError(f"Value '{string}' is not supported by the production model")


def encode_product_name_id_value(value: int, *, npad: int) -> str:
    """
    Convert the production model ID from the value used internally
    in the code to the value used in product names.

    The production model will follow the following convention:
    - global coverage ID: 1, 2, ..., 6, -1 (=__)
    - major cycle ID: 1, 2, ..., 7, -1 (=__)
    - repeat cycle ID: 1, 2, ..., 8, -1 (=__), -2 (=DR).

    Parameters
    ----------
    value: int
        The encoded value.

    Raises
    ------
    ValueError if the input encoded value is invalid.

    Return
    ------
    str
        The string ID. Integers are not 0-padded.

    """
    if value >= 0:
        return f"{value}".rjust(npad, "0")
    if value == -1:
        return "".rjust(npad, "_")
    if value == -2:
        return "DR"
    raise ValueError(f"Value '{value}' is not supported by the production model")


# IDs annotation translation functions
def translate_global_coverage_id(
    identifier: int,
) -> int:
    """Translate global coverage ID"""
    if identifier in range(1, 7):
        return identifier
    return 0


def translate_major_cycle_id(
    identifier: int,
) -> int:
    """Translate major cycle ID"""
    if identifier in range(1, 8):
        return identifier
    return 0


def translate_repeat_cycle_id(
    identifier: int,
) -> int:
    """Translate repeat cycle ID"""
    if identifier in range(1, 9):
        return identifier
    return 0


def translate_frame_id(
    identifier: int,
) -> int:
    """Translate frame ID"""
    if identifier in range(1, 7):
        return identifier
    return 0


def translate_com_phase_negative_values(
    identifier: int,
) -> int:
    """Generic function to translate negative COMMISSIONING PHASE values,
       before writing in NetCdf or main annotation (which supports only UINT types):

        -1, coming from "__" in TDS folder name, becomes 0
        -2, coming from "DR" in TDS folder name, becomes 0

    See also:
        encode_product_name_id_value, decode_product_name_id_value
    """
    return max(0, identifier)
