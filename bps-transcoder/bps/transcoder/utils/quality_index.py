# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to elaborate the overall quality index for L1a/b/c products
---------------------------------------------------------------------
"""

# Overall quality indices are (unsigned) integers associated to L1 products
# that aim to summarize high-level information on the execution of a processor
# into a single integer value. The encoding uses the binary representation of
# the integer value as a list of Boolean flags associated to a specific
# execution steps and/or submodules, with 0 representing a successeful step or
# condition, and 1 a failure or an unmet condition.
#
# Example.
#
#  +---------+
#  |  bit{1} |  <-- Pass/fail condition 1 for sub-module 1.
#  +---------+
#  |  bit{2} |  <-- Pass/fail condition 2 for sub-module 1.
#  +---------+
#  |         |
#  |   ...   |
#  |         |
#  +---------+
#  |  bit{k} |  <-- Pass/fail condition N for sub-module M.
#  +---------+
#  |         |
#  |   ...   |
#  |         |
#  +---------+
#  | bit{32} |  <-- Pass/fail condition 32 for sub-module K.
#  +---------+
#
# Finally, the bitset is converted to an integer using little-endian
# convention, that is:
#
#   Q := \sum_{k=1,...,32} bit{k} x 2^{k-1}.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar

import numpy as np

# Quality indices are encoded as unsigned int as of now.
QUALITY_BITSET_SIZE = 32


class InvalidOverallProductQualityIndexValue(ValueError):
    """Raised when an invalid quality index value is used."""


class InvalidOverallProductQualityBitset(ValueError):
    """Raised when an invalid quality index value is used."""


class QualityBitSetEndianness(Enum):
    """The endiannes of the overall quality index."""

    BIG_ENDIAN = 0
    LITTLE_ENDIAN = 1

    @staticmethod
    def endian_fn(endianness: QualityBitSetEndianness) -> Callable:
        """The conversion function."""
        if endianness is QualityBitSetEndianness.BIG_ENDIAN:
            return lambda bitset: bitset
        elif endianness is QualityBitSetEndianness.LITTLE_ENDIAN:
            return lambda bitset: np.flip(bitset)
        raise ValueError(f"Invalid endianness {endianness}")


def quality_index_to_bitset(
    quality_index: np.uint32,
    *,
    endianness: QualityBitSetEndianness = QualityBitSetEndianness.LITTLE_ENDIAN,
) -> list[np.uint8]:
    """
    Convert a quality index to a bitset.

    Parameters
    ----------
    quality_index: np.uint32
        The input overall quality index value, which is an "unsignedInt".

    endianness: QualityBitSetEndianness
        The endiannes of the overall quality index. It defaults to LITTLE_ENDIAN.

    Raises
    ------
    InvalidOverallProductQualityIndexValue

    Return
    ------
    list[np.uint8]
        The overall quality index bitset.

    """
    if quality_index < 0 or quality_index >= 2**QUALITY_BITSET_SIZE:
        raise InvalidOverallProductQualityIndexValue(
            f"Quality index must be an integer in [0, 2**{QUALITY_BITSET_SIZE}-1]"
        )

    endian_fn = QualityBitSetEndianness.endian_fn(endianness)
    return endian_fn([np.uint8(b) for b in np.binary_repr(quality_index, width=QUALITY_BITSET_SIZE)])


def quality_bitset_to_index(
    quality_bitset: list[np.uint8],
    *,
    endianness: QualityBitSetEndianness = QualityBitSetEndianness.LITTLE_ENDIAN,
) -> np.uint32:
    """
    Convert a quality bitset to an index.

    Parameters
    ----------
    quality_bitset: list[np.uint8]
        The input quality bit-set. This must be a list of 32 (boolean) bits.

    endianness: QualityBitSetEndianness
        The endiannes of the overall quality index. It defaults to LITTLE_ENDIAN.

    Raises
    ------
    InvalidOverallProductQualityBitset

    Return
    ------
    np.uint32
        The overall quality index value.

    """
    if len(quality_bitset) != QUALITY_BITSET_SIZE:
        raise InvalidOverallProductQualityBitset(f"Quality bitsets must have size {QUALITY_BITSET_SIZE}")

    # 0-padding handled by int(). '0b' tells int() to use binary representation.
    endian_fn = QualityBitSetEndianness.endian_fn(endianness)
    quality_index = int(
        "0b{:s}".format("".join(str(b) for b in endian_fn(np.array(quality_bitset, dtype=np.uint8)))),
        2,
    )
    assert 0 <= quality_index < 2**QUALITY_BITSET_SIZE, "Quality index out of bounds"

    return np.uint32(quality_index)


@dataclass
class QualityIndex:
    """Generic class for a quality index

    Example
    -------

    @dataclass
    class AQualityIndex(QualityIndex):
        a: bool # <-- up to 32 boolean properties
        b: bool
        c: bool

        _bit_map = {
            "a": 1, # <-- for each properties the corresponding bit position (0...31)
            "b": 5,
            "c": 0,
        }
    """

    _bit_map: ClassVar[dict[str, int]] = {}

    def __post_init__(self):
        """Validate the bit-map."""
        # Check that the bit-map did not miss some field.
        assert sorted(f.name for f in fields(self)) == sorted(self._bit_map), (
            "not all fields of QualityIndex are mapped to bit positions"
        )

        # Check that the bit position are not duplicated.
        assert len(set(self._bit_map.values())) == len(self._bit_map.values()), (
            "QualityIndex contains duplicated bit locations"
        )

    @classmethod
    def from_bitset(cls, bitset: list[np.uint8]) -> QualityIndex:
        """Initialize the object from a bitset."""
        assert len(bitset) == QUALITY_BITSET_SIZE

        quality_dict = {}
        for class_field in fields(cls):
            bit_position = cls._bit_map.get(class_field.name)
            assert bit_position is not None
            quality_dict[class_field.name] = bool(bitset[bit_position])

        return cls(**quality_dict)

    @classmethod
    def decode(cls, index: np.uint8):
        """Decode quality index or bitset"""
        bitset = quality_index_to_bitset(index)
        return cls.from_bitset(bitset)

    def encode(self) -> int:
        """Encode quality index"""
        bitset = [np.uint8(0)] * QUALITY_BITSET_SIZE

        for class_field in fields(self):
            bit_position = self._bit_map.get(class_field.name)
            assert bit_position is not None
            bitset[bit_position] = np.uint8(getattr(self, class_field.name))

        return int(quality_bitset_to_index(bitset))
