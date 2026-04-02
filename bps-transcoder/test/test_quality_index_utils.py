# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import dataclass

import numpy as np
from bps.transcoder.utils.quality_index import (
    QUALITY_BITSET_SIZE,
    QualityBitSetEndianness,
    QualityIndex,
    quality_bitset_to_index,
    quality_index_to_bitset,
)

# Just a shortcut.
BIG = QualityBitSetEndianness.BIG_ENDIAN
LITTLE = QualityBitSetEndianness.LITTLE_ENDIAN


def _to_bool(bitset: list[int]) -> list[bool]:
    return np.array(bitset, dtype=bool).tolist()


class QualityIndexUtilsTests(unittest.TestCase):
    """Helper class to test the quality index utility library."""

    def test_quality_index_to_bitset_manual_bigendian(self):
        """Test a bunch of values."""
        bitset_5 = _to_bool([0 for _ in range(29)] + [1, 0, 1])
        self.assertListEqual(
            quality_index_to_bitset(5, endianness=BIG),
            bitset_5,
        )

        bitset_7 = _to_bool([0 for _ in range(28)] + [1, 0, 1, 0])
        self.assertListEqual(
            quality_index_to_bitset(10, endianness=BIG),
            bitset_7,
        )

        bitset_1945 = _to_bool([0 for _ in range(21)] + [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.assertListEqual(
            quality_index_to_bitset(1945, endianness=BIG),
            bitset_1945,
        )

    def test_quality_bitset_to_index_manual_bigendian(self):
        """Test a bunch of bitsets."""
        bitset_0 = np.zeros(QUALITY_BITSET_SIZE, dtype=np.uint8)
        self.assertEqual(
            quality_bitset_to_index(bitset_0, endianness=BIG),
            0,
        )

        bitset_3 = [0 for _ in range(30)] + [1, 1]
        self.assertEqual(
            quality_bitset_to_index(bitset_3, endianness=BIG),
            3,
        )

        bitset_17 = [0 for _ in range(27)] + [1, 0, 0, 0, 1]
        self.assertEqual(
            quality_bitset_to_index(bitset_17, endianness=BIG),
            17,
        )

        bitset_1861 = [0 for _ in range(21)] + [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1]
        self.assertEqual(
            quality_bitset_to_index(bitset_1861, endianness=BIG),
            1861,
        )

    def test_conversion_littleendian_special_case(self):
        bitset_269484032 = [0 for _ in range(32)]
        bitset_269484032[20] = 1
        bitset_269484032[28] = 1
        self.assertEqual(
            quality_bitset_to_index(bitset_269484032, endianness=LITTLE),
            269484032,
        )

    def test_roundtrip_conversion_littleendian(self):
        np.random.seed(seed=1)
        for gt in map(int, np.floor(np.random.rand(100) * 2**QUALITY_BITSET_SIZE)):
            dut = quality_bitset_to_index(quality_index_to_bitset(gt))
            self.assertEqual(gt, dut)


class QualityIndexTestCase(unittest.TestCase):
    """Test of generic class"""

    @dataclass
    class AQualityIndex(QualityIndex):
        """Example of concrete class"""

        a: bool
        b: bool
        c: bool
        _bit_map = {
            "a": 1,
            "b": 5,
            "c": 0,
        }

    @dataclass
    class ABogusQualityIndex(QualityIndex):
        """An example of a bogus quality index."""

        a: bool
        b: bool
        c: bool
        _bit_map = {
            "a": 1,
            "b": 5,
        }

    def setUp(self):
        """Check all values"""
        self.test_values: list[tuple[QualityIndexTestCase.AQualityIndex, int]] = [
            (self.AQualityIndex(c=False, a=False, b=False), 0),
            (self.AQualityIndex(c=True, a=False, b=False), 2**0),
            (self.AQualityIndex(c=False, a=True, b=False), 2**1),
            (self.AQualityIndex(c=True, a=True, b=False), 2**0 + 2**1),
            (self.AQualityIndex(c=False, a=False, b=True), 2**5),
            (self.AQualityIndex(c=True, a=False, b=True), 2**0 + 2**5),
            (self.AQualityIndex(c=False, a=True, b=True), 2**1 + 2**5),
            (self.AQualityIndex(c=True, a=True, b=True), 2**0 + 2**1 + 2**5),
        ]

    def test_encode(self):
        """Encode index"""
        for obj, index in self.test_values:
            self.assertEqual(obj.encode(), index)

    def test_decode(self):
        """Decode index"""
        for obj, index in self.test_values:
            self.assertEqual(self.AQualityIndex.decode(np.uint32(index)), obj)

    def test_bogus_initialization(self):
        """Expect failing init assert."""
        with self.assertRaises(AssertionError):
            self.ABogusQualityIndex(a=True, b=False, c=True)


if __name__ == "__main__":
    unittest.main()
