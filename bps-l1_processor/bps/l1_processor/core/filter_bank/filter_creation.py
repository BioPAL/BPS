# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Filter bank creation utils
--------------------------
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import signal


def get_frequency_axis(axis_size):
    """Get the normalized frequency axis"""
    return ((np.arange(axis_size) / axis_size + 1 / 2) % 1) - 1 / 2


def shift_filter(frequency_axis: np.ndarray, filter_spectrum: np.ndarray, shift: float):
    """Filter shift"""
    shifted_data = filter_spectrum * np.exp(-1j * 2 * np.pi * (frequency_axis * shift))
    return np.real(np.fft.ifft(shifted_data))


def create_bank(input_filter: np.ndarray, bank_size: int):
    """Create a bank of filters by shifting the input filter"""
    filter_spectrum = np.fft.fft(input_filter)
    frequency_axis = get_frequency_axis(len(filter_spectrum))

    bank = np.empty((bank_size, input_filter.size))
    for index, output_filter in enumerate(bank):
        shift = index / bank_size
        output_filter[:] = shift_filter(frequency_axis, filter_spectrum, shift)

    return bank


class FilterBuilder(Protocol):
    """Protocol for objects that can build a filter"""

    def build(self, filter_size: int) -> np.ndarray:  # type: ignore
        """Build filter of size filter_size"""


def create_filter_bank(
    filter_builder: FilterBuilder,
    filter_size: int,
    bank_size: int,
) -> np.ndarray:
    """Build a filter bank by filter shifting"""
    reference_filter = filter_builder.build(filter_size)
    return create_bank(reference_filter, bank_size)


@dataclass
class SincFilterBuilder:
    """Builder of Sinc filters"""

    bandwidth: float

    def build(self, filter_size: int) -> np.ndarray:
        """Build the filter"""
        sinc_filter = np.sinc(self.bandwidth * (np.arange(filter_size) - (filter_size - 1) / 2)) * np.blackman(
            filter_size
        )
        return sinc_filter * np.sqrt(self.bandwidth)


@dataclass
class FirFilterBuilder:
    """Builder of FIR filters"""

    bands: np.ndarray
    gains_at_bands: np.ndarray

    def build(self, filter_size: int) -> np.ndarray:
        """Build the filter"""
        fir_filter = signal.firls(numtaps=filter_size, bands=self.bands, desired=self.gains_at_bands)
        return fir_filter / np.sqrt(self.bands[1])
