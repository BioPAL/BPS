# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

import numpy as np
import scipy as sp
from bps.stack_cal_processor.core.signal_processing import (
    prepare_spectral_window_removal,
)


def get_spectral_window(
    window_par,
    window_band,
    data_length,
    sampling_step,
):
    """The reference spectral window function."""
    freq_axis = np.fft.fftfreq(data_length, sampling_step)
    max_freq = 1 / sampling_step * window_band / 2
    freq_axis_window = freq_axis[np.abs(freq_axis) < max_freq]
    n_axis = freq_axis_window / np.diff(freq_axis_window[:2])

    spectral_window = window_par - (1 - window_par) * np.cos(2 * np.pi * n_axis / len(n_axis))
    spectral_window = spectral_window / np.sqrt(np.mean(np.abs(spectral_window) ** 2))

    return max_freq, sp.fft.fftshift(spectral_window)


class SignalProcessingTest(unittest.TestCase):
    """Minimal tests on some signal-processing core routines."""

    def test_prepare_spectral_window_removal(self):
        np.random.seed(seed=1)

        compression_window_parameter = np.random.rand()
        compression_window_band = np.random.rand()
        sampling_step = 1e-6
        num_samples = 100

        (
            _,
            dut_cutoff_sampling_frequency,
            dut_centered_sampling_frequencies,
        ) = prepare_spectral_window_removal(
            compression_window_parameter=compression_window_parameter,
            compression_window_band=compression_window_band,
            sampling_step=sampling_step,
            num_samples=num_samples,
        )

        (
            gt_cutoff_sampling_frequency,
            gt_centered_sampling_frequencies,
        ) = get_spectral_window(
            compression_window_parameter,
            compression_window_band,
            num_samples,
            sampling_step,
        )

        np.testing.assert_almost_equal(
            dut_cutoff_sampling_frequency,
            gt_cutoff_sampling_frequency,
        )
        np.testing.assert_almost_equal(
            dut_centered_sampling_frequencies,
            gt_centered_sampling_frequencies,
        )


if __name__ == "__main__":
    unittest.main()
