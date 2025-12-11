# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l., DLR, Deimos Space
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import filters, shift


def Regular2DInterp(data, naInt, nrInt):
    Na = len(data[:, 0])
    Nr = len(data[0, :])
    fint = RegularGridInterpolator((np.arange(Na), np.arange(Nr)), data, bounds_error=False, fill_value=None)
    return fint((naInt, nrInt))


def smooth(array, window_len, phase=False, allow_even=True):
    """Smooth function.

    Args:
        data (ndarray any): 2D array to be filtered
        window_len (int or list): Window for filtering. List or as a single number, in which case the size is equal for all axes.
        phase (bool, optional): Smooths interferometric phases with the phase=True keyword. Defaults to False.
        allow_even (bool, optional): Allows even window size. Results are shifted afterwards by 0.5 pixel, requiring an extra interpolation.

    Returns:
        result (ndarray any): Filtered 2D array

    Note:
        `scipy.ndimage` has no NaNn handling. Moroever, the sliding window implementation
        might currupt data outside the window. To avoid that, we perform a `np.nan_to_num`.
        Note that filtered pixels across areas originally invalid will be biased.
    """
    dtype = array.dtype
    if (
        dtype == "int"
        or dtype == "int8"
        or dtype == "int16"
        or dtype == "uint"
        or dtype == "uint8"
        or dtype == "uint16"
    ):
        invalid = 255
        mask = array == invalid
    else:
        invalid = np.nan
        mask = ~np.isfinite(array)

    if not allow_even:
        if isinstance(window_len, (list, tuple, np.ndarray)):
            window_len = [int(window_len[0] // 2) * 2 + 1, int(window_len[1] // 2) * 2 + 1]
        else:
            window_len = int(window_len // 2) * 2 + 1

    array = np.nan_to_num(array)
    if np.iscomplexobj(array):
        array = filters.uniform_filter(array.real, window_len, mode="constant") + 1j * filters.uniform_filter(
            array.imag, window_len, mode="constant"
        )
    elif phase is True:
        array = np.angle(smooth(np.exp(1j * array), window_len))
    else:
        array = filters.uniform_filter(array.real, window_len, mode="constant")

    # the following interpolation will also further blur the data.
    if isinstance(window_len, (list, tuple, np.ndarray)):
        if window_len[0] % 2 == 0 or window_len[1] % 2 == 0:
            # print('Shifting after smoothing')
            array = shift(array, (-((window_len[0] - 1) % 2) / 2, -((window_len[1] - 1) % 2) / 2))
    else:
        if window_len % 2 == 0:
            # print('Shifting after smoothing')
            array = shift(array, -((window_len - 1) % 2) / 2)

    array[mask] = invalid

    return array
