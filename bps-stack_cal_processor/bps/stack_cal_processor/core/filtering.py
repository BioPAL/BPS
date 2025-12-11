# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Filtering Utilities
-------------------
"""

# NOTE: This module wraps OpenCV's filter2D, which under some circumstances
# seems to be performing better than scipy and numpy built-in utilities. The
# user should consider on a case-by-case basis when this module is to be
# preferred over scipy since it may depend on many factors including
# convolution parameters, hardware, multithreading architecture etc. It is
# observed though that with small convolution kernels (e.g. ~10 x ~10)
# cv2.filter2D seems to be outperforming scipy.

from __future__ import annotations

import enum
from math import floor

import cv2
import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common.roi_utils import RegionOfInterest


# Just an enumeration for the commonly used window types.
class ConvolutionWindowType(enum.Enum):
    """An enumeration to identify the convolution windows."""

    HAMMING = "HAMMING"
    """The Hamming window."""

    KAISER = "KAISER"
    """The Kaiser window."""

    UNIFORM = "UNIFORM"
    """The uniform (box-car) window."""

    NONE = "NONE"
    """No convolutional window."""


# Just a map for the OpenCV extrapolation types. See OpenCV official
# documentation for more details.
class ConvolutionBorderType(enum.Enum):
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    DEFAULT = cv2.BORDER_DEFAULT
    ISOLATED = cv2.BORDER_ISOLATED


# Same as sp.signal's convention.
class ConvolutionRoiMode(enum.Enum):
    SAME = "SAME"
    VALID = "VALID"


def convolve_2d(
    data: npt.NDArray,
    kernel: npt.NDArray[float],
    *,
    border_type: ConvolutionBorderType = ConvolutionBorderType.CONSTANT,
    roi_mode: ConvolutionRoiMode = ConvolutionRoiMode.SAME,
) -> npt.NDArray:
    """
    Convolve 2D data with a selected kernel.

    This method wraps cv2.filter2D and it is usually preferred over scipy
    utilities when small convlution kernels are employed (e.g. 5 x 5). For
    larger kernels, the advantages (if any) are usually negligible.

    This method does not support complex kernels.

    Parameters
    ----------
    data: npt.NDArray
        The input data (real or complex).

    kernel: npt.NDArray[float]
        The convolution kernel.

    border_type: ConvolutionBorderType
        The extrapolation method. See OpenCV documentation for details.

    roi_mode: ConvolutionRoiMode = ConvolutionRoiMode.SAME
        If the entire image or only the valid part is returned.

    Raises
    ------
    ValueError: When input data and kernel is not 2D
    NotImplementedError: When a complex kernel is provided.

    Return
    ------
    npt.NDArray
        The convolved data.

    """
    if data.ndim != 2:
        raise ValueError(
            "Input data must be [Nrow x Ncol] array (ndim=2)",
        )
    if kernel.ndim != 2:
        raise ValueError(
            "Convolution kernel must be [Nrow x Ncol] array (ndim=2)",
        )
    if np.iscomplexobj(kernel):
        raise NotImplementedError("Complex convolution kernels are not supported yet")

    # NOTE: cv2.filter2D does not support complex data.
    data_out = cv2.filter2D(
        np.real(data),
        ddepth=-1,
        kernel=kernel,
        borderType=border_type.value,
    )

    if np.iscomplexobj(data):
        data_out = data_out + 1j * cv2.filter2D(
            np.imag(data),
            ddepth=-1,
            kernel=kernel,
            borderType=border_type.value,
        )

    if roi_mode is ConvolutionRoiMode.SAME:
        return data_out

    if roi_mode is ConvolutionRoiMode.VALID:
        valid_roi = filter_validity_roi(
            data_shape=data.shape,
            kernel_shape=kernel.shape,
        )
        return data_out[
            valid_roi[0] : valid_roi[0] + valid_roi[2],
            valid_roi[1] : valid_roi[1] + valid_roi[3],
        ]

    raise ValueError(f"{roi_mode} is not a supported for convolution")


def uniform_filter_2d(
    data: npt.NDArray,
    filter_shape: tuple[int, int],
    *,
    border_type: ConvolutionBorderType = ConvolutionBorderType.DEFAULT,
    roi_mode: ConvolutionRoiMode = ConvolutionRoiMode.SAME,
) -> npt.NDArray:
    """
    Apply a uniform filter using cv2.filter2D.

    This method wraps cv2.filter2D and it is usually preferred over scipy
    utilities when filter window is relatively small (e.g. 5 x 5). For
    larger windows, the advantages (if any) are usually negligible.

    Parameters
    ----------
    data: npt.NDArray
        The input data.

    filter_shape: tuple[int, int]
        The filter shape. Both must be integers.

    border_type: ConvolutionBorderType
        The extrapolation method. See OpenCV documentation for details.

    roi_mode: ConvolutionRoiMode = ConvolutionRoiMode.SAME
        If the entire image or only the valid part is returned.

    Raises
    ------
    ValueError: When the filter_shape is invalid.

    Return
    ------
    npt.NDArray
        The filtered data.

    """
    _raise_if_invalid_2d_shape(filter_shape)

    return convolve_2d(
        data,
        kernel=_boxcar_kernel(filter_shape),
        border_type=border_type,
        roi_mode=roi_mode,
    )


def build_sparse_uniform_filter_matrix(
    *,
    input_size: int,
    subsampling_step: int | npt.ArrayLike,
    uniform_filter_window_size: int,
    axis: int,
    border_type: ConvolutionBorderType = ConvolutionBorderType.CONSTANT,
    dtype: np.dtype = np.float64,
) -> tuple[sp.sparse.csr_matrix | sp.sparse.csc_matrix, npt.NDArray[int]]:
    """
    Compute the (sparse) matrix that simultaneously executes:
      1) Uniform filtering
      2) Uniform subsampling

    In order to execute such filtering/subsampling of an [N x M] data matrix,
    compute the matrix by passing input_size=N (if filtering on the rows) or
    input_size=M (if filtering on the columns) and left/right multiply the input
    data by the matrix returned by this method.

    The user should decide on a case-by-case basis when this method is to be
    preferred over a combination of full a convolution and a subsampling. Usually,
    for small kernels and large subsamplign steps, this approach is more memory
    efficient and may be advantageous during memory demanding operations.

    Example
    -------
        # ...
        F, _ = build_sparse_uniform_matrix(
            input_size=raw_data.shape[0],
            subsampling_step=5,
            uniform_filter_window_size=7,
            axis=0,
        )

        filtered_data = F @ raw_data

    Parameters
    ----------
    input_size: int
        The size of the original data.

    subsampling_step: Union[int, npt.NDArray[int]]
        Uniformly subsample by taking 1 input every `subsampling_step` or
        a set of indices.

    uniform_filter_window_size: int
        The window size for the uniform filter. This must be an odd integer.

    axis: int
        The filtering direction. 0=filter on the rows, 1=filter the columns.
        If axis=0, a CSR sparse matrix is used to speed up row-vector multiplication.
        If axis=1, a CSC is used to speed up vector-column multiplications.

    border_type: ConvolutionBorderType = ConvolutionBorderType.CONSTANT
        If CONSTANT, the normalization term for the uniform filter is always
        the same. This is equivalent to add a 0-padding. If ISOLATED,
        the normalization term is equivalent to the number of element in the
        window (that is, it will be smaller at the matrix borders). Other method
        are not supported.

    dtype: np.dtype = np.float64
        The floating point precision of the output matrices. It defaults
        to np.float64

    Raises
    -------
    ValueError: If invalid inputs are passed.

    Return
    ------
    Union[sp.sparse.csc_matrix, sp.sparse.csr_matrix]
        The matrix that filters and subsamples the data.

    npt.NDArray[int]
        The subsampled indices.

    """
    # Quick checks on the input.
    if input_size <= 0:
        raise ValueError("input size must be a positive integer")
    if uniform_filter_window_size < 3 or uniform_filter_window_size % 2 == 0:
        raise ValueError("filter window size must be a positive odd integer")

    indices = np.array(subsampling_step)
    if indices.size == 1:
        indices = np.arange(0, input_size, subsampling_step)

    half_window_size = floor(uniform_filter_window_size / 2)

    # Just a quick lambda to retrieve the normalization value, in order to avoid
    # checking cases all the times, we define them with the same signature.
    if border_type == ConvolutionBorderType.CONSTANT:

        def normalization(*args):
            # pylint: disable=unused-argument
            return 1 / uniform_filter_window_size

    elif border_type == ConvolutionBorderType.ISOLATED:

        def normalization(begin, end):
            return 1 / (end - begin)

    else:
        raise ValueError("Only 'CONSTANT' and 'ISOLATED' are supported border types")

    # NOTE: We first use a LIL sparse matrix so we can easily/efficiently
    # execute block operations on it. Converting a LIL matrix to a CSC or CSR seems
    # to be having no noticeable runtime/memory overhead.
    if axis == 0:
        filter_matrix = sp.sparse.lil_matrix((indices.size, input_size), dtype=dtype)
        for row, index in enumerate(indices):
            begin = max(index - half_window_size, 0)
            end = min(index + half_window_size + 1, input_size)
            filter_matrix[row, begin:end] = np.full(end - begin, normalization(begin, end), dtype=dtype)
        return filter_matrix.tocsr(), indices.reshape(-1, 1)
    if axis == 1:
        filter_matrix = sp.sparse.lil_matrix((input_size, indices.size), dtype=dtype)
        for col, index in enumerate(indices):
            begin = max(index - half_window_size, 0)
            end = min(index + half_window_size + 1, input_size)
            filter_matrix[begin:end, col] = np.full(
                end - begin,
                normalization(begin, end),
                dtype=dtype,
            )
        return filter_matrix.tocsc(), indices.reshape(1, -1)

    raise ValueError("axis can only be 0 or 1")


def filter_validity_roi(
    data_shape: tuple[int, int],
    kernel_shape: tuple[int, int],
) -> RegionOfInterest:
    """
    Return the ROI on which filtering/convolution is valid, that is,
    the region within the original data that does not rely on padded
    borders.

    Parameters
    ----------
    kernel_shape: tuple[int, int]
        The shape of the kernel.

    data_shape: tuple[int, int]
        The shape of the data.

    Raises
    ------
    ValueError: If parameter is not a array shape.

    Return
    ------
    ROI (i.e. tuple[int, int, int, int])
        The validity ROI.

    """
    _raise_if_invalid_2d_shape(data_shape)
    _raise_if_invalid_2d_shape(kernel_shape)

    return (
        int(kernel_shape[0] / 2),
        int(kernel_shape[1] / 2),
        data_shape[0] - 2 * int(kernel_shape[0] / 2),
        data_shape[1] - 2 * int(kernel_shape[1] / 2),
    )


def _boxcar_kernel(shape: tuple[int, ...]) -> npt.NDArray[float]:
    """The uniform filter kernel."""
    _raise_if_invalid_2d_shape(shape)
    return np.full(shape, 1 / np.prod(shape))


def _raise_if_invalid_2d_shape(shape: tuple[int, int]):
    """Throw a ValueError if shape is invalid."""
    if len(shape) < 2 or any(d <= 0 for d in shape):
        raise ValueError(f"{shape} is not a valid shape")
