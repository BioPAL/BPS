# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities for Selecting Inteferometric Pairs
--------------------------------------------
"""

from itertools import combinations

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.core.utils import critical_vertical_wavenumber


class MultiBaselineInterferometricPairingError(RuntimeError):
    """Handle error in Multi-baseline pairing."""


def multi_baseline_interferometric_pair_indices(
    num_images: int,
    *,
    vertical_wavenumbers: tuple[npt.NDArray[float], ...],
    range_bandwidth: float,
    cb_ratio_threshold: float,
) -> tuple[tuple[int, int], ...]:
    """
    The Multi-Baseline (MB) interferometric pairs (f{i}, f{j). The
    pair are selected so they are guaranteed to have sufficient
    overlap (in terms of critical baseline)

    Example
    -------
    Unordered adjacency diagram of the MB pairs for topological
    for say 45% overalapping with CB:

          f1 f2 f3 f4 f5 f6 f7 ...
       f1     *  *  *
          f2     *  *  *
             f3     *  *  *
                f4     *  *  *
                   f5     *  *
                      f6     * ...
                         f7
                            ...

    Parameters
    ----------
    num_images: int
        Number of images in the stack.

    vertical_wavenumbers: tuple[np.NDArray[float], ...] [rad/m]
        The Nimg vertical wavenumbers of shape [Nazm x Nrng].

    range_bandwidth: float [Hz]
        Bandwidth in slant-range direction. Defaults to 6MHz.

    cb_ratio_threshold: float
        Threshold on percentage of critical baseline to keep an
        inteferometric pair.

    Raises
    ------
    ValueError

    Return
    ------
    tuple[tuple[int, int], ...]
        The inteferometric pair indices.

    """
    if num_images < 2:
        raise ValueError("Need at least 2 images in stack")
    if len(vertical_wavenumbers) != num_images:
        raise ValueError("Invalid number of Kz data")
    if range_bandwidth <= 0.0:
        raise ValueError("Range bandwidth must be positive")
    if cb_ratio_threshold <= 0.0:
        raise ValueError("CB ratio threshold must be positive")

    # Trivial case.
    if num_images <= 1:
        return tuple()

    # The baselines (as percentage of CB) for all inteferometric pairs.
    interferometric_pairs = tuple(combinations(range(num_images), 2))
    critical_baseline_ratios = critical_baseline_ratio(
        interferometric_pairs,
        vertical_wavenumbers,
        range_bandwidth,
    )

    # The interferometric pairs that have baseline below threshold.
    interferometric_pairs = tuple(
        itf_pair
        for itf_pair in interferometric_pairs
        if np.abs(critical_baseline_ratios[itf_pair]) <= cb_ratio_threshold
    )

    # We want to make sure that each image belongs to at least 1 inteferometric
    # pair, otherwise it won't be possible to estimate any quantity via MB
    # estimation for images that are not covered by pairs.
    if not validate_interferometric_pairs(interferometric_pairs, num_images):
        raise MultiBaselineInterferometricPairingError(
            "Selected MB thresh={} results in too few pairs {}".format(cb_ratio_threshold, interferometric_pairs)
        )

    return interferometric_pairs


def single_baseline_interferometric_pair_indices(
    num_images: int,
    *,
    reference_image_index: int,
) -> tuple[tuple[int, int], ...]:
    """
    The Single-Baseline (SB) method uses only the interferogram
    pairs that contain the reference image (i.e. the coregistration
    primary or the calibration reference).

    Example
    -------
    Unordered adjacency diagram of the SB pairs for reference
    image 4:

          f1 f2 f3 f4 f5 f6 f7 ...
       f1           *
          f2        *
             f3     *
                f4    *  *  *  ...
                   f5
                      f6
                         f7
                            ...

    Parameters
    ---------
    num_images: int
        The number of stack frames.

    reference_image_index: int
        The reference image index.

    Raises
    ------
    ValueError

    Return
    ------
    tuple[tuple[int, int], ...]
        The inteferometric pair indices.

    """
    # Trivial case.
    if num_images == 0:
        return tuple()

    # Check if the reference image index is consistent with the size of the
    # stack.
    if not 0 <= reference_image_index < num_images:
        raise ValueError(f"Invalid {reference_image_index=}")

    return tuple((reference_image_index, i) for i in range(num_images) if i != reference_image_index)


def sequential_baseline_interferometric_pair_indices(
    num_images: int,
    *,
    reference_image_index: int,
    flatten: bool,
) -> tuple[tuple[int, int], ...] | tuple[int, int]:
    """
    Compute the squential interferometric pairs.

    Sequential interferometric pairs are defined as follows. Assume
    f5 to be the reference image (e.g. calibration primary):

                   f1  f2  f3  f4  f5  f6  f7  f8  f9
                               *   **   *
                           *   *        *   *
                       *   *                *   *
                   *   *                        *   *

    Parameters
    ----------
    num_images: int
        The number of stack frames.

    reference_image_index: int
        The reference image index.

    flatten: bool
        If true, sequentially yields 1 pair at a time (left first, then
        right). If false, yields the left/right pair at each call.

    Raises
    ------
    ValueError

    Example
    -------
    >>> list(sequential_baseline_interferometric_pair_indices(7, reference_image_index=2, flatten=False))
    [((1, 2), (2, 3)), ((0, 1), (3, 4)), ((4, 5),), ((5, 6),)]

    >>> list(sequential_baseline_interferometric_pair_indices(7, reference_image_index=2, flatten=True))
    [(1, 2), (2, 3), (0, 1), (3, 4), (4, 5), (5, 6)]

    Yields
    ------
    tuple[tuple[int, int], ...]
        If `flatten` is false, the next left and right interferometric
        pair indices and a single tuple if only 1 adjacent pair exists.
        If `flatten` is set to true, the next pair (left first, then
        right).

    """
    if num_images < 2:
        raise ValueError("Al least 2 images are needed for sequential pairs")
    if not 0 <= reference_image_index < num_images:
        raise ValueError(f"Invalid {reference_image_index=}")

    # Helper function to compute the next left pair.
    def move_left_fn(pair):
        if pair is None or pair[0] <= 0:
            return None
        return (pair[0] - 1, pair[1] - 1)

    # Helper function to compute the next right pair.
    def move_right_fn(pair):
        if pair is None or pair[1] >= num_images - 1:
            return None
        return (pair[0] + 1, pair[1] + 1)

    left_pair = None
    if reference_image_index > 0:
        left_pair = (reference_image_index - 1, reference_image_index)

    right_pair = None
    if reference_image_index < num_images - 1:
        right_pair = (reference_image_index, reference_image_index + 1)

    while left_pair is not None or right_pair is not None:
        # If flattened, yield one pair at a time. Otherwise, yield
        # both pairs. We need to reverse the left tuple to have it
        # sorted as (primary, secondary).
        if flatten:
            if left_pair is not None:
                yield left_pair[::-1]
            if right_pair is not None:
                yield right_pair
        else:
            pair_out = []
            if left_pair is not None:
                pair_out.append(left_pair[::-1])
            if right_pair is not None:
                pair_out.append(right_pair)
            yield tuple(pair_out)

        left_pair = move_left_fn(left_pair)
        right_pair = move_right_fn(right_pair)


def validate_interferometric_pairs(
    interferometric_pair_indices: tuple[tuple[int, int], ...],
    num_images: int,
) -> bool:
    """
    Check if the current set of interferometric pair indices can be
    used for a differential estimation.

    It is apparent from the information theory's standpoint that that is
    the case if and only if the undirected graph with Nimg vertices and
    edges defined by the interferometric pairs must be connected (i.e.
    it has 1 and only 1 connected component).

    Parameters
    ----------
    interferometric_pair_indices: tuple[tuple[int, int], ...]
        The collection of image index pairs that needs to be verified.

    num_images: int
        The total number of images.

    Raises
    ------
    ValueError

    Return
    ------
    bool
        True, if the set of interferometric pair indices are
        usable for a differential estimation.

    """
    if num_images <= 0:
        raise ValueError(f"{num_images=} must be positive")
    if len(interferometric_pair_indices) > 0 and (
        np.max(interferometric_pair_indices) >= num_images or np.min(interferometric_pair_indices) < 0
    ):
        raise ValueError(f"Invalid interferometric pairs of {num_images=}")

    # First check that all image indexes are contained in at least 1 pair. If
    # not, we cannot do any differential estimation for each image.
    if np.unique(interferometric_pair_indices).size != num_images:
        return False

    # We now only need to prove that the adjacency matrix of the directed graph
    # defined by the interferometric pairs has and only 1 connected component.
    # It is well know from graph theory that the number of connected components
    # of a graph is the dimension of the null-space of it's Laplacian matrix.
    adjacency_matrix = np.zeros((num_images, num_images), dtype=np.int8)
    for i, j in interferometric_pair_indices:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    laplacian_matrix = sp.sparse.csgraph.laplacian(adjacency_matrix)
    return sp.linalg.null_space(laplacian_matrix).shape[1] == 1


def critical_baseline_ratio(
    interferometric_pairs: tuple[tuple[int, int], ...],
    vertical_wavenumbers: tuple[npt.NDArray[float], ...],
    range_bandwidth: float,
) -> dict[tuple[int, int], float]:
    """
    Compute the inteferometric baselines as percentage of the critical
    baselines. Note that the results are snapped to the expected
    baselines for the BIOMASS system (e.g. 0%CB, +/-15%CB, +/-30%CB,
    +/-45%CB, ..., +/-90%CB).

    Parameters
    ----------
    interferometric_pairs: tuple[tuple[int, int], ...]
        Baselines will be evaluated for the specified inteferometric
        pairs.

    vertical_wavenumbers: tuple[npt.NDArray[float], ...] [rad/m]
        Nimg vertival wavenumber data of shape [Nazm x Nrng].

    range_bandwidth: float [Hz]
        THe bandwidth in slant-range direction.

    Raises
    ------
    ValueError

    Return
    ------
    dict[tuple[int, int], float]  [%CB]
        Dictionary with the percentage of critical baseline for each
        interferometric pair.

    """
    if range_bandwidth <= 0:
        raise ValueError("Range bandwidth must be positive")
    if min(min(p) for p in interferometric_pairs) < 0 or len(vertical_wavenumbers) <= max(
        max(p) for p in interferometric_pairs
    ):
        raise ValueError("Kz data is inconsistent with the interferometric pairs ")

    critical_kz = critical_vertical_wavenumber(range_bandwidth)
    assert critical_kz > 0, "Critical Kz cannot be negative"

    # Compute the baselines
    return {
        (p, s): np.mean(vertical_wavenumbers[p] - vertical_wavenumbers[s]) / critical_kz
        for p, s in interferometric_pairs
    }
