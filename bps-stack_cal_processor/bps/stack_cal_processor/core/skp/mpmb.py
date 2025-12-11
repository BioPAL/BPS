# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to assemble the Multi-Polarimetric-Multi-Baseline (MPMB) Matrix
-------------------------------------------------------------------------
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from itertools import product

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common.utils import floating_point_error_handler
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_numeric_types_equal,
)
from bps.stack_cal_processor.core.skp.utils import SkpRuntimeError


def get_mpmb_unique_pairs(
    *,
    num_polarizations: int,
    num_images: int,
    include_diagonal: bool = True,
    exclude_polarization_cross_cov: bool = False,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """
    The unique MPMB interferometric pairs.

    The function returns a list of the form

       (((0,0),(0,1)), ((0,0),(0,2)), ..., ((p0,n0),(p1,n1)), ...),

    where pX,nX is a polarization/image index pair. This function is meant to populate
    a MPMB matrix of size [Npol x Nimg, Npol x Nimg, Nazm, Nrg] that represents a
    block matrix where each block is Nimg x Nimg and the blocks are ordered in
    a Npol x Npol matrix.

              <- Nimg ->
           ^  +--------+--------+--------+ <--+
           |  |        |        |        |    |
         Nimg |        |        |        |    |
           |  |        |        |        |    |
           V  +--------+--------+--------+    |
              |        |        |        |    |
              |        |        |        |    +-- Npol blocks
              |        |        |        |    |
              +--------+--------+--------+    |
              |        |        |        |    |
              |        |        |        |    |
              |        |        |        |    |
              +--------+--------+--------+ <--+

    Parameters
    ----------
    num_polarizations: int
        The number of polarizations.

    num_images: int
        The number of images.

    include_diagonal: bool = True
        If true, include the interferometric pair ((pX,nX),(pX,nX)) on the diagonal.

    exclude_polarization_cross_cov: bool = False
        If true, consider only the interferometric pair ((pX,nX),(pX,nY)). It
        defaults to False.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    tuple[tuple[tuple[int,int], tuple[int,int]], ...]
        The MPMB interferometric (unique) pairs.

    """
    if num_polarizations <= 0:
        raise SkpRuntimeError("MPMB need at least 1 polarization")
    if num_images <= 0:
        raise SkpRuntimeError("MPMB needs at least 1 image")

    mpmb_pair_indices = tuple(product(range(num_polarizations), range(num_images)))
    mpmb_triu_indices = np.triu_indices(num_polarizations * num_images, 0 if include_diagonal else 1)

    mpmb_indices = ((mpmb_pair_indices[i], mpmb_pair_indices[j]) for i, j in zip(*mpmb_triu_indices))
    if exclude_polarization_cross_cov:
        mpmb_indices = filter(lambda m: m[0][0] == m[1][0], mpmb_indices)
    return tuple(mpmb_indices)


def to_mpmb_indices(
    pol_image_index_pair_0: tuple[int, int],
    pol_image_index_pair_1: tuple[int, int],
    *,
    num_polarizations: int,
    num_images: int,
) -> tuple[int, int]:
    """
    Convert two polarization/image index pairs in the MPMB matrix indices.

    This assumes a MPMB matrix of size [NpolxNimg, NpolxNimg, Nazm, Nrg] that
    represents a block matrix whose blocks are Nimg x Nimg. The blocks are
    ordered in a Npol x Npol matrix.

              <- Nimg ->
           ^  +--------+--------+--------+ <--+
           |  |        |        |        |    |
         Nimg |        |        |        |    |
           |  |        |        |        |    |
           V  +--------+--------+--------+    |
              |        |        |        |    |
              |        |        |        |    +-- Npol blocks
              |        |        |        |    |
              +--------+--------+--------+    |
              |        |        |        |    |
              |        |        |        |    |
              |        |        |        |    |
              +--------+--------+--------+ <--+

    Parameters
    ---------
    pol_image_index_pair_0: tuple[int, int]
        First interferometric index pair (pol/img).

    pol_image_index_pair_1: tuple[int, int]
        Second interferometric index pair (pol/img).

    num_polarizations: int
        Number of polarizations.

    num_images: int
        Number of images.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    int
        Row index.

    int
        Col index.

    """
    polarization_index_0, image_index_0 = pol_image_index_pair_0
    polarization_index_1, image_index_1 = pol_image_index_pair_1

    if num_polarizations <= 0:
        raise SkpRuntimeError("MPMB need at least 1 polarization")
    if num_images <= 0:
        raise SkpRuntimeError("MPMB needs at least 1 image")
    if not 0 <= polarization_index_0 * image_index_0 < num_images * num_polarizations:
        raise SkpRuntimeError("pol/img pair is out of range")
    if not 0 <= polarization_index_1 * image_index_1 < num_images * num_polarizations:
        raise SkpRuntimeError("pol/img pair is out of range")

    return (
        num_images * polarization_index_0 + image_index_0,
        num_images * polarization_index_1 + image_index_1,
    )


def assemble_mpmb_coherence_matrix_multithreaded(
    *,
    images: tuple[tuple[npt.NDArray[complex], ...], ...],
    azimuth_filter_matrix: sp.sparse.csr_matrix,
    range_filter_matrix: sp.sparse.csc_matrix,
    exclude_polarization_cross_cov: bool,
    dtypes: EstimationDType,
    num_worker_threads: int = 1,
) -> npt.NDArray[complex]:
    """
    Assemble the MPMB coherence matrix.

    The MPMB coherence matrix is defined as a matrix such that the entry i,j of
    that matrix is the coherence of the i-th and j-th multi-polarimetric
    pair. that is

       MPMB(pn0, pn0) = Coh(images[n0][p0], images[n1][p1])

    This assumes a MPMB matrix that represents a [Npol*Nimg x Npol*Nimg x Nazm x Nrg]
    block matrix whose blocks are [Nimg x Nimg]. The blocks are ordered in a
    [Npol x Npol] matrix.

              <- Nimg ->
           ^  +--------+--------+--------+ <--+
           |  |        |        |        |    |
         Nimg |        |        |        |    |
           |  |        |        |        |    |
           V  +--------+--------+--------+    |
              |        |        |        |    |
              |        |        |        |    +-- Npol blocks
              |        |        |        |    |
              +--------+--------+--------+    |
              |        |        |        |    |
              |        |        |        |    |
              |        |        |        |    |
              +--------+--------+--------+ <--+

    Parameters
    ----------
    images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    azimuth_filter_matrix: sp.sparse.csr_matrix
        The [Nazm' x Nazm] azimuth filtering matrix that simultaneously
        downsamples and filters along the azimuth components.

    range_filter_matrix: sp.sparse.csc_matrix
        The [Nrng x Nrng'] range filtering matrix that simultaneously
        downsamples and filters along the range components.

    exclude_polarization_cross_cov: bool
        If true, the MPMB coherence matrix will be zero on the
        cross polarizations.

    dtypes: EstimationDType
        The floating-point precision used for the calculations.

    num_worker_threads: int = 1
        How many threads to allocate for the assemblage.

    Raises
    ------
    AssertionError

    Return
    ------
    npt.NDArray[complex]
        The [Npol*Nimg x Npol*Nimg x Nazm' x Nrng'] multi-polarimetric
        multi-baseline coherence matrix.

    """
    # The stack images sizes.
    num_images = len(images)
    num_polarizations = len(images[0])

    # We need to list all possible inteferometric pairs. We compute also the
    # values on the diagonal matrix to later normalize the values.
    mpmb_unique_pairs = get_mpmb_unique_pairs(
        num_polarizations=num_polarizations,
        num_images=num_images,
        include_diagonal=True,
        exclude_polarization_cross_cov=exclude_polarization_cross_cov,
    )

    # NOTE: There is no significant different in how rng/azm and img/pol are
    # ordered. Extracting the MPMB coherence (i.e. slicing the matrix by fixing
    # azm and rng) takes in both case O(500ns).
    mpmb_coherences = np.zeros(
        (
            num_polarizations * num_images,
            num_polarizations * num_images,
            azimuth_filter_matrix.shape[0],
            range_filter_matrix.shape[1],
        ),
        dtype=dtypes.complex_dtype,
    )

    # NOTE: The routine to populate the MPMB coherence. Since we are computing
    # all possible interferometric pairs, the interferogram on the the same
    # pol/img pair can be reused to compute the normalization terms for the
    # off-diagonal coherences. Note also that the index pair (i,j) below is only
    # related to the upper triangular part of the MPMB matrix, as per
    # construction of the unique index pairs.

    # Execute all multithreaded. We need to process the diagonal separately
    # to avoid messing up the normalization of the off-diagonal terms.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # First, we compute the MPMB covariance.
        mpmb_assemblage_indices = tuple(
            (
                mpmb_pair[0],
                mpmb_pair[1],
                *to_mpmb_indices(
                    mpmb_pair[0],
                    mpmb_pair[1],
                    num_polarizations=num_polarizations,
                    num_images=num_images,
                ),
            )
            for mpmb_pair in mpmb_unique_pairs
        )

        def populate_mpmb_covariance_matrix_fn(args):
            return _populate_mpmb_covariance_matrix(
                mpmb_output_matrix=mpmb_coherences,
                mpmb_pair_h=args[0],
                mpmb_pair_k=args[1],
                coh_i=args[2],
                coh_j=args[3],
                images=images,
                azimuth_filter_matrix=azimuth_filter_matrix,
                range_filter_matrix=range_filter_matrix,
            )

        # Assemble and check that we hit no unexpected errors.
        assemblage_ok = reduce(
            lambda a, b: a and b,
            executor.map(populate_mpmb_covariance_matrix_fn, mpmb_assemblage_indices),
        )

        # Normalize the off-diagonal terms.
        def normalize_mpmb_covariance_off_diagonal_matrix_fn(coh_indices):
            return _normalize_mpmb_covariance_off_diagonal_matrix(
                mpmb_output_matrix=mpmb_coherences,
                coh_i=coh_indices[0],
                coh_j=coh_indices[1],
                hermitize=True,
            )

        # Normalize and check that we hit no unexpected errors.
        assemblage_ok &= reduce(
            lambda a, b: a and b,
            executor.map(
                normalize_mpmb_covariance_off_diagonal_matrix_fn,
                zip(*np.triu_indices(num_polarizations * num_images, 1)),
            ),
        )

        # Normalize he diabonal terms.
        def normalize_mpmb_covariance_diagonal_matrix_fn(coh_i):
            return _normalize_mpmb_covariance_diagonal_matrix(
                mpmb_output_matrix=mpmb_coherences,
                coh_i=coh_i,
            )

        # Normalize and check that we hit no unexpected errors.
        assemblage_ok &= reduce(
            lambda a, b: a and b,
            executor.map(
                normalize_mpmb_covariance_diagonal_matrix_fn,
                range(num_polarizations * num_images),
            ),
        )

    assert_numeric_types_equal(mpmb_coherences, expected_dtype=dtypes.complex_dtype)

    return mpmb_coherences, assemblage_ok


def _populate_mpmb_covariance_matrix(
    *,
    mpmb_output_matrix: npt.NDArray[complex],
    mpmb_pair_h: tuple[int, int],
    mpmb_pair_k: tuple[int, int],
    coh_i: int,
    coh_j: int,
    images: tuple[tuple[npt.NDArray[complex], ...], ...],
    azimuth_filter_matrix: sp.sparse.csr_matrix,
    range_filter_matrix: sp.sparse.csc_matrix,
) -> bool:
    """Populate in place the (i, j)-block of the MPMB cohrence matrix."""
    # We ignore the floating point errors, we simply warn the users, we will
    # handle them later.
    #
    # NOTE: This is called in multiple threads, GIL makes it safe.
    np.seterr(all="call")
    np.seterrcall(
        partial(
            floating_point_error_handler,
            logging.DEBUG,
            _populate_mpmb_covariance_matrix.__name__,
        )
    )

    mpmb_output_matrix[coh_i, coh_j, ...] = (
        azimuth_filter_matrix
        @ (images[mpmb_pair_h[1]][mpmb_pair_h[0]] * np.conj(images[mpmb_pair_k[1]][mpmb_pair_k[0]]))
        @ range_filter_matrix
    )

    # We need to make sure that we are not dividing by zeros or doing any nasty
    # operations with NaN, inf from. This also handles possible floating point
    # errors we suppressed.
    valid_ij = np.isfinite(mpmb_output_matrix[coh_i, coh_j])
    mpmb_output_matrix[coh_i, coh_j, ~valid_ij] = 0.0

    return np.all(valid_ij)


def _normalize_mpmb_covariance_off_diagonal_matrix(
    *,
    mpmb_output_matrix: npt.NDArray[complex],
    coh_i: int,
    coh_j: int,
    hermitize: bool,
) -> bool:
    """Normalize in-place the (i, j)-block of the MPMB covariance."""
    # We ignore the floating point errors, we simply warn the users, we will
    # handle them later.
    #
    # NOTE: This is called in multiple threads, GIL makes it safe.
    np.seterr(all="call")
    np.seterrcall(
        partial(
            floating_point_error_handler,
            logging.DEBUG,
            _normalize_mpmb_covariance_off_diagonal_matrix.__name__,
        )
    )

    # NOTE: These terms are purely real, since they are the diagonal
    # terms. We take the real part to force casting.
    cov_ij = np.real(mpmb_output_matrix[coh_i, coh_i, ...] * mpmb_output_matrix[coh_j, coh_j, ...])

    # We need to make sure that we are not dividing by zeros or doing any nasty
    # operations with NaN, inf from. This also handles possible floating point
    # errors we suppressed.
    valid_ij = (cov_ij > 0) & np.isfinite(cov_ij)
    mpmb_output_matrix[coh_i, coh_j, valid_ij] /= np.sqrt(cov_ij[valid_ij])
    mpmb_output_matrix[coh_i, coh_j, ~valid_ij] = 0.0

    # MPMB coherence matrices are blockwise Hermitian.
    if hermitize:
        mpmb_output_matrix[coh_j, coh_i] = np.conj(mpmb_output_matrix[coh_i, coh_j])

    return np.all(valid_ij)


def _normalize_mpmb_covariance_diagonal_matrix(
    *,
    mpmb_output_matrix: npt.NDArray[complex],
    coh_i: int,
) -> bool:
    """Normalize in place the (i, i)-block of the covariance matrix."""
    # We ignore the floating point errors, we simply warn the users, we will
    # handle them later.
    #
    # NOTE: This is called in multiple threads, GIL makes it safe.
    np.seterr(all="call")
    np.seterrcall(
        partial(
            floating_point_error_handler,
            _normalize_mpmb_covariance_diagonal_matrix.__name__,
        )
    )

    cov_ii = np.real(mpmb_output_matrix[coh_i, coh_i])
    valid_ii = (cov_ii > 0) & np.isfinite(cov_ii)
    mpmb_output_matrix[coh_i, coh_i, valid_ii] = 1.0
    mpmb_output_matrix[coh_i, coh_i, ~valid_ii] = 0.0

    return np.all(valid_ii)
