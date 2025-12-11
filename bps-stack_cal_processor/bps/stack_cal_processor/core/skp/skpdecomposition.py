# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Sum of Krokecker Products (SKP) decomposition
-------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numba as nb
import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.skp.utils import (
    SkpRuntimeError,
    cache_volume_noise,
    joint_diagonalization,
    normalize_coherence,
)

# A constant that can be tuned but it is not user-defined.
SKP_RANK = 2

# The altitude search hypotheses/sanmples.
Z_RANGE_DEFAULT = (-60.0, 60.0)  # [m]
Z_RANGE_DEFAULT_SAMPLES = 61  # That is, 1 sample every 1 m.


# NOTE: The routine within this module use lots of matrices. It makes sense to
# call matrices with capitalized letters. We suppress the pylint error to avoid
# too many complains.
# pylint: disable=invalid-name


@nb.njit(cache=True, nogil=True)
def skp_low_rank_decomposition(
    mpmb_coherence: npt.NDArray[complex],
    C_shape: tuple[int, int],
    R_shape: tuple[int, int],
    rank: int,
    dtype: np.dtype,
) -> tuple[
    tuple[npt.NDArray[complex], ...],
    tuple[npt.NDArray[complex], ...],
    tuple[float, ...],
    bool,
]:
    """
    Compute the Sum-of-Kronecker decomposition of matrix A by running a
    Singular Value Decomposition (SVD) of a rearranged matrix derived
    from A. In other words, A is decomposed as

        sum_{k=1...rank} l{k} B{k} [+] C{k},

    where [+] is the Kronecker product. The decomposition is an approximation
    and the quality of the approximation is controlled by the rank parameter.
    The lower-rank SKP decomposition of A is obtained via SVD decomposition of
    the rearranged matrix RA defined as follows: (n=Nimg, m=Npol)

              n                                     n^2
            +---+---+---+           +----------------------------------+
          n | 0 | 1 | 2 |         m |           0   (0/3/6)            |
            |---+---+---|           |----------------------------------|
        A = | 3 | 4 | 5 |  -> RA =  |           1   (1/4/7)            |
            |---+---+---|           |----------------------------------|
            | 6 | 7 | 8 |           |           2   (2/5/8)            |
            +---+---+---+           +----------------------------------+

    where RA{i}[k,:] = col_wise_flattening(A{3*k+i}), i=1...m, k=1...m.

    Parameters
    ----------
    mpmb_coherence: npt.NDarray[complex]
        The [Nimg * Npol x Nimg * Npol] multi-polarimetric multi-baseline
        coherence matrix.

    C_shape: tuple[int, int]
        Shape of the matrices C{k} (k=1,...,rank)

    R_shape: tuple[int, int]
        Shape of the matrices R{k} (k=1,...,rank)

    rank: int
        Maximum approximation rank.

    dtype: np.dtype
        The floating point precision of the output matrices.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    tuple[npt.NDArray[complex], ...]
        The matrices [C{k} for k=1, ..., rank].

    tuple[npt.NDArray[complex], ...]
        The matrices [R{k} for k=1, ..., rank].

    tuple[float, ...]
        The eigenvalues [S{k} for k=1, ..., rank].

    bool
        As to whether the solution is usable or not.

    """
    # Minimal checks on the input.
    if rank < 1:
        raise SkpRuntimeError("approximation rank must be a positive integer")
    if mpmb_coherence.shape[0] != C_shape[0] * R_shape[0]:
        raise SkpRuntimeError("mpmb_coherence.rows must be equal to C.rows * R.rows")
    if mpmb_coherence.shape[1] != C_shape[1] * R_shape[1]:
        raise SkpRuntimeError("mpmb_coherence.cols must be equal to C.cols * R.cols")

    # This notation may seem silly, but in practice C and R are square
    # matrices, so this makes it easier to read...
    m, m_ = C_shape
    n, n_ = R_shape

    # NOTE: In order to JIT the code, we need to implement the following, without
    # closures and without vstack, since neither are supported by numba.
    #
    #   np.vstack(
    #       tuple(
    #           np.reshape(mpmb_coherence[i * n : (i + 1) * n, k * n_ : (k + 1) * n_], -1, order="F")
    #           for k in range(m_)
    #           for i in range(m)
    #       )
    #   )
    #
    RA = np.empty((m * m_, n * n_), dtype=dtype)

    # pylint: disable=not-an-iterable
    for k in nb.prange(m_):
        for i in nb.prange(m):
            RA[k * m_ + i, :] = mpmb_coherence[i * n : (i + 1) * n, k * n_ : (k + 1) * n_].T.flatten()

    # Compute the single value decomposition.
    U, L, Vt = np.linalg.svd(RA)

    # NOTE: We need to transpose the matrix Vt (but not conjugate it).
    # Again, numba complains with reshaping with Fortran-like ordering. We need to
    # somehow rearrange
    #
    #   np.array([np.reshape(U[:, r], C_shape, order="F") for r in range(rank)]),
    #   np.array([np.reshape(Vt[r, :], R_shape, order="F") for r in range(rank)]).
    #
    # Below, flattening is required to guarantee contiguous memory, otherwise numba
    # fails to release the GIL.
    Cs = np.empty((rank, *C_shape), dtype=dtype)
    Rs = np.empty((rank, *R_shape), dtype=dtype)

    # Flag if the decomposition is valid.
    solution_is_usable = np.isfinite(U).all() and np.isfinite(L).all() and np.isfinite(Vt).all()
    if not solution_is_usable:
        return Cs, Rs, L, False

    for r in nb.prange(rank):
        Cs[r, ...] = np.reshape(U[..., r].flatten(), C_shape).T
        Rs[r, ...] = np.reshape(Vt[r, ...].flatten(), R_shape).T
        # To be usable, the solution must have full rank.
        solution_is_usable &= np.linalg.matrix_rank(Cs[r]) == min(*C_shape)
        solution_is_usable &= np.linalg.matrix_rank(Rs[r]) == min(*R_shape)
        if not solution_is_usable:
            break

    return Cs, Rs, L[0:rank], solution_is_usable


@nb.njit(cache=True, nogil=True)
def skp_semi_positiveness_joint_range(
    Cs: npt.NDArray[complex],
    Rs: npt.NDArray[complex],
    dtype: np.dtype,
) -> tuple[npt.NDArray[float], npt.NDArray[float], bool]:
    """
    Compute the range of values for parameters `a` and `b` such that matrices
    Cv, Cg, Rg, and Rv defined as

       Cv(a) := (a-1)*C{0} + ( a )*C{1},
       Cg(b) := (1-b)*C{0} - ( b )*C{1},
       Rg(a) := ( a )*R{0} + (1-a)*R{1},
       Rv(b) := ( b )*R{0} + (1-b)*R{1}

    are semi-positive definite.

    Parameters
    ----------
    Cs: npt.NDArray[complex]
        The C{k} matrices in the SKP decomposition (k=0,1).

    Rs: npt.NDArray[complex]
        The R{k} matrices in the SKP decomposition (k=0,1).

    dtype: np.dtype
        The floating point precision of the output.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    npt.NDArray[float]
        The [a_min, a_max] range of semi-positiveness definition.

    npt.NDArray[float]
        The [b_min, b_max] range of semi-positiveness definition

    bool
        A flag as to whether the solition is usable or not.

    """
    # Just a check on the shape.
    if Cs.shape[0] != SKP_RANK or Rs.shape[0] != 2:
        raise SkpRuntimeError("Only SKP rank=2 is supported.")
    if not np.isfinite(Cs).all():
        raise SkpRuntimeError("Cs matrix contains infs or NaNs")
    if not np.isfinite(Rs).all():
        raise SkpRuntimeError("Rs matrix contains infs or NaNs")

    # All good for now.
    is_solution_usable = True

    # Semi-positive definiteness of Rs.
    U = joint_diagonalization(Rs[0], Rs[1])
    Uh = U.conj().T

    D0 = np.real(np.diag(Uh @ Rs[0] @ U))
    D1 = np.real(np.diag(Uh @ Rs[1] @ U))
    Dd = D1 - D0
    if np.all(Dd == 0):
        Dd = np.full_like(Dd, np.finfo(dtype).eps)

    D_lh = D1 / Dd
    D0_gt_D1 = D0 > D1
    D0_lt_D1 = D0 < D1

    ab_l, ab_h = -np.inf, np.inf
    if np.any(D0_gt_D1):
        ab_l = np.max(D_lh[D0_gt_D1])
    if np.any(D0_lt_D1):
        ab_h = np.min(D_lh[D0_lt_D1])

    # Semi-positive definteness of Cs.
    U = joint_diagonalization(Cs[0], Cs[1])
    Uh = U.conj().T

    D0 = np.real(np.diag(Uh @ Cs[0] @ U))
    D1 = np.real(np.diag(Uh @ Cs[1] @ U))
    Ds = D0 + D1
    if np.all(Ds == 0):
        Ds = np.full_like(Ds, np.finfo(dtype).eps)

    D_lh = D0 / Ds
    D0_gt_mD1 = D0 > -D1
    D0_lt_mD1 = D0 < -D1

    a_l, a_h = -np.inf, np.inf
    if np.any(D_lh[D0_gt_mD1]):
        a_l = np.max(D_lh[D0_gt_mD1])
    if np.any(D_lh[D0_lt_mD1]):
        a_h = np.min(D_lh[D0_lt_mD1])

    b_l, b_h = -np.inf, np.inf
    if np.any(D_lh[D0_lt_mD1]):
        b_l = np.max(D_lh[D0_lt_mD1])
    if np.any(D_lh[D0_gt_mD1]):
        b_h = np.min(D_lh[D0_gt_mD1])

    if a_l < b_h:
        is_solution_usable = False

    a = np.array([max(a_l, ab_l), min(a_h, ab_h)], dtype=dtype)
    b = np.array([max(b_l, ab_l), min(b_h, ab_h)], dtype=dtype)
    if a[1] < a[0] or b[1] < b[0]:
        is_solution_usable = False

    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        is_solution_usable = False

    return a, b, is_solution_usable


@nb.njit(cache=True, nogil=True)
def skp_estimate_scattering_coherences(
    mpmb_coherence: npt.NDArray[complex],
    num_images: int,
    num_polarizations: int,
    fallback_noise: npt.NDArray[complex],
    float_dtype: np.dtype,
    complex_dtype: np.dtype,
) -> tuple[npt.NDArray[complex], bool]:
    """
    Compute the interferometric covariance of the ground scattering mechanism.

    Parameters
    ----------
    mpmb_coherence: npt.NDArray[complex]
        THe [Nimg*Npol x Nimg*Npol] Multi-polarimetric coherence matrix.

    num_images: int
        The number of images in the stack (Nimg).

    num_polarizations: int
        The number of polarizations (Npol).

    fallback_noise: npt.NDArray[complex]
        The volumetric noise to use in case the SKP decomposition fails.
        This must be a [Nimg x Nimg] complex matrix.

    float_dtype: np.dtype
        The floating point precision of the float values.

    complex_dtype: np.dtype
        The floating point precision of the complex values.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    npt.NDArray[complex]
         The 4 coherence matrices associated with ground and canopy encode
         as a [4 x Nimg x Nimg] array.

    bool
         As to whether the decompositions encountered some errors.

    """
    # First, compute the SKP decomposition.
    Cs, Rs, L, valid_skp_decomposition = skp_low_rank_decomposition(
        mpmb_coherence=mpmb_coherence,
        C_shape=(num_polarizations, num_polarizations),
        R_shape=(num_images, num_images),
        rank=SKP_RANK,
        dtype=complex_dtype,
    )

    if not valid_skp_decomposition or np.prod(Rs[:, 0, 0]) == 0:
        # If something goes wrong, we return the fallback solution.
        return (
            fallback_skp_scattering_coherences(
                mpmb_coherence,
                fallback_noise,
                num_images,
                num_polarizations,
                complex_dtype,
            ),
            valid_skp_decomposition,
        )

    # Rescale Cs and Rs.
    # pylint: disable-next=not-an-iterable
    for r in nb.prange(SKP_RANK):
        Cs[r, ...] *= Rs[r, 0, 0] * L[r]
        Rs[r, ...] /= Rs[r, 0, 0]

    # Compute the ground and canopy matrices.
    (a_l, a_h), (b_l, b_h), valid_joint_range = skp_semi_positiveness_joint_range(Cs, Rs, float_dtype)
    if not valid_joint_range:
        # Again, if something goes wrong, we return the fallback solution.
        return (
            fallback_skp_scattering_coherences(
                mpmb_coherence,
                fallback_noise,
                num_images,
                num_polarizations,
                complex_dtype,
            ),
            valid_joint_range,
        )

    # We will store the output here. Always 4 matrices are returned.
    Rskp = np.empty((4, num_images, num_images), dtype=complex_dtype)

    # Compute the ground and canopy matrices.
    Rg_a_l, Rg_a_h = a_l * Rs[0] + (1 - a_l) * Rs[1], a_h * Rs[0] + (1 - a_h) * Rs[1]
    Rv_b_l, Rv_b_h = b_l * Rs[0] + (1 - b_l) * Rs[1], b_h * Rs[0] + (1 - b_h) * Rs[1]

    # Convert into coherence matrices.
    coh_Rg_a_l = normalize_coherence(Rg_a_l, complex_dtype)
    coh_Rg_a_h = normalize_coherence(Rg_a_h, complex_dtype)
    coh_Rv_b_l = normalize_coherence(Rv_b_l, complex_dtype)
    coh_Rv_b_h = normalize_coherence(Rv_b_h, complex_dtype)

    # Compute the quality as follows:
    #
    #  Coh(A) = 1/|Tri| sum_{(i,j) in Tri} |a{i,j}|/sqrt(a{i,i}*a{j,j}),
    #
    # where Tri is the upper triangular part of A.
    #
    # NOTE: this may be a bit inefficient, but this is the only way to have
    # this precompiled with numba.
    alpha = 2 / (num_images**2 - num_images)
    q_Rg_a_l = alpha * np.sum(np.abs(np.triu(coh_Rg_a_l, 1)))
    q_Rg_a_h = alpha * np.sum(np.abs(np.triu(coh_Rg_a_h, 1)))
    q_Rv_b_l = alpha * np.sum(np.abs(np.triu(coh_Rv_b_l, 1)))
    q_Rv_b_h = alpha * np.sum(np.abs(np.triu(coh_Rv_b_h, 1)))

    # Select the SKP decomposition according to the best coherence.
    if max(q_Rg_a_l, q_Rg_a_h) > max(q_Rv_b_l, q_Rv_b_h):
        Rskp[0, ...] = coh_Rg_a_l if q_Rg_a_l > q_Rg_a_h else coh_Rg_a_h
        Rskp[1, ...] = coh_Rg_a_h if q_Rg_a_l > q_Rg_a_h else coh_Rg_a_l
        Rskp[2, ...] = coh_Rv_b_l if q_Rv_b_l > q_Rv_b_h else coh_Rv_b_h
        Rskp[3, ...] = coh_Rv_b_h if q_Rv_b_l > q_Rv_b_h else coh_Rv_b_l
        return Rskp, valid_skp_decomposition and valid_joint_range

    Rskp[0, ...] = coh_Rv_b_l if q_Rv_b_l > q_Rv_b_h else coh_Rv_b_h
    Rskp[1, ...] = coh_Rv_b_h if q_Rv_b_l > q_Rv_b_h else coh_Rv_b_l
    Rskp[2, ...] = coh_Rg_a_l if q_Rg_a_l > q_Rg_a_h else coh_Rg_a_h
    Rskp[3, ...] = coh_Rg_a_h if q_Rg_a_l > q_Rg_a_h else coh_Rg_a_l

    return Rskp, valid_skp_decomposition and valid_joint_range


@nb.njit(cache=True, nogil=True)
def fallback_skp_scattering_coherences(
    mpmb_coherence: npt.NDArray[complex],
    volumetric_noise: npt.NDArray[complex],
    num_images: int,
    num_polarizations: int,
    dtype: np.dtype,
) -> tuple[npt.NDArray[complex], ...]:
    """
    Fallback strategy for the SKP scattering coherence when the SKP
    decomposition fails.

    The output scattering coherence Rskp is defined as:

      Rskp[0] := MPMB_Coherence[0, 0],
      Rskp[2] := MPMB_Coherence[Npol-1, Npol-1],
      Rskp[3] := Random volumetric noise,

    where MPMB_Coherence[i, j] is the (i, j) matrix block of size
    [Nimg x Nimg]. The remaining term is defined as follows:

      Rskp[1] :=
         MPMB_Coherence[1,1] if Npol==3,
         (MPMB_Coherence[1,1] + MPMB_Coherence[2,2]) / 2, if Npol==4

    Note that SKP supports only 3 or 4 polarizations.

    Parameters
    ----------
    mpmb_coherence: npt.NDArray[complex]
        The [Nimg*Npol x Nimg*Npol] multi-polarimetric multi-baseline
        coherence matrix.

    volumetric_noise: npt.NDArray[complex]
        The [Nimg x Nimg] random matrix representing a volumetric noise.

    num_images: int
        The number of images (Nimg > 0).

    num_polarizations: int
        The number of polarizations (Npol). Must be 3 or 4.

    dtype: np.dtype
        The floating point precision of the output.

    Return
    ------
    npt.NDArray[complex]
        The [4 x Nimg x Nimg] SKP scattering coherence matrices.

    """
    # For the sake of efficiency, we skip all checks on the arguments.
    # All inputs have been validated upstream of this method.
    Rskp = np.empty((4, num_images, num_images), dtype=dtype)

    Rskp[0, ...] = normalize_coherence(mpmb_coherence[0:num_images, 0:num_images], dtype)
    Rskp[2, ...] = normalize_coherence(mpmb_coherence[-num_images:, -num_images:], dtype)
    Rskp[3, ...] = volumetric_noise

    if num_polarizations == 3:
        Rskp[1, ...] = normalize_coherence(
            mpmb_coherence[
                num_images : 2 * num_images,
                num_images : 2 * num_images,
            ],
            dtype,
        )

    if num_polarizations == 4:
        Rskp[1, ...] = normalize_coherence(
            0.5
            * (
                mpmb_coherence[
                    num_images : 2 * num_images,
                    num_images : 2 * num_images,
                ]
                + mpmb_coherence[
                    2 * num_images : 3 * num_images,
                    2 * num_images : 3 * num_images,
                ]
            ),
            dtype,
        )

    return Rskp


@nb.njit(cache=True, nogil=True)
def skp_processing(
    *,
    mpmb_coherences: npt.NDArray[complex],
    subsampled_vertical_wavenumbers: npt.NDArray[float],
    azm: int,
    rng: int,
    coreg_primary_image_index: int,
    num_images: int,
    num_polarizations: int,
    spectra_z: npt.NDArray[float],
    volumetric_noise: npt.NDArray[complex],
    float_dtype: np.dtype,
    complex_dtype: np.dtype,
) -> tuple[int, int, npt.NDArray[complex], bool]:
    """
    Run the SKP processing for a selected pixel (i.e. azimuth and range).

    Parameters
    ----------
    mpmb_coherences: npt.NDArray[complex]
        The MPMB (subsampled) coherences packed as an array of shape
        [Npol*Nimg x Npol*Nimg x Nazm' x Nrng'].

    subsampled_vertical_wavenumbers: npt.NDArray[float] [rad/m]
        The vertical wavenumbers (1 per image) stacked as [Nimg x Nazm x Nrng].

    azm: int
        The selected azimuth index (in the subsampled image). Must be between
        0 and Nazm'.

    rng: int
        The selected range index (in thesubsampled image). Must be between
        0 and Nrng'.

    coreg_primary_image_index: int
        The index associated to the primary image used for coregistration.

    num_images: int
        The total number of images (i.e. 3 for INT or up to 7 for TOM).

    num_polarizations: int
        The total number of polarizations (i.e. 3 or 4).

    spectra_z: npt.NDArray[float]
        The possible Z spectra.

    volumetric_noise: npt.NDArray[complex]
        A cached volumetric noise to be used as fallback in case
        of a failed SKP decomposition. It must be of shape [Nimg x Nimg].

    float_dtype: np.dtype
        The floating point precision of the float values to be used
        during estimation.

    complex_dtype: np.dtype
        The floating point precision of the complex values to be used
        during estimation.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    azm: int
        The input azimuth index (forwarded).

    rng: int
        The input range index (forwarded).

    skp_phi_cal: npt.NDArray[float] [rad]
        The SKP calibration phases, stored as a [1 x Nimg] array.

    bool
        If the solution is usable or not (e.g. SVD failures).

    """
    # Compute the scattering covariances that will be associated to ground and
    # canopy/vegetation. We expect 4 matrices packed into an array.
    Rs, valid_skp_estimate = skp_estimate_scattering_coherences(
        mpmb_coherence=mpmb_coherences[..., azm, rng],
        num_images=num_images,
        num_polarizations=num_polarizations,
        fallback_noise=volumetric_noise,
        float_dtype=float_dtype,
        complex_dtype=complex_dtype,
    )
    if Rs.shape[0] != 4:
        raise SkpRuntimeError("SKP decomposition failed")

    # Estimating the "quality" of each scattering mechanism for calibration.
    qualities = np.empty((4, 1), dtype=float_dtype)
    phi_cal = np.empty((num_images, 4), dtype=float_dtype)

    errors = False
    # pylint: disable-next=not-an-iterable
    for k in nb.prange(4):
        if np.isnan(Rs[k]).any():
            qualities[k] = 0
            phi_cal[:, k] = 0
        else:
            (
                U,
                L,
                _,
            ) = np.linalg.svd(Rs[k])
            eta = np.sum(L)
            qualities[k] = (L[0] / eta) if eta != 0 else 0
            beta = U[coreg_primary_image_index, 0]
            phi_cal[:, k] = (
                # NOTE: Since numba-0.57, np.angle seems to have problems when
                # being assigned to a float matrix. We force casting to float
                # via np.real.
                np.real(np.angle(U[:, 0] / beta)) if beta != 0.0 else np.zeros(U[:, 0].shape, dtype=float_dtype)
            )
            if eta == 0.0 or beta == 0.0:
                errors = True

    # Steering matrix.
    # NOTE: We need to flatten first otherwise numba complains.
    curr_kz = np.reshape(subsampled_vertical_wavenumbers[:, azm, rng].flatten(), (num_images, 1))

    # Current calibration phases.
    A_cal = np.exp(1j * curr_kz * spectra_z).astype(complex_dtype) / num_images
    curr_phi_cal = phi_cal[:, np.argmax(qualities)]
    M_cal = np.diag(np.exp(-1j * curr_phi_cal).astype(complex_dtype))

    # Spectra estimation.
    spectra_aux = np.empty((spectra_z.size, 4), dtype=complex_dtype)
    # pylint: disable-next=not-an-iterable
    for k in nb.prange(4):
        Rs[k] = M_cal @ Rs[k] @ M_cal.conj().T
        spectra_aux[:, k] = np.diag(np.abs(A_cal.conj().T @ Rs[k] @ A_cal))

    # NOTE: Unfortunately numba does not support extra arguments for argmax.
    # We also need to cast spectra_aux to real, for the same reason:
    spectra_aux = np.real(spectra_aux)

    max_indices = np.empty((4,), dtype=np.int32)

    max_indices[0] = np.argmax(spectra_aux[:, 0])
    max_indices[1] = np.argmax(spectra_aux[:, 1])
    max_indices[2] = np.argmax(spectra_aux[:, 2])
    max_indices[3] = np.argmax(spectra_aux[:, 3])

    # Define the relative Z associated to the ground.
    ground_relative_z = spectra_z[np.min(max_indices)]

    # The calibration calibration phases.
    skp_phi_cal = curr_phi_cal + curr_kz.flatten() * ground_relative_z
    return (
        azm,
        rng,
        skp_phi_cal,
        errors and valid_skp_estimate,
    )


def compute_skp_calibration_phases_multithreaded(
    *,
    mpmb_coherences: npt.NDArray[complex],
    vertical_wavenumbers: npt.NDArray[float],
    coreg_primary_image_index: int,
    azimuth_estimation_indices: npt.NDArray[int],
    range_estimation_indices: npt.NDArray[int],
    num_images: int,
    num_polarizations: int,
    num_worker_threads: int,
    dtypes: EstimationDType,
) -> tuple[npt.NDArray[float], bool]:
    """
    Compute the SKP calibration phases.

    Parameters
    ----------
    mpmb_coherences: npt.NDArray[complex]
        The MPMB (subsampled) coherences packed as an array of shape
        [Npol*Nimg x Npol*Nimg x Nazm' x Nrng'].

    vertical_wavenumbers:
        The vertical wavenumbers (1 per image) stacked as [Nimg x Nazm' x Nrng'].

    coreg_primary_image_index: int
        The index of the coregistration primary image.

    azimuth_estimation_indices: npt.NDArray[int]
        Subsampling indices wrt to the full frame axis that corresponds to the
        pixels on which the SKP calibration phase is computed (azimuth direction).

    range_estimation_indices: npt.NDArray[int]
        Subsampling indices wrt to the full frame axis that corresponds to the
        pixels on which the SKP calibration phase is computed (range direction).

    num_images: int
        Number of frames in the stack.

    num_polarizations: int
        TNumber of polarizations in the stack.

    num_worker_threads: int
        Number of threads assigned to the task.

    dtypes: EstimationDType
        The estimation floating point accuracy.

    Raises
    ------
    SkpRuntimeError, ValueError

    Return
    ------
    skp_calibration_phases: npt.NDArray[float] [rad]
        The phase screens for each stack input frames and packed as
        a [Nimg x Nazm' x Nrng'] matrix.

    errors: bool
        As to whether errors have occurred during the estimation.

    """
    # Do minimal checks here.
    if num_polarizations * num_images != mpmb_coherences.shape[0]:
        raise SkpRuntimeError("MPMB-coherence and the stack have mismatching shape")
    if vertical_wavenumbers.shape[0] != num_images:
        raise SkpRuntimeError("Kz and stack have mismatching dimensions")
    if not (mpmb_coherences.shape[2] == vertical_wavenumbers.shape[1] == azimuth_estimation_indices.size) or not (
        mpmb_coherences.shape[3] == vertical_wavenumbers.shape[2] == range_estimation_indices.size
    ):
        raise SkpRuntimeError("SKP estimation windows, MPMB coherence and Kz have incompatible dimensions")

    # We cache some random volumetric noise to enable repeatabilty. This
    # also will speed up the runtime a bit. The memory and runtime overhead
    # is negligible.
    volumetric_noise_cache = cache_volume_noise(
        num_images,
        azimuth_estimation_indices.size,
        range_estimation_indices.size,
        dtypes=dtypes,
    )

    # The initialization for the spectra estimator.
    spectra_z = np.linspace(
        *Z_RANGE_DEFAULT,
        Z_RANGE_DEFAULT_SAMPLES,
        dtype=dtypes.float_dtype,
    )

    # The output SKP phase.
    skp_calibration_phases = np.empty(
        (
            azimuth_estimation_indices.size,
            range_estimation_indices.size,
            num_images,
        ),
        dtype=dtypes.float_dtype,
    )

    # Flag to store any error during the decomposition.
    errors = False

    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core routine for the SKP decomposition.
        def skp_processing_fn(pixel):
            return skp_processing(
                mpmb_coherences=mpmb_coherences,
                subsampled_vertical_wavenumbers=vertical_wavenumbers,
                azm=pixel[0],
                rng=pixel[1],
                coreg_primary_image_index=coreg_primary_image_index,
                num_images=num_images,
                num_polarizations=num_polarizations,
                spectra_z=spectra_z,
                volumetric_noise=volumetric_noise_cache[pixel[0], pixel[1]],
                float_dtype=dtypes.float_dtype,
                complex_dtype=dtypes.complex_dtype,
            )

        for azm, rng, phi, err in executor.map(
            skp_processing_fn,
            product(
                range(azimuth_estimation_indices.size),
                range(range_estimation_indices.size),
            ),
        ):
            skp_calibration_phases[azm, rng, ...] = phi
            errors = err or errors

    return (
        np.moveaxis(
            skp_calibration_phases,
            [0, 1, 2],
            [1, 2, 0],
        ),
        errors,
    )
