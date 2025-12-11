# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Split-Spectrum Method
-------------------------
"""

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.configuration import StackDataSpecs


def split_spectrum(
    *,
    f_c: float,
    f_l: float,
    f_h: float,
    phi_l: npt.NDArray[float],
    phi_h: npt.NDArray[float],
) -> npt.NDArray[float]:
    r"""
    The Split-Spectrum algorithm.

    The Split-Spectrum splits the interferograms into dispersive (D) and
    nondispersive (ND) components by leveraging the equalities:

      phi_D  = f_l * f_h * (phi_l * f_h - phi_h * f_l) / (f_c * (f_h^2 - f_l^2)),
      phi_ND = f_c * (phi_h * f_h - phi_l * f_l) / (f_h^2 - f_l^2).

    where:
      f_c is the central (carrier) frequency,
      f_l is the lower frequency (absolute),
      f_h is the upper frequency (absolute).

    Note that this function only returns the dispersive component.

    Parameters
    ----------
    f_c: float [Hz]
        The central (carrier) frequency.

    f_l: float [Hz]
        The lower frequency f_l (absolute).

    f_h: float [Hz]
        The upper frequency f_h (absolute).

    phi_l: npt.NDArray[float] [rad]
        The phases associated to the lower frequency/look.

    phi_h npt.NDArray[float] [rad]
        The phases associated to the upper frequency/look.

    Return
    ------
    npt.NDArray[float] [rad]
        The dispersive (slow-varying) ionospheric phase.

    """
    # The mid frequency.
    alpha = f_l / f_h
    beta = f_l * f_h**2 / (f_c * (f_h**2 - f_l**2))
    return (phi_l - alpha * phi_h) * beta


def compute_split_spectrum_biases(
    *,
    range_coreg_shifts: npt.NDArray[float],
    synth_phases: npt.NDArray[float],
    l1_iono_phases: npt.NDArray[float] | None,
    l1_iono_shifts: npt.NDArray[float] | None,
    stack_specs: StackDataSpecs,
) -> npt.NDArray[float]:
    """
    Compute the split-spectrum bias due to the coregisration method
    and the L1 ionospheric correction.

    Parameters
    ----------
    range_coreg_shifts: np.NDArray[float] [s]
        The [Nazm x Nrng] coregistration shifts.

    synth_phases: npt.NDArray[float] [rad]
        The [Nazm x Nrng] synthetic phases from the DEM.

    l1_iono_phases: Optional[npt.NDArray[float]] [rad]
        Optionally, the [Nazm x Nrng] ionospheric phase screen estimated
        by the L1 processor.

    l1_iono_shifts: Optional[npt.NDArray[float]] [m]
        Optionally, the [Nazm x Nrng] range shifts due to the ionosphere
        estimated and applied by the L1 processor.

    stack_specs: StackDataSpecs
        The stack specs object.

    Return
    ------
    npt.NDArray[float]
        The split-spectrum phase biases in [rad * s].

    """
    if l1_iono_phases is None:
        l1_iono_phases = np.zeros_like(range_coreg_shifts)
    if l1_iono_shifts is None:
        l1_iono_shifts = np.zeros_like(range_coreg_shifts)

    # First, we compute the term due to the coregistration:
    #
    #   bias_coreg := -4 * pi/c * rg_shift_sr_m
    #               = -2 * pi * rg_shifts_px * range_sampling_step.
    #
    bias = -(2 * np.pi * stack_specs.range_sampling_step) * range_coreg_shifts

    # Second, we encode the bias coming from the flattning phase:
    #
    #   bias_flat := - 1/fc + synth_phases.
    #
    bias += synth_phases / stack_specs.central_frequency

    # Third, we encode the bias term due to the L1 iono phase correction.
    #
    #   bias_coreg_L1a := 4 * pi/c * l1_iono_shifts_m.
    #
    bias += (4 * np.pi / sp.constants.speed_of_light) * l1_iono_shifts

    # Forth, we encode the bias term due to the L1 iono phase correction:
    #
    #   bias_phi_L1a := = -1/fc + l1_iono_phases
    #
    bias -= l1_iono_phases / stack_specs.central_frequency

    return bias
