# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Library with geometrical utilities
----------------------------------
"""

import numpy as np
import numpy.typing as npt
import scipy as sp


def compute_vertical_wavenumbers(
    *,
    incidence_angles_p: npt.NDArray[float],
    incidence_angles_s: npt.NDArray[float],
    central_frequency: float,
) -> npt.NDArray[float]:
    r"""
    Compute the vertical wavenumbers as follows:

        Kz = 4 * \pi / \lambda * (\alpha{s} - \alpha{p}) / sin(\alpha{p}),

    where \lambda is the RADAR wavelength and \alpha is the incidence angle.
    The indices 'p' and 's' respectively represent the indices of the coregistration
    primary image and a secondary image.

    Parameters
    ----------
    incidence_angles_p: npt.NDArray[float] [rad]
        The [Nazm x Nrng] array containing the incidence angles of the coregistration
        primary image.

    incidence_angles_s: npt.NDArray[float] [rad]
        The [Nazm x Nrng] array containing the incidence angles of the secondary
        image.

    central_frequency: float [Hz]
        The RADAR central (carrier) frequency.

    Return
    ------
    npt.NDArray[float] [rad/m]
        The [Nazm x Nrng] array of vertical wavenumbers.

    """
    if central_frequency <= 0:
        raise ValueError(f"Negative {central_frequency=}")

    wavelength = sp.constants.speed_of_light / central_frequency
    return 4 * np.pi / wavelength * (incidence_angles_s - incidence_angles_p) / np.sin(incidence_angles_p)
