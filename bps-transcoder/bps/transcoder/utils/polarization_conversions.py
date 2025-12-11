# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Converts Betweem Polarization Types
-----------------------------------
"""

from typing import Literal

from arepytools.io.metadata import EPolarization
from bps.common.io.common_types import PolarisationType


class UnsupportedPolarizationError(ValueError):
    """Raised when input polarization does not map onto arepytools EPolarization."""


def translate_polarization(polarization: EPolarization) -> PolarisationType:
    """
    Convert an ArePyTools EPolarization object to BPS PolarisationType

    Arguments
    ---------
        polarization: Epolarization
            The input polarization instance.

    Throw
    -----
        UnsupportedPolarizationError

    Return
    ------
        PolarisationType
            The converted polarization instance.
    """
    if polarization == EPolarization.hh:
        return PolarisationType.HH
    if polarization == EPolarization.hv:
        return PolarisationType.HV
    if polarization == EPolarization.xx:
        return PolarisationType.XX
    if polarization == EPolarization.vh:
        return PolarisationType.VH
    if polarization == EPolarization.vv:
        return PolarisationType.VV

    raise UnsupportedPolarizationError(polarization)


def translate_polarization_tag(
    poltag: str,
    poltype: Literal["arepytools", "bps"],
) -> EPolarization | PolarisationType:
    """
    Parse the polarization tag (e.g. "H/H") to a polarization object.

    Arguments
    ---------
        poltag: str
           The polarization tag.

        poltype: Literal["arepytools", "bps"]
            "arepytools": EPolarization,
            "bps": PolarisationType.

    Throw
    -----
        ValueError, UnsupportedPolarizationError

    Return
    ------
        Union[EPolarization, PolarisationType]
            The parse polarization object.
    """
    if poltype not in {"arepytools", "bps"}:
        raise ValueError(f"'{poltype}' must be either 'arepytools' or 'bps'")

    std_tag = poltag.lower().replace("/", "")
    if std_tag == "hh":
        return EPolarization.hh if poltype == "arepytools" else PolarisationType.HH
    if std_tag == "hv":
        return EPolarization.hv if poltype == "arepytools" else PolarisationType.HV
    if std_tag == "xx":
        return EPolarization.xx if poltype == "arepytools" else PolarisationType.XX
    if std_tag == "vh":
        return EPolarization.vh if poltype == "arepytools" else PolarisationType.VH
    if std_tag == "vv":
        return EPolarization.vv if poltype == "arepytools" else PolarisationType.VV

    raise UnsupportedPolarizationError(poltag)
