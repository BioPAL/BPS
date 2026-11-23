# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Converts between polarization types
-----------------------------------
"""

from arepytools.io.metadata import EPolarization
from bps.common.io import common


class UnsupportedPolarizationError(ValueError):
    """Raised when input polarization does not map onto arepytools EPolarization."""


def translate_arepytools_polarization_to_common(polarization: EPolarization) -> common.PolarisationType:
    """Convert an ArePyTools EPolarization object to BPS PolarisationType"""
    if polarization == EPolarization.hh:
        return common.PolarisationType.HH
    if polarization == EPolarization.hv:
        return common.PolarisationType.HV
    if polarization == EPolarization.xx:
        return common.PolarisationType.XX
    if polarization == EPolarization.vh:
        return common.PolarisationType.VH
    if polarization == EPolarization.vv:
        return common.PolarisationType.VV

    raise UnsupportedPolarizationError(polarization)


def translate_common_polarization_to_arepytools(polarization: common.PolarisationType) -> EPolarization:
    if polarization == common.PolarisationType.HH:
        return EPolarization.hh
    if polarization == common.PolarisationType.HV:
        return EPolarization.hv
    if polarization == common.PolarisationType.XX:
        return EPolarization.xx
    if polarization == common.PolarisationType.VH:
        return EPolarization.vh
    if polarization == common.PolarisationType.VV:
        return EPolarization.vv

    raise UnsupportedPolarizationError(polarization)
