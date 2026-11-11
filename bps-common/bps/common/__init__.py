# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS common
----------
"""

__version__ = "4.6.0.dev0"

from bps.common.common import (
    STRIPMAP_SWATHS,
    AcquisitionMode,
    MissionPhaseID,
    Polarization,
    Swath,
    retrieve_aux_product_data_content,
    retrieve_aux_product_data_single_content,
)


def get_version_in_bps_format(version: str) -> str:
    """Convert a version to the BPS format.

    Parameters
    ----------
    version : str
        version in standard format (e.g. 2.3.5)

    Returns
    -------
    str
        version in BPS format (e.g. 02.35)
    """
    major, minor, patch = (int(x) for x in version.split(".")[0:3])
    if minor > 9 or patch > 9:
        raise RuntimeError(f"Minor and patch version cannot exceed '9': {version}")

    minor_bps = minor * 10 + patch
    assert minor_bps < 99
    return f"{major:02d}.{minor_bps:02d}"


BIOMASS_PROCESSOR_VERSION = get_version_in_bps_format(__version__)
"""Expected processor version for all BIOMASS CPF processor job orders"""

__all__ = [
    "__version__",
    "Swath",
    "AcquisitionMode",
    "MissionPhaseID",
    "Polarization",
    "retrieve_aux_product_data_content",
    "retrieve_aux_product_data_single_content",
    "STRIPMAP_SWATHS",
]
