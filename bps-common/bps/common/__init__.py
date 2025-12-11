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

__version__ = "4.2.2"

from bps.common.common import (
    STRIPMAP_SWATHS,
    AcquisitionMode,
    MissionPhaseID,
    Polarization,
    Swath,
    retrieve_aux_product_data_content,
    retrieve_aux_product_data_single_content,
)

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
