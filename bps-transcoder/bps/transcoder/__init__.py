# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS Transcoder package
----------------------
"""

from typing import Literal

__version__ = "4.5.0.dev0"

BPS_PPD_VERSION = "V3.1.0"

BPS_AUX_FMT_VERSION = "V3.7.0"

BPS_ATBD_L1_VERSION = "V1.2.5"
BPS_ATBD_STA_VERSION = "V1.4.0"
BPS_ATBD_FH_VERSION = "V2.2.0"
BPS_ATBD_FD_VERSION = "V2.1.8"
BPS_ATBD_AGB_VERSION = "V3.1.4"


BPS_PFD_L1_VERSION = "V1.7"
BPS_PFD_FH_VERSION = "V3.5"
BPS_PFD_FD_VERSION = "V3.5"
BPS_PFD_AGB_VERSION = "V3.5"

ANNOTATION_VERSION = Literal["v450", "v445"]
LATEST_VERSION_STR: ANNOTATION_VERSION = "v450"

ANNOTATION_LEVEL = Literal["l1ab", "l1c", "l2a fh", "l2a fd", "l2a gn", "l2a tfh", "l2b fh", "l2b fd", "l2b agb"]
PFD_VERSION_MAPPING: dict[ANNOTATION_VERSION, dict[ANNOTATION_LEVEL, str]] = {
    "v450": {
        "l1ab": BPS_PFD_L1_VERSION,
        "l1c": BPS_PFD_L1_VERSION,
        "l2a fh": BPS_PFD_FH_VERSION,
        "l2a fd": BPS_PFD_FD_VERSION,
        "l2a gn": BPS_PFD_AGB_VERSION,
        "l2a tfh": BPS_PFD_FH_VERSION,
        "l2b fh": BPS_PFD_FH_VERSION,
        "l2b fd": BPS_PFD_FD_VERSION,
        "l2b agb": BPS_PFD_AGB_VERSION,
    },
    "v445": {
        "l1ab": "V1.6.1",
        "l1c": "V1.6.1",
        "l2a fh": "V3.4",
        "l2a fd": "V3.4",
        "l2a gn": "V3.4",
        "l2a tfh": "V3.4",
        "l2b fh": "V3.4",
        "l2b fd": "V3.4",
        "l2b agb": "V3.4",
    },
}
