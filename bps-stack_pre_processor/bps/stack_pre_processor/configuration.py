# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Coregistration configuration structure
--------------------------------------------
"""

from dataclasses import dataclass

from bps.common.io import common


@dataclass
class PrimaryImageSelectionConf:
    """Primary image selection configuration."""

    primary_image_selection_information: common.PrimaryImageSelectionInformationType
    rfi_decorrelation_threshold: float
    faraday_decorrelation_threshold: float
