# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Constraints on allocated resources
----------------------------------
"""

from functools import partial

from bps.common.device_resources_constraints import (
    RequiredResources,
    raise_if_resources_are_not_enough,
)

L1_REQUIRED_RESOURCES = RequiredResources(
    suggested_cpu_cores=6,
    ram_size_mb=10240,
    ramdisk_size_mb=15360,
    disk_size_mb=6144,
)

raise_if_resources_are_not_enough_for_l1_processing = partial(
    raise_if_resources_are_not_enough, required_resources=L1_REQUIRED_RESOURCES
)
