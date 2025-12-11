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

from dataclasses import dataclass

from bps.common import bps_logger
from bps.common.joborder import DeviceResources


def _is_resource_enough(required_mb: int | None, provided_mb: int, tag: str) -> bool:
    if required_mb is not None:
        if provided_mb < required_mb:
            bps_logger.error(f"Available {tag} ({provided_mb} [MB]) lower than required ({required_mb} [MB])")
            return False
    return True


@dataclass
class RequiredResources:
    """Resources required to meet target performance"""

    suggested_cpu_cores: int | None
    ram_size_mb: int | None
    ramdisk_size_mb: int | None
    disk_size_mb: int | None

    def _check_suggested_cpu_cores(self, provided_cores: int):
        if self.suggested_cpu_cores is not None:
            if provided_cores < self.suggested_cpu_cores:
                bps_logger.warning(
                    f"Available threads ({provided_cores}) lower than "
                    + f"those required to meet target performances ({self.suggested_cpu_cores})"
                )

    def _is_ram_size_enough(self, provided_ram_size_mb: int) -> bool:
        return _is_resource_enough(required_mb=self.ram_size_mb, provided_mb=provided_ram_size_mb, tag="RAM")

    def _is_ramdisk_size_enough(self, provided_ramdisk_size_mb: int) -> bool:
        return _is_resource_enough(
            required_mb=self.ramdisk_size_mb,
            provided_mb=provided_ramdisk_size_mb,
            tag="RAMDISK",
        )

    def _is_disk_size_enough(self, provided_disk_size_mb: int) -> bool:
        return _is_resource_enough(
            required_mb=self.disk_size_mb,
            provided_mb=provided_disk_size_mb,
            tag="disk space",
        )

    def are_provided_resources_enough(self, provided_resources: DeviceResources) -> bool:
        """Wether there are enough resources"""

        self._check_suggested_cpu_cores(provided_resources.num_threads)

        ram_is_enough = self._is_ram_size_enough(provided_resources.available_ram)

        ramdisk_is_enough = (
            self._is_ramdisk_size_enough(provided_resources.ramdisk_size)
            if provided_resources.ramdisk_size is not None
            else True
        )

        disk_space_is_enough = self._is_disk_size_enough(provided_resources.available_space)

        return ram_is_enough and ramdisk_is_enough and disk_space_is_enough


def raise_if_resources_are_not_enough(required_resources: RequiredResources, provided_resources: DeviceResources):
    """Raise RunTimeError when device resources are not enough for processing"""
    if not required_resources.are_provided_resources_enough(provided_resources):
        raise RuntimeError("Cannot proceed, available hardware resources are not enough")
