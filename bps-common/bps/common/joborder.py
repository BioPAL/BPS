# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Job Order structures common to multiple processing levels
---------------------------------------------------------
"""

import enum
from dataclasses import dataclass
from pathlib import Path

from arepytools.timing.precisedatetime import PreciseDateTime


@dataclass
class ProcessorConfiguration:
    """Processor configuration"""

    class LogLevel(enum.Enum):
        """logging levels"""

        ERROR = "ERROR"
        WARNING = "WARNING"
        PROGRESS = "PROGRESS"
        INFO = "INFO"
        DEBUG = "DEBUG"

    file_class: str
    """File class of the product to be generated"""

    stdout_log_level: LogLevel
    """Log level of stdout"""

    stderr_log_level: LogLevel
    """Log level of stderr"""

    keep_intermediate: bool
    """Wether to keep intermediate files"""

    azimuth_interval: tuple[PreciseDateTime, PreciseDateTime] | None = None
    """Time of interest of the processing request - start, stop [Utc]"""


@dataclass
class DeviceResources:
    """Available device resources"""

    num_threads: int
    """Available threads"""

    available_ram: int
    """Available RAM [MegaByte]"""

    available_space: int
    """Available disk space [MegaByte]"""

    ramdisk_size: int | None = None
    """Available disk space on the ramdisk [MegaByte]"""

    ramdisk_mount_point: Path | None = None
    """Mount point of the ram disk"""
