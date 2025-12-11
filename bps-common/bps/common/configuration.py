# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS configuration
-----------------
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import bps.common.io.aresys_configuration_models as are_conf
from bps.common.io.parsing import serialize
from bps.common.joborder import ProcessorConfiguration


@dataclass
class BPSConfiguration:
    """BPS binaries high level configuration"""

    class LogLevel(Enum):
        """logging levels"""

        ERROR = "ERROR"
        WARNING = "WARNING"
        PROGRESS = "PROGRESS"
        INFO = "INFO"
        DEBUG = "DEBUG"

    stdout_log_level: LogLevel
    """Log level of stdout"""

    stderr_log_level: LogLevel
    """Log level of stderr"""

    node_name: str
    """Name of the processing node"""

    processor_name: str
    """Processor name"""

    processor_version: str
    """Processor version"""

    task_name: str
    """Task name"""


def fill_bps_configuration_file(
    processor_configuration: ProcessorConfiguration,
    *,
    node_name: str,
    processor_name: str,
    processor_version: str,
    task_name: str,
) -> BPSConfiguration:
    """Fill BPS configuration from JobOrder information"""
    return BPSConfiguration(
        stdout_log_level=BPSConfiguration.LogLevel(processor_configuration.stdout_log_level.value),
        stderr_log_level=BPSConfiguration.LogLevel(processor_configuration.stderr_log_level.value),
        node_name=node_name,
        processor_name=processor_name,
        processor_version=processor_version,
        task_name=task_name,
    )


def translate_configuration_to_model(
    configuration: BPSConfiguration,
) -> are_conf.AresysXmlDoc:
    """Translate BPS configuration to the corresponding XSD model"""
    logger_conf = are_conf.BpsloggerConfType(
        node_name=configuration.node_name,
        processor_name=configuration.processor_name,
        processor_version=configuration.processor_version,
        task_name=configuration.task_name,
        std_out_log_level=are_conf.Bpsloglevels(configuration.stdout_log_level.value),
        std_err_log_level=are_conf.Bpsloglevels(configuration.stderr_log_level.value),
    )
    bps_conf = are_conf.BpsconfType(bpslogger_conf=logger_conf)
    return are_conf.AresysXmlDoc(
        number_of_channels=1,
        version_number=2.6,
        description="BPS configuration",
        channel=[are_conf.AresysXmlDocType.Channel(bpsconf=[bps_conf], number=1, total=1)],
    )


def serialize_bps_configuration_file(
    configuration: BPSConfiguration,
) -> str:
    """Serialize BPSConfiguration to XML"""
    model = translate_configuration_to_model(configuration)
    return serialize(model)


def write_bps_configuration_file(configuration: BPSConfiguration, file: Path):
    """Write BPS configuration file"""
    file.write_text(serialize_bps_configuration_file(configuration), encoding="utf-8")
