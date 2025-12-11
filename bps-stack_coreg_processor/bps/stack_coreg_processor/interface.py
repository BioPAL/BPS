# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Coregistration Processor (Interface Module)
-------------------------------------------------
"""

import json
from dataclasses import dataclass
from pathlib import Path

import xmltodict
from bps.common import bps_logger
from bps.common.io.common_types.models import CoregistrationMethodType
from bps.stack_coreg_processor.configuration import (
    CoregStackProcessorInternalConfiguration,
)
from bps.stack_coreg_processor.input_file import BPSCoregProcessorInputFile
from bps.stack_coreg_processor.serialization import (
    serialize_coreg_config_file,
    serialize_coreg_input_file,
)
from bps.stack_coreg_processor.utils import StackCoregProcessorRuntimeError

# Map from BPSStackProcessor coregistration method (i.e. "AUTOMATIC",
# "FULL_ACCURACY", and "GEOMETRY") to L1c product coregistration methods
# (i.e. "AUTOMATIC", "GEOMETRY", and "GEOMETRY_AND_DATA")
COREG_CONF_TO_L1C_CONF = {
    CoregStackProcessorInternalConfiguration.CoregMode.AUTOMATIC: CoregistrationMethodType.AUTOMATIC,
    CoregStackProcessorInternalConfiguration.CoregMode.FULL_ACCURACY: CoregistrationMethodType.GEOMETRY_AND_DATA,
    CoregStackProcessorInternalConfiguration.CoregMode.GEOMETRY: CoregistrationMethodType.GEOMETRY,
}


class CoregStackProcessorConfError(ValueError):
    """Handle wrong configurations for the BPSStackProcessor."""


def write_coreg_input_file(input_file: BPSCoregProcessorInputFile, output_path: Path):
    """
    Write Stack Coregistration Processor input file.

    Parameters
    ----------
    input_file: BPSCoregProcessorInputFile
        The input file object.

    output_path: Path
        Path to the output file.

    Raises
    ------
    OSError
        If writing to disk failed.

    """
    output_path.write_text(serialize_coreg_input_file(input_file), encoding="utf-8")
    if not output_path.exists():
        raise OSError(f"Writing to {output_path} failed")


def write_coreg_configuration_file(conf: CoregStackProcessorInternalConfiguration, output_path: Path):
    """
    Write the BPSStackProcessor configuration file.

    Parameters
    ----------
    conf: CoregStackProcessorInternalConfiguration
        The configuration object.

    output_path: Path
        Path to the output file.

    Raises
    ------
    OSError
        If writing to disk failed.

    """
    output_path.write_text(serialize_coreg_config_file(conf), encoding="utf-8")
    if not output_path.exists():
        raise OSError(f"Writing to {output_path} failed")


@dataclass
class StackCoregProcInterfaceFiles:
    """Stack Coreg Processor interface files."""

    base_dir: Path
    coreg_config_file: Path
    coreg_primary_config_file: Path

    def input_file(self, index: int) -> Path:
        """
        The input file for a coregistration job.

        Parameters
        ----------
        index: int
             A unique index that identifies the job.

        Return
        ------
        Path
            The path of the input file.

        """
        return self.base_dir.joinpath(f"stackInputFile_STA_P{index}.xml")

    @classmethod
    def from_base_dir(cls, base_dir: Path):
        """
        Setup the paths of the StackCoreg interface products.

        Parameters
        ----------
        base_dir: Path
            Directory where the StackCoreg interface files are saved.

        Return
        ------
        StackCoregProcInterfaceFiles
            Struct with the default file paths.

        """
        return cls(
            base_dir=base_dir,
            coreg_config_file=base_dir / "stackCoregConfig.xml",
            coreg_primary_config_file=base_dir / "stackCoregPrimaryConfig.xml",
        )


def write_stop_and_resume_file(
    output_path: Path,
    succeeded_mask: tuple[bool, ...],
    needs_resume: bool,
):
    """
    Write the stop-and-resume status file.

    Parameters
    ----------
    output_path: Path
        The destination path of the status file.

    succeded_mask: tuple[bool, ...]
        A [1 x Nimg] boolean mask that specifies which image coregistration
        succeeded (true) or failed (false).

    needs_resume: bool
        A boolean flag that specifies if the whole coregistration task
        needs be resumed or it can be considered completed.

    Raises
    ------
    StackCoregProcessorRuntimeError

    """
    try:
        with output_path.open(mode="w", encoding="utf-8") as f:
            json.dump(
                {"succeeded_mask": succeeded_mask, "needs_resume": needs_resume},
                f,
                indent=2,
            )
        if not output_path.exists():
            bps_logger.warning("Could not export stack's state file. Resuming STA_P will not be possible")
        # pylint: disable-next=broad-exception-caught
    except Exception as err:
        bps_logger.error(
            "Error occurred while writing stack's coreg state file %s to working directory",
            output_path,
        )
        raise StackCoregProcessorRuntimeError(err) from err


def load_stop_and_resume_file(
    stop_and_resume_path: Path,
) -> tuple[tuple[bool, ...], bool]:
    """
    Load the stop-and-resume status file.

    Parameters
    ----------
    stop_and_resume_path: Path
        Path to the stop-and-resume status file.

    Raises
    ------
    StackCoregProcessorRuntimeError

    Return
    ------
    succeded_mask: tuple[bool, ...]
        A [1 x Nimg] boolean mask that specifies which image coregistration
        succeeded (true) or failed (false).

    needs_resume: bool
        A boolean flag that specifies if the whole coregistration task
        needs be resumed or it can be considered completed.

    """
    try:
        with stop_and_resume_path.open(mode="r", encoding="utf-8") as f:
            execution_summary = json.load(f)
            return (
                execution_summary["succeeded_mask"],
                execution_summary["needs_resume"],
            )
    except KeyError as err:
        bps_logger.error(
            "dictionary key error occurred while loading stack's pre-proc state file %s from working directory",
            stop_and_resume_path,
        )
        raise StackCoregProcessorRuntimeError(err) from err
        # pylint: disable-next=broad-exception-caught
    except Exception as err:
        bps_logger.error(
            "Error occurred while loading stack's pre-proc state file %s from working directory",
            stop_and_resume_path,
        )
        raise StackCoregProcessorRuntimeError(err) from err


def load_actualized_coregistration_parameters_raise_if_invalid(
    actualized_param_path: Path,
) -> dict:
    """
    Parse the actualizedCoregistrationParameters.xml file.

    Parameters
    ----------
    actualized_param_path: Path
        Path to the actualized parameters file.

    Raises
    ------
    StackCoregProcessorRuntimeError

    Return
    ------
    dict:
        The parameter pact as a dictionary.

    """
    if not actualized_param_path.is_file():
        raise StackCoregProcessorRuntimeError(f"Cannot find {actualized_param_path}")

    # The expected tags in the parameters.
    expected_tags = (
        "ActualizedCoregistrationMode",
        "R2",
        "R2Range",
        "R2Azimuth",
        "ValidBlocksPercentage",
        "CommonBandAverageBandwidthPercentage",
    )

    coreg_params = xmltodict.parse(actualized_param_path.read_text())
    if "CoregistrationParams" not in coreg_params:
        raise StackCoregProcessorRuntimeError(f"{actualized_param_path} misses 'CoregistrationParams'")
    coreg_params = coreg_params["CoregistrationParams"]

    missing_tags = tuple(t in coreg_params for t in expected_tags if t not in coreg_params)
    if len(missing_tags) > 0:
        raise StackCoregProcessorRuntimeError(f"{actualized_param_path} misses {missing_tags}")

    return {
        "coregistration_method": COREG_CONF_TO_L1C_CONF[
            CoregStackProcessorInternalConfiguration.CoregMode(coreg_params["ActualizedCoregistrationMode"])
        ],
        "quality_coregistration_azimuth": float(coreg_params["R2Azimuth"]),
        "quality_coregistration_range": float(coreg_params["R2Range"]),
        "quality_coregistration": float(coreg_params["R2"]),
        "valid_blocks_ratio": 0.01 * float(coreg_params["ValidBlocksPercentage"]),
        "common_band_average_bandwidth_ratio": 0.01 * float(coreg_params["CommonBandAverageBandwidthPercentage"]),
    }
