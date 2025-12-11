# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utility to Handle the Working Directory
---------------------------------------
"""

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from bps.common import bps_logger
from bps.stack_processor.execution.stack_coreg_execution_manager import (
    STOP_AND_RESUME_PATH as COREG_STOP_AND_RESUME_PATH,
)
from bps.stack_processor.execution.stack_preproc_execution_manager import (
    STOP_AND_RESUME_PATH as PRE_PROC_STOP_AND_RESUME_PATH,
)
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
    validate_aux_pps_xsd_schema,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.parsing import (
    parse_aux_pps,
    parse_stack_job_order,
)
from bps.stack_processor.interface.internal.intermediates import (
    STACK_CAL_PROC_INTERMEDIATE_FOLDER,
    STACK_COREG_PROC_INTERMEDIATE_FOLDER,
    STACK_PRE_PROC_INTERMEDIATE_FOLDER,
)

# Previous configuration, used to check the validity of an already existing
# working directory.
JOB_ORDER_PREVIOUS_FILE_NAME = "job_order_previous.xml"
AUX_PPS_PREVIOUS_FILE_NAME = "aux_pps_previous.xml"


class StackWorkingDirParsingError(RuntimeError):
    """Raised when a previous file in current working directory does not parse."""


class StackWorkingDirError(RuntimeError):
    """Raised when the current working directory is invalid."""


def raise_if_aux_pps_is_not_compatible_for_resume(
    current_aux_pps: AuxiliaryStaprocessingParameters,
    working_dir: Path,
):
    """
    Check that the current AUX-PPS file is compatible with one possibly saved
    in the working directory.

    Parameters
    ----------
    current_aux_pps: AuxiliaryStaprocessingParameters
        Object of the current AUX-PPS.

    working_dir: Path
        Path to the working directory.

    Raises
    ------
    StackWorkingDirParsingError, StackWorkingDirError

    """
    # If a previous AUX-PPS exists, compare its content to the current one.
    previous_aux_pps_path = working_dir / AUX_PPS_PREVIOUS_FILE_NAME
    if previous_aux_pps_path.is_file():
        # Parse the previuos AUX-PPS, raise if it does not parse.
        try:
            validate_aux_pps_xsd_schema(previous_aux_pps_path)
            previous_aux_pps = parse_aux_pps(previous_aux_pps_path.read_text(encoding="utf-8"))
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:
            raise StackWorkingDirParsingError(f"Stored AUX-PPS {previous_aux_pps_path} is no longer valid.") from exc

        # Check compatibility between AUX-PPSs.
        errors = []
        if current_aux_pps.general != previous_aux_pps.general:
            errors.append("general configurations")

        if current_aux_pps.primary_image_selection != previous_aux_pps.primary_image_selection:
            errors.append("primary image selections")

        if current_aux_pps.coregistration != previous_aux_pps.coregistration:
            errors.append("coregistration settings")

        if current_aux_pps.rfi_degradation_estimation != previous_aux_pps.rfi_degradation_estimation:
            errors.append("RFI degradation estimations")

        if len(errors) > 0:
            raise StackWorkingDirError(
                "Cannot resume STA_P. Current AUX-PPS not compatible with previous one. "
                "Found compatibility issues: {}".format(errors)
            )

    # Check that it is a fresh run and not a corrupted working directory.
    else:
        if (working_dir / PRE_PROC_STOP_AND_RESUME_PATH).is_file() or (
            working_dir / COREG_STOP_AND_RESUME_PATH
        ).is_file():
            raise StackWorkingDirError(
                "Woring directory not as expected. Breakpoint completion file found but no previous AUX-PPS"
            )


def raise_if_job_order_is_not_compatible_for_resume(
    current_job_order: StackJobOrder,
    working_dir: Path,
):
    """
    Check that the job order file is compatible with the one possibly already saved
    in the working directory.

    Parameters
    ----------
    current_job_order: StackJobOrder
        Object of the current job order.

    working_dir: Path
        Path to the working directory.

    Raises
    ------
    StackWorkingDirParsingError, StackWorkingDirError

    """
    # If a previous job_order exists, compare its content to the current one.
    previous_job_order_path = working_dir / JOB_ORDER_PREVIOUS_FILE_NAME
    if previous_job_order_path.is_file():
        # Parse the previuos job order, raise if it does not parse.
        try:
            previous_job_order = parse_stack_job_order(previous_job_order_path.read_text(encoding="utf-8"))
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:
            raise StackWorkingDirParsingError(
                f"Stored job order {previous_job_order_path} is no longer valid."
            ) from exc

        # Check compatibility between job orders.
        errors = []
        if not _same_stack(current_job_order.input_stack, previous_job_order.input_stack):
            errors.append("input L1a stack")
        if not _same_optional_file(
            current_job_order.processing_parameters.primary_image,
            previous_job_order.processing_parameters.primary_image,
        ):
            errors.append("coregistration primary images")
        if current_job_order.intermediate_files != previous_job_order.intermediate_files:
            errors.append("intermediate files")

        delta_azimuth_interval = np.array(
            _value_or_zero(current_job_order.processor_configuration.azimuth_interval)
        ) - np.array(_value_or_zero(previous_job_order.processor_configuration.azimuth_interval))
        if not np.all(np.isclose(delta_azimuth_interval.astype(np.float64), 0.0)):
            errors.append("azimuth TOI")

        if len(errors) > 0:
            raise StackWorkingDirError(
                "Cannot resume STA_P. Current job order not compatible with previous one. "
                "Found compatibility issues: {}".format(errors)
            )
    else:
        stack_pre_proc_stop_and_resume_path = (
            working_dir / STACK_PRE_PROC_INTERMEDIATE_FOLDER / PRE_PROC_STOP_AND_RESUME_PATH
        )
        stack_coreg_proc_stop_and_resume_path = (
            working_dir / STACK_COREG_PROC_INTERMEDIATE_FOLDER / COREG_STOP_AND_RESUME_PATH
        )
        stack_cal_proc_stop_and_resume_path = (
            working_dir / STACK_CAL_PROC_INTERMEDIATE_FOLDER / COREG_STOP_AND_RESUME_PATH
        )
        if (
            stack_pre_proc_stop_and_resume_path.exists()
            or stack_coreg_proc_stop_and_resume_path.exists()
            or stack_cal_proc_stop_and_resume_path.exists()
        ):
            raise StackWorkingDirError(
                "Woring directory not as expected. Breakpoint completion file found but no previous job order"
            )


def store_current_aux_pps(current_aux_pps_path: Path, working_dir: Path):
    """
    Store the current aux_pps for next run.

    Parameters
    ----------
    current_aux_pps_path: Path
        Path to the current AUX-PPS.

    working_dir: Path
        Path to the working directory.
    """

    previous_aux_pps_path = working_dir / AUX_PPS_PREVIOUS_FILE_NAME
    # Create or overwrite the previous AUX-PPS with the content of the AUX-PPS from this run.
    shutil.copyfile(
        current_aux_pps_path,
        previous_aux_pps_path,
    )
    if not previous_aux_pps_path.is_file():
        bps_logger.warning(f"Could not copy current AUX-PPS into {previous_aux_pps_path}. Resume will not be possible")
    else:
        bps_logger.info(f"Copied current AUX-PPS into {previous_aux_pps_path}")


def store_current_job_order(current_job_order_path: Path, working_dir: Path):
    """
    Store the current job order for next run.

    Parameters
    ----------
    current_job_order_path: Path
        Path to the current job order.

    working_dir: Path
        Path to the working directory.
    """

    previous_job_order_path = working_dir / JOB_ORDER_PREVIOUS_FILE_NAME
    # Create or overwrite the previous job order with the content of the job order from this run.
    shutil.copyfile(
        current_job_order_path,
        previous_job_order_path,
    )
    if not previous_job_order_path.is_file():
        bps_logger.warning(
            f"Could not copy current job order into {previous_job_order_path}. Resume will not be possible"
        )
    else:
        bps_logger.info(f"Copied current job order into {previous_job_order_path}")


def remove_intermediate_outputs(
    *,
    stack_pre_proc_brk_dir: Path | None = None,
    stack_coreg_proc_brk_dir: Path | None = None,
    stack_cal_proc_brk_dir: Path | None = None,
    stack_working_dir: Path | None = None,
):
    """
    Remove the STA_P by-products and intermediates.

    Parameters
    ----------
    stack_pre_proc_working_dir: Path | None = None
        Optionally, remove the STA_P pre-processor's breakpoint directory.

    stack_coreg_proc_working_dir: Path | None = None
        Optionally, remove the STA_P coreg-processor's breakpoint directory.

    stack_cal_proc_working_dir: Path | None = None
        Optionally, remove the STA_P cal-processor's breakpoint directory.

    stack_working_dir: Path | None = None
        If provided, remove all the content of the working directory except
        for the log file.

    """
    if stack_pre_proc_brk_dir is not None:
        _remove_content(stack_pre_proc_brk_dir, keep_files=[PRE_PROC_STOP_AND_RESUME_PATH])
    if stack_coreg_proc_brk_dir is not None:
        _remove_content(stack_coreg_proc_brk_dir, keep_files=[COREG_STOP_AND_RESUME_PATH])
    if stack_cal_proc_brk_dir is not None:
        _remove_content(stack_cal_proc_brk_dir, keep_files=[COREG_STOP_AND_RESUME_PATH])
    if stack_working_dir is not None:
        shutil.rmtree(stack_working_dir / STACK_PRE_PROC_INTERMEDIATE_FOLDER, ignore_errors=True)
        shutil.rmtree(stack_working_dir / STACK_COREG_PROC_INTERMEDIATE_FOLDER, ignore_errors=True)
        shutil.rmtree(stack_working_dir / STACK_CAL_PROC_INTERMEDIATE_FOLDER, ignore_errors=True)
        (stack_working_dir / JOB_ORDER_PREVIOUS_FILE_NAME).unlink()
        (stack_working_dir / AUX_PPS_PREVIOUS_FILE_NAME).unlink()


def _same_stack(
    input_stack_p: list[Path],
    input_stack_s: list[Path],
) -> bool:
    """Compare two input stack and check that they are the same."""
    return len(input_stack_p) == len(input_stack_s) and all(
        pp.samefile(ps) for pp, ps in zip(input_stack_p, input_stack_s)
    )


def _same_optional_file(file_p: Path | None, file_s: Path | None) -> bool:
    """Check that 2 optional files are identical. Both None are identical."""
    if file_p is None and file_s is None:
        return True
    if file_p is not None and file_s is not None:
        return file_s.samefile(file_p)
    return False


def _value_or_zero(opt: Any | None) -> Any | float:
    """Return opt if not None else 0.0"""
    return opt if opt is not None else 0.0


def _remove_content(root_dir: Path | str, keep_files: list[str]):
    """Remove the content of directory except files matching regexes."""
    for path in Path(root_dir).glob("*"):
        if any(re.search(regex, path.name) is not None for regex in keep_files):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
