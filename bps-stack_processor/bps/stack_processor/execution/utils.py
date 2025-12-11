# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Processor's Execution Utility Library
-------------------------------------------
"""

from pathlib import Path
from shutil import which

from arepyextras.runner import Environment
from bps.common import bps_logger
from bps.stack_processor import BPS_STACK_COREG_PROC_EXE_NAME


def setup_coreg_processor_env(working_dir: Path, num_omp_threads: int = 1) -> tuple[Environment, str]:
    """
    Utility to setup the BPS environment to execute the coregistration binary.

    Parameters
    ----------
    working_dir: Path
        Path to an existing working directory.

    num_omp_threads: int = 1
        Number of threads used by OpenMP. It defaults to 1.

    Raises
    ------
    NotADirectoryError, RuntimeError, ValueError

    Return
    ------
    coreg_env: Environment
        Object containing the binary's environment specs.

    coreg_exe: str
        Name of the coregistration binary.

    """
    if not working_dir.is_dir():
        raise NotADirectoryError(f"{working_dir}")
    if num_omp_threads < 1:
        raise ValueError(f"Got {num_omp_threads=}. It must be positive")

    if num_omp_threads > 1:
        bps_logger.warning(
            "Currently, only single-treaded execution of %s is supported",
            BPS_STACK_COREG_PROC_EXE_NAME,
        )

    coreg_env = Environment(working_dir)
    coreg_env.import_env_variable("OMP_NUM_THREADS", default=num_omp_threads)
    coreg_env.import_env_variable("PATH", is_list=True)
    coreg_env.import_env_variable("LD_LIBRARY_PATH", is_list=True)
    bps_logger.debug("STA_P's coreg binary environment: %s", coreg_env.__dict__)

    coreg_exe = BPS_STACK_COREG_PROC_EXE_NAME
    bps_logger.debug("STA_P's coreg binary: %s", coreg_exe)
    if which(coreg_exe) is None:
        raise RuntimeError(f"{coreg_exe} is not available/installed")

    return coreg_env, coreg_exe
