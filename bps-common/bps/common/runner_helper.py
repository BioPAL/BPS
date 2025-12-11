# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Runner helper module
--------------------
"""

import logging
import subprocess
from typing import Any

from arepyextras.runner import Environment
from bps.common import bps_logger


def run_application(
    env: Environment,
    application: str,
    input_file: str,
    step: int,
    logger: logging.Logger = bps_logger._BPS_LOGGER,
):
    """Execute an application within an environment.

    Parameters
    ----------
    env : Environment
        The environment object
    application : str
        Application name
    input_file : str
        Input file location
    step : int
        Step index
    logger : logging.Logger, optional
        Logger object, by default it uses 'bps_logger'

    Raises
    ------
    RuntimeError
        Raised when the processing fails
    """
    run_application_args(env, application, [input_file, step], logger)


def run_application_args(
    env: Environment,
    application: str,
    args: list[Any],
    logger: logging.Logger = bps_logger._BPS_LOGGER,
):
    """Execute an application within an environment.

    Parameters
    ----------
    env : Environment
        The environment object
    application : str
        Application name
    args : List[Any]
        Command line arguments
    logger : logging.Logger, optional
        Logger object, by default it uses 'bps_logger'

    Raises
    ------
    RuntimeError
        Raised when the processing fails
    """

    try:
        popen_args, popen_kwargs = env.build_run_command_arguments(application, *args)
    except ValueError as exc:
        raise RuntimeError(f"Command '{application}' not found") from exc

    logger.info("%s started", application)

    with subprocess.Popen(popen_args, **popen_kwargs) as process:
        process.wait()
        returncode = process.returncode

    if returncode != 0:
        command_line = " ".join(popen_args)

        raise RuntimeError(f"Command failed with code {returncode}:\n" + f"'{command_line}'\n")
    logger.info("%s completed", application)
