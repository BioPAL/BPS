# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS processors initialization
----------
"""

import os
from contextlib import contextmanager
from pathlib import Path

from bps.common import bps_logger

DEFAULT_INTERMEDIATE_DATA_DIR_NAME = "intermediate_data_dir"


def get_intermediate_data_dir_name(job_order_file: Path, add_data_dir: bool) -> Path:
    """Returns the intermediate data dir:
    - the directory containing the joborder if add_data_dir is False
    - a directory inside the joborder if add_data_dir is True
    """
    return job_order_file.parent.joinpath(DEFAULT_INTERMEDIATE_DATA_DIR_NAME) if add_data_dir else job_order_file.parent


def processor_setup(
    job_order_file: str | Path,
    working_dir: str | Path | None,
    processor_name: str,
    processor_tag: str,
    processor_version: str,
    add_data_dir: bool = False,
) -> tuple[Path, Path]:
    """Initial setup

    Parameters
    ----------
    job_order_file : Union[str, Path]
        path to the joborder file
    working_dir : Optional[Union[str, Path]]
        working directory, if None will be set to JobOrder folder
    processor_name : str
        name of the processor
    processor_tag : str
        tag of the processor
    processor_version : str
        version of the processor

    Returns
    -------
    tuple[Path, Path]
        working directory path and job order path
    """
    job_order_path = Path(job_order_file).absolute()

    working_dir = working_dir or get_intermediate_data_dir_name(job_order_path, add_data_dir)
    working_dir_path = Path(working_dir).absolute()
    working_dir_path.mkdir(exist_ok=True, parents=True)

    bps_logger.init_logger(processor=processor_tag, task=processor_tag, version=processor_version)
    bps_logger.enable_console_logging()
    bps_logger.enable_file_logging(working_dir_path)

    bps_logger.info("%s started", processor_name)
    bps_logger.info("Working directory: %s", working_dir_path)
    bps_logger.info("Job order file: %s", job_order_path)

    return working_dir_path, job_order_path


@contextmanager
def working_directory(path: Path):
    """Run code in path, restore previous dir on exit

    Parameters
    ----------
    path : Path
        working directory
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
