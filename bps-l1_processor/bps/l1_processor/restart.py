# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Restart module
--------------
"""

import shutil
from contextlib import contextmanager
from pathlib import Path

from bps.common import bps_logger


def _move(src_dir: Path, dest_dir: Path):
    for item in src_dir.iterdir():
        item_rel = item.relative_to(src_dir)
        dest = dest_dir.joinpath(item_rel)
        if item.is_file():
            shutil.move(str(item), str(dest))
        else:
            dest.mkdir(exist_ok=True)
            _move(item, dest)

    remaining_content = list(src_dir.iterdir())
    if len(remaining_content) != 0:
        raise RuntimeError(f"Unexpected content left in recovery dir: {remaining_content}")
    src_dir.rmdir()


def load_recovery_data(recovery_dir: Path, working_dir: Path) -> None:
    """Load recovery data to the working folder"""
    _move(recovery_dir, working_dir)

    bps_logger.info("Recovery data correctly loaded")


@contextmanager
def optional_data_recovery(recovery_dir: Path | None, working_dir: Path):
    """Optional data recovery system

    Parameters
    ----------
    recovery_dir : Optional[Path]
        if provided it will load/save recovery data from the folder
    working_dir : Path
        current working directory
    """
    if recovery_dir is None:
        yield
    else:
        if recovery_dir.exists():
            bps_logger.info("Recovery folder found: %s", recovery_dir)
            load_recovery_data(recovery_dir, working_dir)
            bps_logger.info("Recovery folder moved to working_dir: %s and deleted", working_dir)
        assert not recovery_dir.exists()

        try:
            yield
        finally:
            shutil.copytree(str(working_dir), str(recovery_dir))
            original_log_file = bps_logger.get_log_file()
            assert original_log_file is not None
            bps_logger.add_file_handler(recovery_dir.joinpath(original_log_file.name))
            bps_logger.info(
                "Working directory %s saved to recovery folder: %s ",
                working_dir,
                recovery_dir,
            )
