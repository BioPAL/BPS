# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to handle BPS L1 processor status
-------------------------------------------
"""

from pathlib import Path

from bps.l1_processor.status_system.status import BPSL1ProcessorStatus
from bps.l1_processor.status_system.translate_status import (
    dump_json,
    load_json,
    translate_bps_l1_processor_status_to_model,
    translate_model_to_bps_l1_processor_status,
)


def read_bps_l1_processor_status_from_file(path: Path) -> BPSL1ProcessorStatus:
    """Read a BPS L1 processor status from file"""

    if not path.exists():
        raise RuntimeError(f"Invalid path: {path}")

    data = load_json(path)
    return translate_model_to_bps_l1_processor_status(model=data)


def initialize_bps_l1_processor_status_from_file(path: Path) -> BPSL1ProcessorStatus:
    """Initialize a BPS L1 processor status from a file"""

    if path.exists():
        return read_bps_l1_processor_status_from_file(path=path)

    return BPSL1ProcessorStatus()


def write_bps_l1_processing_status(status: BPSL1ProcessorStatus, path: Path) -> None:
    """Write a BPS L1 processor status file"""

    model = translate_bps_l1_processor_status_to_model(status=status)

    dump_json(data=model, path=path)
