# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 processor status translation
-----------------------------------
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bps.l1_processor.status_system.status import (
    BPSL1ProcessorStateInfo,
    BPSL1ProcessorStatus,
    BPSL1ProcessorStep,
)

BPS_L1_STATUS_NAME = "status"
BPS_L1_COMPLETED_STATE_NAME = "completed_state"


@dataclass
class DataBPSL1ProcessorStatus:
    """BPS L1 processor status data model"""

    data: dict[str, Any] = field(default_factory=dict)


def load_json(path: Path) -> DataBPSL1ProcessorStatus:
    """Load json to data model"""

    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    return DataBPSL1ProcessorStatus(data=json_data)


def dump_json(data: DataBPSL1ProcessorStatus, path: Path):
    """Dump data model to json file"""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data.data, file)


def translate_step_to_model(step: BPSL1ProcessorStep) -> str:
    """Translate step to model"""

    return step.name


def translate_state_info_to_model(state: BPSL1ProcessorStateInfo) -> dict:
    """Translate state info to model"""

    return {BPS_L1_COMPLETED_STATE_NAME: translate_step_to_model(state.completed_step)}


def translate_bps_l1_processor_status_to_model(
    status: BPSL1ProcessorStatus,
) -> DataBPSL1ProcessorStatus:
    """Translate status to model"""

    status_states = [translate_state_info_to_model(state) for state in status.states]

    status_data = {BPS_L1_STATUS_NAME: status_states}

    return DataBPSL1ProcessorStatus(data=status_data)


def translate_model_to_step(model: str) -> BPSL1ProcessorStep:
    """Translate model to step"""

    return BPSL1ProcessorStep[model]


def translate_model_to_state_info(model: dict) -> BPSL1ProcessorStateInfo:
    """Translate model to state info"""

    completed_states = model[BPS_L1_COMPLETED_STATE_NAME]

    return BPSL1ProcessorStateInfo(completed_step=translate_model_to_step(completed_states))


def translate_model_to_bps_l1_processor_status(
    model: DataBPSL1ProcessorStatus,
) -> BPSL1ProcessorStatus:
    """Translate model to status"""

    model_states = model.data[BPS_L1_STATUS_NAME]

    states = [translate_model_to_state_info(state_info_model) for state_info_model in model_states]

    return BPSL1ProcessorStatus(states)
