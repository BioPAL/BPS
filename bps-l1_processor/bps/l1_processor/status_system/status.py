# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 processor status
-----------------------
"""

from dataclasses import dataclass
from enum import Enum


class BPSL1ProcessorStep(Enum):
    """BPS L1 processor step"""

    PRE_PROCESSOR = 0
    CORE_PROCESSOR = 1
    POST_PROCESSOR = 2
    PARC_PROCESSOR = 3


def are_steps_consecutive(steps: list[BPSL1ProcessorStep]) -> bool:
    """Check if steps are consecutive"""

    return all(index == step.value for index, step in enumerate(steps))


@dataclass
class BPSL1ProcessorStateInfo:
    """BPS L1 processor state"""

    completed_step: BPSL1ProcessorStep


class BPSL1ProcessorStatus:
    """BPS L1 processor status"""

    _states: list[BPSL1ProcessorStateInfo]

    def __init__(self, states: list[BPSL1ProcessorStateInfo] | None = None):
        self._states = states if states is not None else []

        try:
            self._validate_inner_states()
        except InvalidState as exc:
            raise InvalidState("Not consecutive states") from exc

    @property
    def states(self):
        """States of the program"""
        return self._states

    def add_state(self, state: BPSL1ProcessorStateInfo):
        """Append a state info if it is valid"""

        self._states.append(state)
        try:
            self._validate_inner_states()
        except InvalidState as exc:
            raise RuntimeError(f"Error during append of {state}") from exc

    def _validate_inner_states(self):
        """Check the states list validity"""

        if not are_steps_consecutive([status_info.completed_step for status_info in self._states]):
            raise InvalidState("Not consecutive states")

    def is_step_completed(self, step: BPSL1ProcessorStep) -> bool:
        """Check if a step is completed"""

        return step in [status_info.completed_step for status_info in self.states]


class InvalidState(RuntimeError):
    """Raised when inner state is invalid"""
