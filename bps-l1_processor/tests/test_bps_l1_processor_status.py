# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 processor status tests
-----------------------------
"""

import unittest

from bps.l1_processor.status_system.status import (
    BPSL1ProcessorStateInfo,
    BPSL1ProcessorStatus,
    BPSL1ProcessorStep,
    InvalidState,
    are_steps_consecutive,
)


class BPSL1ProcessorStatusTest(unittest.TestCase):
    """Tests BPSL1ProcessorStatus class"""

    def setUp(self) -> None:
        self.partial_states = [
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.PRE_PROCESSOR),
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.CORE_PROCESSOR),
        ]

        self.all_states = [
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.PRE_PROCESSOR),
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.CORE_PROCESSOR),
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.POST_PROCESSOR),
        ]

        self.invalid_states = [
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.CORE_PROCESSOR),
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.PRE_PROCESSOR),
        ]

    def test_bps_l1_processor_status_initialization(self):
        """Test constructor"""

        bps_status = BPSL1ProcessorStatus(states=self.partial_states)
        self.assertEqual(bps_status.states, self.partial_states)

        bps_status = BPSL1ProcessorStatus()
        self.assertEqual(bps_status.states, [])

        self.assertRaises(InvalidState, BPSL1ProcessorStatus, self.invalid_states)

    def test_bps_l1_processor_status_add_state(self):
        """Test add_state member function"""

        bps_status = BPSL1ProcessorStatus(states=self.partial_states)
        bps_status.add_state(state=BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.POST_PROCESSOR))
        self.assertEqual(bps_status.states, self.all_states)

        bps_status = BPSL1ProcessorStatus(states=self.partial_states)
        self.assertRaises(
            RuntimeError,
            bps_status.add_state,
            BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.CORE_PROCESSOR),
        )

    def test_bps_l1_processor_status_is_step_completed(self):
        """Test is_step_completed member function"""

        bps_status = BPSL1ProcessorStatus(states=self.partial_states)

        self.assertTrue(bps_status.is_step_completed(step=BPSL1ProcessorStep.PRE_PROCESSOR))

        self.assertFalse(bps_status.is_step_completed(step=BPSL1ProcessorStep.POST_PROCESSOR))


class BPSL1ProcessorStepTest(unittest.TestCase):
    """Tests BPSL1ProcessorStep enum"""

    def setUp(self) -> None:
        self.consecutive_steps = [
            BPSL1ProcessorStep.PRE_PROCESSOR,
            BPSL1ProcessorStep.CORE_PROCESSOR,
            BPSL1ProcessorStep.POST_PROCESSOR,
        ]

        self.partial_consecutive_steps = [
            BPSL1ProcessorStep.PRE_PROCESSOR,
            BPSL1ProcessorStep.CORE_PROCESSOR,
        ]

        self.invalid_consecutive_steps = [
            BPSL1ProcessorStep.CORE_PROCESSOR,
            BPSL1ProcessorStep.POST_PROCESSOR,
        ]

        self.non_consecutive_steps = [
            BPSL1ProcessorStep.PRE_PROCESSOR,
            BPSL1ProcessorStep.POST_PROCESSOR,
        ]

        self.reversed_steps = [
            BPSL1ProcessorStep.POST_PROCESSOR,
            BPSL1ProcessorStep.CORE_PROCESSOR,
            BPSL1ProcessorStep.PRE_PROCESSOR,
        ]

    def test_are_steps_consecutive(self):
        """Test are_steps_consecutive function"""

        self.assertTrue(are_steps_consecutive(steps=self.consecutive_steps))

        self.assertTrue(are_steps_consecutive(steps=self.partial_consecutive_steps))

        self.assertFalse(are_steps_consecutive(steps=self.invalid_consecutive_steps))

        self.assertFalse(are_steps_consecutive(steps=self.non_consecutive_steps))

        self.assertFalse(are_steps_consecutive(steps=self.reversed_steps))
