# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) - ACRI-ST
# SPDX-License-Identifier: Apache-2.0
"""Placeholder heavy-tier test.

This placeholder is intentionally green so CI orchestration can be validated
before real heavy harness checks are wired.
"""
import pytest


@pytest.mark.heavy
def test_heavy_placeholder() -> None:
    assert True
