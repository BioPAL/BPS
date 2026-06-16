# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) - ACRI-ST
# SPDX-License-Identifier: Apache-2.0
"""Placeholder baseline pixel-witness test.

This placeholder is intentionally green so CI orchestration can be validated
before real baseline pixel-witness checks are wired.
"""
import pytest


@pytest.mark.baseline
def test_baseline_placeholder() -> None:
    assert True
