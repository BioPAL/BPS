# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) - ACRI-ST
# SPDX-License-Identifier: Apache-2.0
"""Placeholder extended-tier test.

This placeholder is intentionally green so CI orchestration can be validated
before real extended harness checks are wired.
"""
import pytest


@pytest.mark.extended
def test_extended_placeholder() -> None:
    assert True
