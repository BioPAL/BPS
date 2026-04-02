# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
IRI wrapper test
----------------
"""

import os
from pathlib import Path

from arepyextras.test import DataRepository, Environment, TestSession
from arepyextras.xml import Document
from bps.l1_processor.settings.l1_binaries import BPS_IRI_WRAPPER_EXE_NAME


def test_iri_app(session: TestSession, env: Environment, data: DataRepository):
    """Test that the ionospheric application runs fine"""

    iri_wrapper = BPS_IRI_WRAPPER_EXE_NAME
    iri_data = data.pull("internal_resources/IRI")
    # FORTRAN appears to concatenate paths using a method akin to "joinpath() +
    # file_name".  a trailing slash is forced to the iri_data to ensure
    # compatibility with this behavior.
    iri_data = str(os.path.normpath(iri_data) + os.sep)

    output_xml_file = Path("ionosphericHeight.xml")

    args = [
        iri_wrapper,
        iri_data,
        2015,
        -82,
        40.5,
        40.0,
        -80.0,
        0.0,
        output_xml_file,
    ]
    process = env.run(*args)

    if process.returncode != 0:
        assert process.stderr is not None
        print(process.stderr.read_text())

    session.expect_run_successful(process, fatal=True)

    assert process.stdout is not None
    session.expect_true(len(process.stdout.read_text()) == 0)

    reference_value = 259163.666

    ionospheric_file = Document(output_xml_file)
    value = float(ionospheric_file.get("//IonosphericHeightModel/IonosphericHeight"))
    tolerance = 0.2  # 20 cm tolerance
    session.expect_almost_equal(value, reference_value, abstol=tolerance)  # type: ignore
