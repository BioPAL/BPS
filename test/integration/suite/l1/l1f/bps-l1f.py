# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1F test
--------
"""

import sys

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession, skip_if
from bps.common import __version__
from bps.common.bps_logger import get_version_in_logger_format

WIN32 = sys.platform == "win32"

CURRENT_VERSION = get_version_in_logger_format(__version__)


@skip_if(WIN32, reason="test_bps_l1f_001 is linux only")
def test_bps_l1f_001(session: TestSession, env: Environment, _: DataRepository):
    """L1 Framing Processor installation and setup (display help)"""

    completed_process = env.run("bps_l1_framing_processor", "--help")

    assert completed_process.stdout is not None
    session.expect_true(completed_process.stdout.read_text().startswith("Usage: bps_l1_framing_processor"))

    if completed_process.returncode != 0:
        assert completed_process.stderr is not None
        print(completed_process.stderr.read_text())

    session.expect_run_successful(completed_process, fatal=True)


@skip_if(WIN32, reason="test_bps_l1f_002 is linux only")
def test_bps_l1f_002(session: TestSession, env: Environment, data: DataRepository):
    """Minimal L1 Framing Processor execution"""

    joborder_filename = data.pull("input/l1f002/joborder")

    joborder = xml.Document(joborder_filename)
    joborder.set(
        '//Selected_Input[File_Type/text()="S3_RAW__0S"]/List_of_File_Names/File_Name/text()',
        data.pull("product/l1f001/l0/S3_RAW__0S"),
    )
    joborder.set(
        '//Selected_Input[File_Type/text()="S3_RAW__0M"]/List_of_File_Names/File_Name/text()',
        data.pull("product/l1f001/l0/S3_RAW__0M"),
    )
    joborder.set(
        "//Input[Input_ID/text()='AUX_ORB___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        data.pull("product/l1f001/aux_files/AUX_ORB"),
    )

    joborder.set("//Output/File_Dir/text()", "output_directory_l1f002")

    joborder.set("//Processor_Version/text()", CURRENT_VERSION)
    joborder.set("//Task_Version/text()", CURRENT_VERSION)

    l1f_proc = env.run("bps_l1_framing_processor", joborder_filename)

    if l1f_proc.returncode != 0:
        assert l1f_proc.stderr is not None
        print(l1f_proc.stderr.read_text())

    session.expect_run_successful(l1f_proc, fatal=True)
