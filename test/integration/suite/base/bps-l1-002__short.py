# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 002 (short) test
-------------------
"""

import arepyextras.xml as xml
from arepyextras.test import DataRepository, Environment, TestSession
from bps.common import __version__
from bps.common.bps_logger import get_version_in_logger_format

CURRENT_VERSION = get_version_in_logger_format(__version__)


def test_bps_l1_002__short(session: TestSession, env: Environment, data: DataRepository):
    """Minimal L1 processing test on short data"""

    job_order_filename = data.pull("input/l1002/joborder")

    output_directory = env.root.parent.joinpath("l1002s").absolute()

    aux_ins_path = data.pull("configurations/l1p/CONF-BPS-INS-002")

    aux_pp1_path = data.pull("configurations/l1p/CONF-BPS-PP1-002")

    joborder = xml.Document(job_order_filename)
    joborder.set(
        '//Selected_Input[File_Type/text()="S2_RAW__0S"]/List_of_File_Names/File_Name/text()',
        data.pull("product/l1002short/l0/S2_RAW__0S"),
    )
    joborder.set(
        '//Selected_Input[File_Type/text()="S2_RAW__0M"]/List_of_File_Names/File_Name/text()',
        data.pull("product/l1002short/l0/S2_RAW__0M"),
    )
    joborder.set(
        "//Input[Input_ID/text()='AUX_ORB___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        data.pull("product/l1002short/aux_files/AUX_ORB"),
    )
    joborder.set(
        "//Input[Input_ID/text()='AUX_ATT___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        data.pull("product/l1002short/aux_files/AUX_ATT"),
    )
    joborder.set(
        "//Input[Input_ID/text()='AUX_INS___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_ins_path,
    )
    joborder.set(
        "//Input[Input_ID/text()='AUX_PP1___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        aux_pp1_path,
    )
    joborder.set(
        "//Input[Input_ID/text()='AUX_TEC___']/List_of_Selected_Inputs/Selected_Input/List_of_File_Names/File_Name/text()",
        data.pull("configurations/l1p/CONF-BPS-TEC-003"),
    )
    joborder.set("//Output/File_Dir/text()", output_directory)

    joborder.set(
        "//Processor_Configuration/Request/TOI/Start/text()",
        "2017-01-11T05:06:05.374860000",
    )
    joborder.set(
        "//Processor_Configuration/Request/TOI/Stop/text()",
        "2017-01-11T05:06:34.596708931",
    )

    joborder.set("//Processor_Version/text()", CURRENT_VERSION)
    joborder.set("//Task_Version/text()", CURRENT_VERSION)

    joborder.set("//Number_of_CPU_Cores", 2)
    joborder.set("//Amount_of_RAM", 15360)
    joborder.set("//Disk_Space", 26624)

    l1_proc = env.run("bps_l1_processor", job_order_filename)

    if l1_proc.returncode != 0:
        assert l1_proc.stderr is not None
        print(l1_proc.stderr.read_text())

    session.expect_run_successful(l1_proc, fatal=True)
