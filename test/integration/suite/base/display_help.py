# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

import sys

from arepyextras.test import (
    DataRepository,
    Environment,
    TestSession,
    skip_if,  # noqa: F401
)

WIN32 = sys.platform == "win32"

APPLICATIONS = [
    "bps_l1_framing_processor",
    "bps_l1_processor",
    "bps_stack_processor",
    "bps_l2a_processor",
    "bps_l2b_fh_processor",
    "bps_l2b_fd_processor",
    "bps_l2b_agb_processor",
]

LINUX_ONLY = ["bps_l1_framing_processor"]


class DisplayHelp:
    def __init__(self, application_name: str):
        self.__doc__ = f"{application_name} correctly display help message"
        self.application = application_name

    def __call__(self, session: TestSession, env: Environment, data: DataRepository):
        completed_process = env.run(self.application, "--help")
        if completed_process.returncode != 0:
            assert completed_process.stderr is not None
            print(completed_process.stderr.read_text())

        assert completed_process.stdout is not None
        session.expect_true(completed_process.stdout.read_text().startswith(f"Usage: {self.application}"))

        session.expect_run_successful(completed_process, fatal=True)


for application in APPLICATIONS:
    test_name = "test_" + application.lower()
    exec(f'{test_name} = DisplayHelp("{application}")')
    if application in LINUX_ONLY:
        exec(f'{test_name} = skip_if(WIN32, reason="{test_name} is linux only")({test_name})')
