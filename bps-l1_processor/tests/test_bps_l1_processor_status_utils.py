# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS L1 processor status utils tests
-----------------------------------
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bps.l1_processor.status_system.status import (
    BPSL1ProcessorStateInfo,
    BPSL1ProcessorStatus,
    BPSL1ProcessorStep,
)
from bps.l1_processor.status_system.status_utils import (
    initialize_bps_l1_processor_status_from_file,
    read_bps_l1_processor_status_from_file,
    write_bps_l1_processing_status,
)
from bps.l1_processor.status_system.translate_status import (
    DataBPSL1ProcessorStatus,
    dump_json,
    load_json,
)


class BPSL1ProcessorStatusUtilsTest(unittest.TestCase):
    """Tests BPS L1 processor status utils"""

    def setUp(self) -> None:
        self.data_bps_status = DataBPSL1ProcessorStatus(
            {
                "status": [
                    {"completed_state": "PRE_PROCESSOR"},
                    {"completed_state": "CORE_PROCESSOR"},
                ]
            }
        )

        self.bps_status = BPSL1ProcessorStatus(
            states=[
                BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.PRE_PROCESSOR),
                BPSL1ProcessorStateInfo(completed_step=BPSL1ProcessorStep.CORE_PROCESSOR),
            ]
        )

        self.status_file_name = "bps_l1_test_status.json"

    def test_dump_load_data_bps_l1_processor_status(self):
        """Test dump and load utils for data model"""

        with TemporaryDirectory() as tempdir:
            file_path = Path(tempdir).joinpath(self.status_file_name)

            dump_json(self.data_bps_status, file_path)

            data_bps_status = load_json(file_path)

            self.assertEqual(data_bps_status.data, self.data_bps_status.data)

    def test_read_bps_l1_processor_status_from_file(self):
        """Test read BPS L1 processor status from file"""

        with TemporaryDirectory() as tempdir:
            file_path = Path(tempdir).joinpath(self.status_file_name)
            dump_json(self.data_bps_status, file_path)

            bps_status = read_bps_l1_processor_status_from_file(file_path)
            self.assertEqual(bps_status.states, self.bps_status.states)

        with TemporaryDirectory() as tempdir:
            file_path = Path(tempdir).joinpath(self.status_file_name)

            self.assertRaises(
                RuntimeError,
                read_bps_l1_processor_status_from_file,
                file_path,
            )

    def test_initialize_bps_l1_processor_status_from_file(self):
        """Test initialize BPS L1 processor status from file"""

        with TemporaryDirectory() as tempdir:
            file_path = Path(tempdir).joinpath(self.status_file_name)
            dump_json(self.data_bps_status, file_path)

            bps_status = initialize_bps_l1_processor_status_from_file(file_path)
            self.assertEqual(bps_status.states, self.bps_status.states)

        with TemporaryDirectory() as tempdir:
            file_path = Path(tempdir).joinpath(self.status_file_name)

            empty_status = BPSL1ProcessorStatus()
            bps_status = initialize_bps_l1_processor_status_from_file(file_path)
            self.assertEqual(bps_status.states, empty_status.states)

    def test_write_bps_l1_processor_status(self):
        """Test write BPS L1 processor status file"""

        with TemporaryDirectory() as tempdir:
            file_path = Path(tempdir).joinpath(self.status_file_name)
            write_bps_l1_processing_status(self.bps_status, file_path)

            data_status = load_json(file_path)
            self.assertEqual(data_status.data, self.data_bps_status.data)


if __name__ == "__main__":
    unittest.main()
