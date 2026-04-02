# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

from arepytools.timing.precisedatetime import InvalidUtcString
from bps.transcoder.utils.product_name import (
    InvalidBIOMASSProductName,
    parse_l1product_name,
)


class ParseL1ProductNameTest(unittest.TestCase):
    def test_parse_stack_product(self):
        parsed_name = parse_l1product_name(
            "BIO_S1_STA__1M_20170104T060310_20170104T060324_I_G01_M01_C01_T001_F001_01_C65LAM"
        )
        self.assertEqual(parsed_name.satellite_id, "BIO")
        self.assertEqual(parsed_name.stripmap_id, "S1")
        self.assertEqual(parsed_name.processor_id, "STA")
        self.assertEqual(parsed_name.is_monitoring, True)
        self.assertEqual(parsed_name.mission_id, "I")
        self.assertEqual(parsed_name.major_cycle, 1)
        self.assertEqual(parsed_name.repeat_cycle, 1)
        self.assertEqual(parsed_name.track_number, 1)
        self.assertEqual(parsed_name.frame_number, 1)
        self.assertEqual(parsed_name.baseline, 1)
        self.assertEqual(parsed_name.creation_stamp, "C65LAM")

    def test_parse_l1_product(self):
        parsed_name = parse_l1product_name(
            "BIO_S3_SCS__1S_20170212T060330_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70"
        )
        self.assertEqual(parsed_name.satellite_id, "BIO")
        self.assertEqual(parsed_name.stripmap_id, "S3")
        self.assertEqual(parsed_name.processor_id, "SCS")
        self.assertEqual(parsed_name.is_monitoring, False)
        self.assertEqual(parsed_name.mission_id, "I")
        self.assertEqual(parsed_name.major_cycle, 1)
        self.assertEqual(parsed_name.repeat_cycle, 1)
        self.assertEqual(parsed_name.track_number, 1)
        self.assertEqual(parsed_name.frame_number, 1)
        self.assertEqual(parsed_name.baseline, 1)
        self.assertEqual(parsed_name.creation_stamp, "C2CY70")

    def test_parse_product_not_framed(self):
        parsed_name = parse_l1product_name(
            "BIO_S1_SCS__1S_20170101T060225_20170101T060414_I_G03_M04_C55_T010_F____01_CWBZWH"
        )
        self.assertEqual(parsed_name.satellite_id, "BIO")
        self.assertEqual(parsed_name.stripmap_id, "S1")
        self.assertEqual(parsed_name.processor_id, "SCS")
        self.assertEqual(parsed_name.is_monitoring, False)
        self.assertEqual(parsed_name.mission_id, "I")
        self.assertEqual(parsed_name.coverage, 3)
        self.assertEqual(parsed_name.major_cycle, 4)
        self.assertEqual(parsed_name.repeat_cycle, 55)
        self.assertEqual(parsed_name.track_number, 10)
        self.assertEqual(parsed_name.frame_number, -1)
        self.assertEqual(parsed_name.baseline, 1)
        self.assertEqual(parsed_name.creation_stamp, "CWBZWH")

    def test_parse_scs_parc(self):
        parsed_name = parse_l1product_name(
            "BIO_S3_SCSX_1S_20170212T060330_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70"
        )
        self.assertEqual(parsed_name.satellite_id, "BIO")
        self.assertEqual(parsed_name.stripmap_id, "S3")
        self.assertEqual(parsed_name.processor_id, "SCSX")
        self.assertEqual(parsed_name.is_monitoring, False)
        self.assertEqual(parsed_name.mission_id, "I")
        self.assertEqual(parsed_name.coverage, 1)
        self.assertEqual(parsed_name.major_cycle, 1)
        self.assertEqual(parsed_name.repeat_cycle, 1)
        self.assertEqual(parsed_name.track_number, 1)
        self.assertEqual(parsed_name.frame_number, 1)
        self.assertEqual(parsed_name.baseline, 1)
        self.assertEqual(parsed_name.creation_stamp, "C2CY70")

    def test_parse_product_with_unavailable_info(self):
        parsed_name = parse_l1product_name(
            "BIO_S3_SCS__1S_20170212T060330_20170212T060344_I_G___M___C___T____F____01_C2CY70"
        )
        self.assertEqual(parsed_name.satellite_id, "BIO")
        self.assertEqual(parsed_name.stripmap_id, "S3")
        self.assertEqual(parsed_name.processor_id, "SCS")
        self.assertEqual(parsed_name.is_monitoring, False)
        self.assertEqual(parsed_name.mission_id, "I")
        self.assertEqual(parsed_name.coverage, -1)
        self.assertEqual(parsed_name.major_cycle, -1)
        self.assertEqual(parsed_name.repeat_cycle, -1)
        self.assertEqual(parsed_name.track_number, -1)
        self.assertEqual(parsed_name.frame_number, -1)
        self.assertEqual(parsed_name.baseline, 1)
        self.assertEqual(parsed_name.creation_stamp, "C2CY70")

    def test_parse_product_with_dr_repeat_cycle(self):
        parsed_name = parse_l1product_name(
            "BIO_S3_SCS__1S_20170212T060330_20170212T060344_I_G___M___CDR_T____F____01_C2CY70"
        )
        self.assertEqual(parsed_name.satellite_id, "BIO")
        self.assertEqual(parsed_name.stripmap_id, "S3")
        self.assertEqual(parsed_name.processor_id, "SCS")
        self.assertEqual(parsed_name.is_monitoring, False)
        self.assertEqual(parsed_name.mission_id, "I")
        self.assertEqual(parsed_name.coverage, -1)
        self.assertEqual(parsed_name.major_cycle, -1)
        self.assertEqual(parsed_name.repeat_cycle, -2)
        self.assertEqual(parsed_name.track_number, -1)
        self.assertEqual(parsed_name.frame_number, -1)
        self.assertEqual(parsed_name.baseline, 1)
        self.assertEqual(parsed_name.creation_stamp, "C2CY70")

    def test_parse_foo_string(self):
        with self.assertRaises(InvalidBIOMASSProductName):
            parse_l1product_name("Ju5t_A_Rand0m_Str1nG")

    def test_parse_invalid_type(self):
        with self.assertRaises(TypeError):
            parse_l1product_name(42)

    def test_parse_invalid_utc_start_time(self):
        with self.assertRaises(InvalidUtcString):
            parse_l1product_name("BIO_S3_SCS__1S_01234567T012345_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70")

    def test_parse_invalid_utc_stop_time(self):
        with self.assertRaises(InvalidUtcString):
            parse_l1product_name("BIO_S3_SCS__1S_20170212T060344_01234567T012345_I_G01_M01_C01_T001_F001_01_C2CY70")


if __name__ == "__main__":
    unittest.main()
