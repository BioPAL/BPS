# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Main annotations translate test
-------------------------------
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bps.common.io.parsing import validate
from bps.transcoder.io import (
    translate_main_annotation_l1ab,
    translate_main_annotation_l1c,
    translate_main_annotation_l2a_fd,
    translate_main_annotation_l2a_fh,
    translate_main_annotation_l2a_gn,
    translate_main_annotation_l2a_tfh,
    translate_main_annotation_l2b_agb,
    translate_main_annotation_l2b_fd,
    translate_main_annotation_l2b_fh,
)

data_folder = Path(__file__).parent / "data"

xsd_folder_annotation = Path(__file__).parent / ".." / ".." / ".." / "xsd" / "biomass-xsd" / "annotation"
xsd_folders = {"v450": xsd_folder_annotation / "v4.5.0", "v445": xsd_folder_annotation / "v4.4.5"}

readers = {
    "L1A": (
        translate_main_annotation_l1ab.read_main_annotation_l1ab,
        translate_main_annotation_l1ab.write_main_annotation_l1ab,
    ),
    "L1B": (
        translate_main_annotation_l1ab.read_main_annotation_l1ab,
        translate_main_annotation_l1ab.write_main_annotation_l1ab,
    ),
    "L1C": (
        translate_main_annotation_l1c.read_main_annotation_l1c,
        translate_main_annotation_l1c.write_main_annotation_l1c,
    ),
    "L2A_FH": (
        translate_main_annotation_l2a_fh.read_main_annotation_l2a_fh,
        translate_main_annotation_l2a_fh.write_main_annotation_l2a_fh,
    ),
    "L2A_FD": (
        translate_main_annotation_l2a_fd.read_main_annotation_l2a_fd,
        translate_main_annotation_l2a_fd.write_main_annotation_l2a_fd,
    ),
    "L2A_GN": (
        translate_main_annotation_l2a_gn.read_main_annotation_l2a_gn,
        translate_main_annotation_l2a_gn.write_main_annotation_l2a_gn,
    ),
    "L2A_TFH": (
        translate_main_annotation_l2a_tfh.read_main_annotation_l2a_tfh,
        translate_main_annotation_l2a_tfh.write_main_annotation_l2a_tfh,
    ),
    "L2B_FD": (
        translate_main_annotation_l2b_fd.read_main_annotation_l2b_fd,
        translate_main_annotation_l2b_fd.write_main_annotation_l2b_fd,
    ),
    "L2B_FH": (
        translate_main_annotation_l2b_fh.read_main_annotation_l2b_fh,
        translate_main_annotation_l2b_fh.write_main_annotation_l2b_fh,
    ),
    "L2B_AGB": (
        translate_main_annotation_l2b_agb.read_main_annotation_l2b_agb,
        translate_main_annotation_l2b_agb.write_main_annotation_l2b_agb,
    ),
}
products = [
    ("v450/bio_fp_agb_l2b_t_g01_tn00e011_b300_annot.xml", "L2B_AGB"),
    ("v450/bio_fp_fd__l2a_20251212t221127_20251218t221151_t_g01_m01_c___t014_f001_annot.xml", "L2A_FD"),
    ("v450/bio_fp_fd__l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_FD"),
    ("v450/bio_fp_fd__l2b_t_g01_tn00e011_b300_annot.xml", "L2B_FD"),
    ("v450/bio_fp_fd__l2b_t_g01_ts03w060_b100_annot.xml", "L2B_FD"),
    ("v450/bio_fp_fh__l2a_20251212t221127_20251215t221149_t_g01_m01_c___t014_f001_annot.xml", "L2A_FH"),
    ("v450/bio_fp_fh__l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_FH"),
    ("v450/bio_fp_fh__l2b_t_g01_tn00e011_b300_annot.xml", "L2B_FH"),
    ("v450/bio_fp_fh__l2b_t_g01_ts03w061_b100_annot.xml", "L2B_FH"),
    ("v450/bio_fp_gn__l2a_20251215t221128_20251227t221156_t_g01_m01_c___t014_f001_annot.xml", "L2A_GN"),
    ("v450/bio_fp_gn__l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_GN"),
    ("v450/bio_fp_tfh_l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_TFH"),
    ("v450/bio_s3_dgm__1s_20260114t050037_20260114t050058_t_g01_m01_c05_t004_f001_annot.xml", "L1B"),
    ("v450/bio_s3_scs__1s_20260114t050037_20260114t050058_t_g01_m01_c05_t004_f001_annot.xml", "L1A"),
    ("v450/bio_s3_sta__1s_20260126t050044_20260126t050104_t_g01_m01_c09_t004_f001_annot_secondary.xml", "L1C"),
    ("v450/bio_s3_sta__1s_20260126t050044_20260126t050104_t_g01_m01_c09_t004_f001_annot.xml", "L1A"),
    ("v445/bio_fp_agb_l2b_t_g01_tn00e011_b300_annot.xml", "L2B_AGB"),
    ("v445/bio_fp_fd__l2a_20251212t221127_20251218t221151_t_g01_m01_c___t014_f001_annot.xml", "L2A_FD"),
    ("v445/bio_fp_fd__l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_FD"),
    ("v445/bio_fp_fd__l2b_t_g01_tn00e011_b300_annot.xml", "L2B_FD"),
    ("v445/bio_fp_fd__l2b_t_g01_ts03w060_b100_annot.xml", "L2B_FD"),
    ("v445/bio_fp_fh__l2a_20251212t221127_20251215t221149_t_g01_m01_c___t014_f001_annot.xml", "L2A_FH"),
    ("v445/bio_fp_fh__l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_FH"),
    ("v445/bio_fp_fh__l2b_t_g01_tn00e011_b300_annot.xml", "L2B_FH"),
    ("v445/bio_fp_fh__l2b_t_g01_ts03w061_b100_annot.xml", "L2B_FH"),
    ("v445/bio_fp_gn__l2a_20251215t221128_20251227t221156_t_g01_m01_c___t014_f001_annot.xml", "L2A_GN"),
    ("v445/bio_fp_gn__l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_GN"),
    ("v445/bio_fp_tfh_l2a_20260114t050037_20260126t050104_t_g01_m01_c___t004_f001_annot.xml", "L2A_TFH"),
    ("v445/bio_s3_dgm__1s_20260114t050037_20260114t050058_t_g01_m01_c05_t004_f001_annot.xml", "L1B"),
    ("v445/bio_s3_scs__1s_20260114t050037_20260114t050058_t_g01_m01_c05_t004_f001_annot.xml", "L1A"),
    ("v445/bio_s3_sta__1s_20260126t050044_20260126t050104_t_g01_m01_c09_t004_f001_annot_secondary.xml", "L1C"),
    ("v445/bio_s3_sta__1s_20260126t050044_20260126t050104_t_g01_m01_c09_t004_f001_annot.xml", "L1A"),
]
tests = [(file, readers[file_type][0], readers[file_type][1], file_type) for file, file_type in products]

xsd_file_names = {
    "L1A": "bio-l1ab-main-annotation.xsd",
    "L1B": "bio-l1ab-main-annotation.xsd",
    "L1C": "bio-l1c-main-annotation.xsd",
    "L2A_FD": "bio-l2a-fd-main-annotation.xsd",
    "L2A_FH": "bio-l2a-fh-main-annotation.xsd",
    "L2A_GN": "bio-l2a-gn-main-annotation.xsd",
    "L2A_TFH": "bio-l2a-tfh-main-annotation.xsd",
    "L2B_FD": "bio-l2b-fd-main-annotation.xsd",
    "L2B_FH": "bio-l2b-fh-main-annotation.xsd",
    "L2B_AGB": "bio-l2b-agb-main-annotation.xsd",
}


class MainAnnotationIOTestCase(unittest.TestCase):
    def test_validate_annotations(self):
        for file, _, _, file_type in tests:
            with self.subTest(file=file):
                version = file[0:4]
                xsd_file = xsd_folders[version] / xsd_file_names[file_type]
                validate(xml_file=data_folder / file, schema=xsd_file)

    def test_read_write_main_annotation(self):
        for file, read, write, _ in tests:
            with self.subTest(file=file):
                annotation = read(data_folder / file)
                with TemporaryDirectory() as temp_dir:
                    temp_file = Path(temp_dir, "annotation.xml")
                    # pyrefly: ignore[bad-argument-type]
                    write(temp_file, annotation)
                    read_back_annotation = read(temp_file)

                    self.assertEqual(annotation, read_back_annotation)


if __name__ == "__main__":
    unittest.main()
