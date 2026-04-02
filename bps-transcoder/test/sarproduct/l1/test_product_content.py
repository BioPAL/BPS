# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path

from bps.transcoder.sarproduct.l1.product_content import L1ProductContent


class L1ProductContentTestCase(unittest.TestCase):
    def test_parse_scs_standard_product(self):
        content = L1ProductContent.from_name(
            "BIO_S3_SCS__1S_20170212T060330_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70"
        )
        assert content.measurement_folder is not None
        assert content.abs_raster is not None
        assert content.phase_raster is not None
        assert content.vrt is not None
        assert content.vrt_xsd is not None

        self.assertEqual(len(content.folders), 6)
        self.assertEqual(len(content.files_and_validators_if_any), 10)
        self.assertEqual(len(content.xsd_schema_files), 8)

        self.assertEqual(
            content.mph_file,
            Path("bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_01_c2cy70.xml"),
        )
        self.assertEqual(content.measurement_folder, Path("measurement"))
        self.assertEqual(
            content.abs_raster,
            Path(
                "measurement",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_i_abs.tiff",
            ),
        )
        self.assertEqual(
            content.phase_raster,
            Path(
                "measurement",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_i_phase.tiff",
            ),
        )
        self.assertEqual(
            content.vrt,
            Path(
                "measurement",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_i.vrt",
            ),
        )
        self.assertEqual(content.annotation_folder, Path("annotation"))
        self.assertEqual(
            content.main_annotation,
            Path(
                "annotation",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_annot.xml",
            ),
        )
        self.assertEqual(
            content.lut,
            Path(
                "annotation",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_lut.nc",
            ),
        )
        self.assertEqual(
            content.navigation_folder,
            Path("annotation", "navigation"),
        )
        self.assertEqual(
            content.orbit,
            Path(
                "annotation",
                "navigation",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_orb.xml",
            ),
        )
        self.assertEqual(
            content.attitude,
            Path(
                "annotation",
                "navigation",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_att.xml",
            ),
        )
        self.assertEqual(content.preview_folder, Path("preview"))
        self.assertEqual(
            content.quicklook_pauli,
            Path(
                "preview",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_ql_pauli.png",
            ),
        )
        self.assertEqual(
            content.quicklook_hsv,
            Path(
                "preview",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_ql_hsv.png",
            ),
        )
        self.assertEqual(
            content.overlay,
            Path(
                "preview",
                "bio_s3_scs__1s_20170212t060330_20170212t060344_i_g01_m01_c01_t001_f001_map.kml",
            ),
        )
        self.assertEqual(content.schema_folder, Path("schema"))
        self.assertEqual(content.l1ab_main_ann_xsd, Path("schema", "bio-l1ab-main-annotation.xsd"))
        self.assertEqual(content.l1_annotation_xsd, Path("schema", "bio-l1-annotations.xsd"))
        self.assertEqual(content.l1_overlay_xsd, Path("schema", "bio-l1-overlay.xsd"))
        self.assertEqual(
            content.l1_overlay_support_xsd,
            Path("schema", "bio-l1-overlay-support.xsd"),
        )
        self.assertEqual(content.common_types_xsd, Path("schema", "bio-common-types.xsd"))
        self.assertEqual(content.aux_orb_xsd, Path("schema", "bio-aux-orbit.xsd"))
        self.assertEqual(content.aux_att_xsd, Path("schema", "bio-aux-attitude.xsd"))
        self.assertEqual(content.vrt_xsd, Path("schema", "bio-l1-vrt.xsd"))

    def test_parse_scs_parc_product(self):
        content = L1ProductContent.from_name(
            "BIO_S3_SCSX_1S_20170212T060330_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70"
        )
        assert content.measurement_folder is not None
        assert content.abs_raster is not None
        assert content.phase_raster is not None
        assert content.vrt is not None
        assert content.vrt_xsd is not None

        self.assertEqual(len(content.folders), 6)
        self.assertEqual(len(content.files_and_validators_if_any), 10)
        self.assertEqual(len(content.xsd_schema_files), 8)

    def test_parse_scs_monitoring_product(self):
        content = L1ProductContent.from_name(
            "BIO_S3_SCS__1M_20170212T060330_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70"
        )
        assert content.measurement_folder is None
        assert content.abs_raster is None
        assert content.phase_raster is None
        assert content.vrt is None
        assert content.vrt_xsd is None

        self.assertEqual(len(content.folders), 5)
        self.assertEqual(len(content.files_and_validators_if_any), 7)
        self.assertEqual(len(content.xsd_schema_files), 7)

    def test_parse_dgm_product(self):
        content = L1ProductContent.from_name(
            "BIO_S3_DGM__1S_20170212T060330_20170212T060344_I_G01_M01_C01_T001_F001_01_C2CY70"
        )
        assert content.measurement_folder is not None
        assert content.abs_raster is not None
        assert content.phase_raster is None
        assert content.vrt is None
        assert content.vrt_xsd is None

        self.assertEqual(len(content.folders), 6)
        self.assertEqual(len(content.files_and_validators_if_any), 7)
        self.assertEqual(len(content.xsd_schema_files), 7)


if __name__ == "__main__":
    unittest.main()
