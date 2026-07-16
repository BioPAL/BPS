# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Aux Ins fill tests
------------------
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from bps.common import MissionPhaseID, Polarization, Swath
from bps.common.common import AcquisitionMode
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsProduct
from bps.l1_pre_processor.aux_ins.chirp import (
    ChirpReplica,
    write_chirp_replicas_to_product_folder,
)
from bps.l1_pre_processor.aux_ins.pattern import (
    BiomassAntennaPattern,
    DoubletID,
    PatternID,
    write_antenna_patterns_to_product_folder,
)


def fill_empty_aux_ins_product(aux_product: Path) -> None:
    """Fill an empty aux ins product with proper naming convention"""
    data_dir = aux_product.joinpath("data")
    data_dir.mkdir(exist_ok=False, parents=False)

    file_names = [
        "bio_aux_ins____20220101t000000_20991231t000000_antenna_patterns.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s1_int.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s1_tom.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s2_int.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s2_tom.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s3_int.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s3_tom.nc",
        "bio_aux_ins____20220101t000000_20991231t000000_ins.xml",
        "bio_aux_ins____20220101t000000_20991231t000000_on_board_filters.nc",
    ]

    for name in file_names:
        data_dir.joinpath(name).touch()


class FillAuxInsContent(unittest.TestCase):
    """Tests of aux ins content fill"""

    def test_fill_aux_ins_content(self):
        """Test of proper aux ins content fill"""
        with TemporaryDirectory() as aux_ins_product:
            aux_ins_product = Path(aux_ins_product)
            fill_empty_aux_ins_product(aux_ins_product)
            content = AuxInsProduct.from_product(aux_ins_product)

            self.assertEqual(
                content.antenna_pattern_file.name,
                "bio_aux_ins____20220101t000000_20991231t000000_antenna_patterns.nc",
            )
            self.assertEqual(
                content.instrument_file.name,
                "bio_aux_ins____20220101t000000_20991231t000000_ins.xml",
            )

            self.assertEqual(len(content.chirp_files), 9)

            expected_names: dict[AcquisitionMode, str] = {
                (
                    Swath.S1,
                    MissionPhaseID.INTERFEROMETRIC,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s1_int.nc",
                (
                    Swath.S1,
                    MissionPhaseID.TOMOGRAPHIC,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s1_tom.nc",
                (
                    Swath.S1,
                    MissionPhaseID.COMMISSIONING,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s1_tom.nc",
                (
                    Swath.S2,
                    MissionPhaseID.INTERFEROMETRIC,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s2_int.nc",
                (
                    Swath.S2,
                    MissionPhaseID.TOMOGRAPHIC,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s2_tom.nc",
                (
                    Swath.S2,
                    MissionPhaseID.COMMISSIONING,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s2_tom.nc",
                (
                    Swath.S3,
                    MissionPhaseID.INTERFEROMETRIC,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s3_int.nc",
                (
                    Swath.S3,
                    MissionPhaseID.TOMOGRAPHIC,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s3_tom.nc",
                (
                    Swath.S3,
                    MissionPhaseID.COMMISSIONING,
                ): "bio_aux_ins____20220101t000000_20991231t000000_chirp_replicas_s3_tom.nc",
            }

            for chirp_id, expected_name in expected_names.items():
                chirp_product = content.chirp_files.get(chirp_id)

                assert chirp_product is not None
                self.assertEqual(chirp_product.name, expected_name)


class WritePatternsTest(unittest.TestCase):
    """Tests of patterns writing"""

    def test_write_antenna_patterns_to_product_folder(self):
        """Test write patterns function"""

        samples = np.arange(5, dtype=float)
        lines = np.arange(4, dtype=float)
        pattern = np.arange(20, dtype=complex).reshape(5, -1)
        bps_pattern = BiomassAntennaPattern(samples, lines, pattern)

        with TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)

            output_product_d1_h = output_dir.joinpath("iPatt2D_D1H")
            output_product_d1_v = output_dir.joinpath("iPatt2D_D1V")
            output_product_d2_h = output_dir.joinpath("iPatt2D_D2H")
            output_product_d2_v = output_dir.joinpath("iPatt2D_D2V")

            patterns: dict[PatternID, BiomassAntennaPattern] = {
                (DoubletID.D1, Polarization.HH): bps_pattern,
                (DoubletID.D1, Polarization.HV): bps_pattern,
                (DoubletID.D1, Polarization.VV): bps_pattern,
                (DoubletID.D1, Polarization.VH): bps_pattern,
                (DoubletID.D2, Polarization.HH): bps_pattern,
                (DoubletID.D2, Polarization.HV): bps_pattern,
                (DoubletID.D2, Polarization.VV): bps_pattern,
                (DoubletID.D2, Polarization.VH): bps_pattern,
            }

            write_antenna_patterns_to_product_folder(
                output_d1_h_product=output_product_d1_h,
                output_d1_v_product=output_product_d1_v,
                output_d2_h_product=output_product_d2_h,
                output_d2_v_product=output_product_d2_v,
                biomass_patterns=patterns,
            )


class WritePReplicasTest(unittest.TestCase):
    """Tests of replicas writing"""

    def test_write_chirp_replicas_to_product_folder(self):
        """Test write replicas function"""

        samples = np.arange(5, dtype=float)
        replica_values = np.arange(5, dtype=complex)
        replica = ChirpReplica(samples, replica_values)

        with TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir, "Replica")

            replicas = {
                Polarization.HH: replica,
                Polarization.HV: replica,
                Polarization.VV: replica,
                Polarization.VH: replica,
            }

            write_chirp_replicas_to_product_folder(output_dir, replicas, "SWATH_NAME")


if __name__ == "__main__":
    unittest.main()
