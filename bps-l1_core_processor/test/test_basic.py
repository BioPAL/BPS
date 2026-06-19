# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

from bps.l1_core_processor.input_file import (
    AntennaProducts,
    BPSL1CoreProcessorInputFile,
    CoreProcessorInputs,
)
from bps.l1_core_processor.translate import (
    translate_bps_antenna_patterns_to_model,
    translate_bps_l1_core_processor_input_file_to_model,
    translate_core_processor_input_to_model,
)


class L1CoreProcessorInputFileTranslateTest(unittest.TestCase):
    """BPS L1 Core Processor Input file translation tests"""

    def setUp(self):
        self.core_inputs = CoreProcessorInputs(
            input_level0_product=Path("raw"),
            processing_options_file=Path("options"),
            processing_parameters_file=Path("params"),
            output_directory=Path("output"),
        )

        self.input_antenna_products = AntennaProducts(Path("d1h"), Path("d2h"), Path("d1v"), Path("d2v"), Path("TXPT"))

    def test_translate_core_inputs(self):
        """Translation of core processor input parameters"""
        core_inputs_model = translate_core_processor_input_to_model(self.core_inputs)
        self.assertEqual(
            core_inputs_model.input_level0_product,
            str(self.core_inputs.input_level0_product),
        )
        self.assertEqual(
            core_inputs_model.processing_options_file,
            str(self.core_inputs.processing_options_file),
        )
        self.assertEqual(
            core_inputs_model.processing_parameters_file,
            str(self.core_inputs.processing_parameters_file),
        )
        self.assertEqual(core_inputs_model.output_path, str(self.core_inputs.output_directory))

    def test_translate_antenna_patterns(self):
        antenna_patterns_model = translate_bps_antenna_patterns_to_model(self.input_antenna_products)

        self.assertEqual(
            antenna_patterns_model.input_antenna_pattern_d1_hproduct,
            str(self.input_antenna_products.d1h_pattern_product),
        )
        self.assertEqual(
            antenna_patterns_model.input_antenna_pattern_d2_hproduct,
            str(self.input_antenna_products.d2h_pattern_product),
        )
        self.assertEqual(
            antenna_patterns_model.input_antenna_pattern_d1_vproduct,
            str(self.input_antenna_products.d1v_pattern_product),
        )
        self.assertEqual(
            antenna_patterns_model.input_antenna_pattern_d2_vproduct,
            str(self.input_antenna_products.d2v_pattern_product),
        )
        self.assertEqual(
            antenna_patterns_model.input_txpower_tracking_product,
            str(self.input_antenna_products.tx_power_tracking_product),
        )

    def test_translate_input_file(self):
        """Translation of the whole input file"""
        input_file = BPSL1CoreProcessorInputFile(
            core_processor_input=self.core_inputs,
            input_antenna_products=self.input_antenna_products,
            input_geomagnetic_field_model_product=Path("GeomMagFolder"),
            input_tec_map_product=None,
            input_climatological_model_file=Path("iono_height.xml"),
            bps_configuration_file=Path("BPSConfFile"),
            bps_log_file=Path("BPSLogFile"),
            input_faraday_rotation_product=None,
            input_phase_screen_product=None,
        )

        input_file_model = translate_bps_l1_core_processor_input_file_to_model(input_file)

        self.assertEqual(len(input_file_model.step), 1)
        self.assertEqual(input_file_model.step[0].number, 1)
        self.assertEqual(input_file_model.step[0].total, 1)

        assert input_file_model.step[0].bpsl1_core_processor is not None
        bps_step = input_file_model.step[0].bpsl1_core_processor

        self.assertIsNotNone(bps_step.core_processor)
        self.assertIsNotNone(bps_step.input_biomass_antenna_pattern2_d)

        self.assertEqual(
            input_file_model.step[0].bpsl1_core_processor.input_geomagnetic_field_model_product,
            str(input_file.input_geomagnetic_field_model_product),
        )

        self.assertEqual(
            input_file_model.step[0].bpsl1_core_processor.input_climatological_model_file,
            str(input_file.input_climatological_model_file),
        )

        self.assertEqual(
            input_file_model.step[0].bpsl1_core_processor.bpsconfiguration_file,
            str(input_file.bps_configuration_file),
        )
        self.assertEqual(
            input_file_model.step[0].bpsl1_core_processor.bpslog_file,
            str(input_file.bps_log_file),
        )


if __name__ == "__main__":
    unittest.main()
