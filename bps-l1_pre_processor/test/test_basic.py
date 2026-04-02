import unittest
from pathlib import Path

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_pre_processor.input_file import L1PreProcessorInputFile
from bps.l1_pre_processor.translate import translate_l1preprocessor_input_file_to_model


class BPSL1PreProcessorInputFile(unittest.TestCase):
    """PreProcessor input file translations"""

    def setUp(self) -> None:
        self.input_file = L1PreProcessorInputFile(
            input_l0s_product=Path("InputL0Sproduct"),
            input_aux_orb_file=Path("Inputorbitfile"),
            input_aux_att_file=Path("Inputattitudefile"),
            input_aux_ins_file=Path("Inputinstrumentparametersfile"),
            input_configuration_file=Path("InputL1processingparametersfile"),
            input_iersbullettin_file=Path("InputIERSbullettinfile"),
            time_of_interest=None,
            output_raw_data_product=Path("OutputRAWdataproduct"),
            bps_configuration_file=Path("BPSConfFile"),
            bps_log_file=Path("BPSLogFile"),
        )

    def test_translation(self):
        """Basic translate test"""
        are_xml_input_file = translate_l1preprocessor_input_file_to_model(self.input_file)
        self.assertEqual(len(are_xml_input_file.step), 1)
        xml_step = are_xml_input_file.step[0]
        self.assertEqual(xml_step.number, 1)
        self.assertEqual(xml_step.total, 1)
        self.assertTrue(xml_step.biomass_l0_import_pre_proc is not None)
        preproc_step = xml_step.biomass_l0_import_pre_proc
        assert preproc_step is not None
        assert preproc_step.time_of_interest is None

    def test_translation_toi(self):
        """Translate input with smart read"""
        self.input_file.time_of_interest = (
            PreciseDateTime.from_numeric_datetime(2000),
            PreciseDateTime.from_numeric_datetime(2099),
        )
        are_xml_input_file = translate_l1preprocessor_input_file_to_model(self.input_file)
        self.assertEqual(len(are_xml_input_file.step), 1)
        xml_step = are_xml_input_file.step[0]
        self.assertEqual(xml_step.number, 1)
        self.assertEqual(xml_step.total, 1)
        self.assertTrue(xml_step.biomass_l0_import_pre_proc is not None)
        preproc_step = xml_step.biomass_l0_import_pre_proc
        assert preproc_step is not None
        assert preproc_step.time_of_interest is not None
        assert preproc_step.time_of_interest.start is not None
        assert preproc_step.time_of_interest.stop is not None
        self.assertEqual(preproc_step.time_of_interest.start.unit, "Utc")
        self.assertEqual(
            preproc_step.time_of_interest.start.value,
            "01-JAN-2000 00:00:00.000000000000",
        )
        self.assertEqual(preproc_step.time_of_interest.stop.unit, "Utc")
        self.assertEqual(
            preproc_step.time_of_interest.stop.value,
            "01-JAN-2099 00:00:00.000000000000",
        )


if __name__ == "__main__":
    unittest.main()
