# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common translate test
---------------------
"""

import unittest

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common, common_types, translate_common


class CommonTranslateTestCase(unittest.TestCase):
    """Test of common types translation"""

    def _assert_enum_translation(self, model_enum, common_enum, translate, translate_to_model):
        enum_pairs = [(model_member, common_enum[model_member.name]) for model_member in model_enum]
        self.assertEqual(len(enum_pairs), len(model_enum))
        self.assertEqual(len(enum_pairs), len(common_enum))

        for model_member, common_member in enum_pairs:
            with self.subTest(enum=model_enum.__name__, member=model_member.name):
                self.assertEqual(translate(model_member), common_member)
                self.assertEqual(translate_to_model(common_member), model_member)

    def test_translate_all_enums(self):
        enum_translations = (
            (
                common_types.RfiMaskType,
                common.RfiMaskType,
                translate_common.translate_rfi_mask_type,
                translate_common.translate_rfi_mask_type_to_model,
            ),
            (
                common_types.RfiMitigationMethodType,
                common.RfiMitigationMethodType,
                translate_common.translate_rfi_mitigation_method_type,
                translate_common.translate_rfi_mitigation_method_type_to_model,
            ),
            (
                common_types.RfiMaskGenerationMethodType,
                common.RfiMaskGenerationMethodType,
                translate_common.translate_rfi_mask_generation_method_type,
                translate_common.translate_rfi_mask_generation_method_type_to_model,
            ),
            (
                common_types.RfiFmmitigationMethodType,
                common.RfiFmmitigationMethodType,
                translate_common.translate_rfi_fmmitigation_method_type,
                translate_common.translate_rfi_fmmitigation_method_type_to_model,
            ),
            (
                common_types.RangeCompressionMethodType,
                common.RangeCompressionMethodType,
                translate_common.translate_range_compression_method_type,
                translate_common.translate_range_compression_method_type_to_model,
            ),
            (
                common_types.RangeReferenceFunctionType,
                common.RangeReferenceFunctionType,
                translate_common.translate_range_reference_function_type,
                translate_common.translate_range_reference_function_type_to_model,
            ),
            (
                common_types.CompressionMethodType,
                common.CompressionMethodType,
                translate_common.translate_compression_method_type,
                translate_common.translate_compression_method_type_to_model,
            ),
            (
                common_types.DcMethodType,
                common.DcMethodType,
                translate_common.translate_dc_method_type,
                translate_common.translate_dc_method_type_to_model,
            ),
            (
                common_types.WeightingWindowType,
                common.WeightingWindowType,
                translate_common.translate_weighting_window_type,
                translate_common.translate_weighting_window_type_to_model,
            ),
            (
                common_types.BistaticDelayCorrectionMethodType,
                common.BistaticDelayCorrectionMethodType,
                translate_common.translate_bistatic_delay_correction_method_type,
                translate_common.translate_bistatic_delay_correction_method_type_to_model,
            ),
            (
                common_types.PolarisationType,
                common.PolarisationType,
                translate_common.translate_polarisation_type,
                translate_common.translate_polarisation_type_to_model,
            ),
            (
                common_types.IonosphereHeightEstimationMethodType,
                common.IonosphereHeightEstimationMethodType,
                translate_common.translate_ionosphere_height_estimation_method_type,
                translate_common.translate_ionosphere_height_estimation_method_type_to_model,
            ),
            (
                common_types.AutofocusMethodType,
                common.AutofocusMethodType,
                translate_common.translate_autofocus_method_type,
                translate_common.translate_autofocus_method_type_to_model,
            ),
            (
                common_types.UomType,
                common.UomType,
                translate_common.translate_uom_type,
                translate_common.translate_uom_type_to_model,
            ),
            (
                common_types.GeodeticReferenceFrameType,
                common.GeodeticReferenceFrameType,
                translate_common.translate_geodetic_reference_frame_type,
                translate_common.translate_geodetic_reference_frame_type_to_model,
            ),
            (
                common_types.PixelRepresentationType,
                common.PixelRepresentationType,
                translate_common.translate_pixel_representation_type,
                translate_common.translate_pixel_representation_type_to_model,
            ),
            (
                common_types.PixelQuantityType,
                common.PixelQuantityType,
                translate_common.translate_pixel_quantity_type,
                translate_common.translate_pixel_quantity_type_to_model,
            ),
            (
                common_types.SwathType,
                common.SwathType,
                translate_common.translate_swath_type,
                translate_common.translate_swath_type_to_model,
            ),
            (
                common_types.BaselineMethodType,
                common.BaselineMethodType,
                translate_common.translate_baseline_method_type,
                translate_common.translate_baseline_method_type_to_model,
            ),
            (
                common_types.DataFormatModeType,
                common.DataFormatModeType,
                translate_common.translate_data_format_mode,
                translate_common.translate_data_format_mode_to_model,
            ),
            (
                common_types.InternalCalibrationSourceType,
                common.InternalCalibrationSourceType,
                translate_common.translate_internal_calibration_source,
                translate_common.translate_internal_calibration_source_to_model,
            ),
            (
                common_types.HeightModelBaseType,
                common.HeightModelBaseType,
                translate_common.translate_height_model_base,
                translate_common.translate_height_model_base_to_model,
            ),
            (
                common_types.LayerType,
                common.LayerType,
                translate_common.translate_lut_layer,
                translate_common.translate_lut_layer_to_model,
            ),
            (
                common_types.PrimaryImageSelectionInformationType,
                common.PrimaryImageSelectionInformationType,
                translate_common.translate_primary_image_selection_information,
                translate_common.translate_primary_image_selection_information_to_model,
            ),
            (
                common_types.SkpPhaseCalibrationPostprocessingType,
                common.SkpPhaseCalibrationPostprocessingType,
                translate_common.translate_skp_phase_calibration_postprocessing_type,
                translate_common.translate_skp_phase_calibration_postprocessing_type_to_model,
            ),
            (
                common_types.CoregistrationMethodType,
                common.CoregistrationMethodType,
                translate_common.translate_coregistration_method,
                translate_common.translate_coregistration_method_to_model,
            ),
            (
                common_types.CoregistrationExecutionPolicyType,
                common.CoregistrationExecutionPolicyType,
                translate_common.translate_coregistration_execution_policy,
                translate_common.translate_coregistration_execution_policy_to_model,
            ),
            (
                common_types.ChannelType,
                common.ChannelType,
                translate_common.translate_channel_type,
                translate_common.translate_channel_type_to_model,
            ),
        )

        for model_enum, common_enum, translate, translate_to_model in enum_translations:
            with self.subTest(enum=model_enum.__name__):
                self._assert_enum_translation(model_enum, common_enum, translate, translate_to_model)

    def test_translate_min_max_type(self):
        model_values = common_types.MinMaxType(min=-2.5, max=7.25)
        common_values = translate_common.translate_min_max_type(model_values)
        assert common_values == common.MinMaxType(min=-2.5, max=7.25)
        assert translate_common.translate_min_max_type_to_model(common_values) == model_values

    def test_translate_min_max_type_with_unit(self):
        model_values = common_types.MinMaxTypeWithUnit(
            min=common_types.FloatWithUnit(value=-2.5, units=common_types.UomType.M),
            max=common_types.FloatWithUnit(value=7.25, units=common_types.UomType.M),
        )
        common_values = translate_common.translate_min_max_type_with_unit(model_values)
        assert common_values == common.MinMaxTypeWithUnit(min=-2.5, max=7.25, units=common.UomType.M)
        assert translate_common.translate_min_max_type_with_unit_to_model(common_values) == model_values

        model_values.max.units = common_types.UomType.RAD
        with self.assertRaises(ValueError):
            translate_common.translate_min_max_type_with_unit(model_values)

    def test_translate_int_array(self):
        model_values = common_types.IntArray(value="1 -2 3", count=3)
        assert translate_common.translate_int_array(model_values) == [1, -2, 3]
        assert translate_common.translate_int_array_to_model([1, -2, 3]) == model_values

        with self.assertRaises(RuntimeError):
            translate_common.translate_int_array(common_types.IntArray(value="1 2", count=3))

    def test_translate_float_with_polarisation(self):
        """Translate float with polarisation"""
        assert translate_common.translate_float_with_polarisation(
            common_types.FloatWithPolarisation(value=3.4, polarisation=common_types.PolarisationType.HV)
        ) == (3.4, common.PolarisationType.HV)

        assert translate_common.translate_float_with_polarisation_to_model(
            (3.4, common.PolarisationType.HV)
        ) == common_types.FloatWithPolarisation(value=3.4, polarisation=common_types.PolarisationType.HV)

    def test_translate_datetime(self):
        """Translate date"""
        assert translate_common.translate_datetime("2024-07-29T15:43:23.368935") == PreciseDateTime.from_utc_string(
            "29-JUL-2024 15:43:23.368935000000"
        )
        assert (
            translate_common.translate_datetime_to_model(
                PreciseDateTime.from_utc_string("29-JUL-2024 15:43:23.368935108184")
            )
            == "2024-07-29T15:43:23.368935"
        )

    def test_translate_bool(self):
        """Translate bool"""
        self.assertTrue(translate_common.translate_bool("true"))
        self.assertFalse(translate_common.translate_bool("false"))

        assert translate_common.translate_bool_to_model(True) == "true"
        assert translate_common.translate_bool_to_model(False) == "false"

    def test_translate_float_with_unit(self):
        """Translate float with units"""
        assert (
            translate_common.translate_float_with_unit(
                common_types.FloatWithUnit(value=3.14, units=common_types.UomType.RAD)
            )
            == 3.14
        )
        assert translate_common.translate_float_with_unit_to_model(
            3.14, units=common.UomType.RAD
        ) == common_types.FloatWithUnit(value=3.14, units=common_types.UomType.RAD)

    def test_translate_double_with_unit(self):
        """Translate double with unit"""
        assert (
            translate_common.translate_double_with_unit(
                common_types.DoubleWithUnit(value=3.14, units=common_types.UomType.DEG)
            )
            == 3.14
        )
        assert translate_common.translate_double_with_unit_to_model(
            3.14, units=common.UomType.DEG
        ) == common_types.DoubleWithUnit(value=3.14, units=common_types.UomType.DEG)

    def test_translate_float_array(self):
        """Translate a float array"""
        assert translate_common.translate_float_array(common_types.FloatArray(value="", count=0)) == []
        assert translate_common.translate_float_array(common_types.FloatArray(value="1 2 3.4", count=3)) == [
            1,
            2,
            3.4,
        ]

        assert translate_common.translate_float_array_to_model([]) == common_types.FloatArray(value="", count=0)
        assert translate_common.translate_float_array_to_model([1, 2, 3.4]) == common_types.FloatArray(
            value="1.0 2.0 3.4", count=3
        )

    def test_translate_float_array_with_units(self):
        """Translate a float array with units"""
        assert (
            translate_common.translate_float_array_with_units(
                common_types.FloatArrayWithUnits(value="", count=0, units=common_types.UomType.HZ)
            )
            == []
        )
        assert translate_common.translate_float_array_with_units(
            common_types.FloatArrayWithUnits(value="1 2 3.4", count=3, units=common_types.UomType.HZ)
        ) == [
            1,
            2,
            3.4,
        ]

        assert translate_common.translate_float_array_with_units_to_model(
            [], common.UomType.HZ
        ) == common_types.FloatArrayWithUnits(value="", count=0, units=common_types.UomType.HZ)
        assert translate_common.translate_float_array_with_units_to_model(
            [1, 2, 3.4], units=common.UomType.HZ
        ) == common_types.FloatArrayWithUnits(value="1.0 2.0 3.4", count=3, units=common_types.UomType.HZ)

    def test_translate_double_array(self):
        """Translate a double array"""
        assert translate_common.translate_double_array(common_types.DoubleArray(value="", count=0)) == []
        assert translate_common.translate_double_array(common_types.DoubleArray(value="1 2 3.4", count=3)) == [
            1,
            2,
            3.4,
        ]

        assert translate_common.translate_double_array_to_model([]) == common_types.DoubleArray(value="", count=0)
        assert translate_common.translate_double_array_to_model([1, 2, 3.4]) == common_types.DoubleArray(
            value="1.0 2.0 3.4", count=3
        )

    def test_translate_double_array_with_units(self):
        """Translate a double array with units"""
        assert (
            translate_common.translate_double_array_with_units(
                common_types.DoubleArrayWithUnits(value="", count=0, units=common_types.UomType.HZ)
            )
            == []
        )
        assert translate_common.translate_double_array_with_units(
            common_types.DoubleArrayWithUnits(value="1 2 3.4", count=3, units=common_types.UomType.C)
        ) == [
            1,
            2,
            3.4,
        ]

        assert translate_common.translate_double_array_with_units_to_model(
            [], common.UomType.HZ
        ) == common_types.DoubleArrayWithUnits(value="", count=0, units=common_types.UomType.HZ)
        assert translate_common.translate_double_array_with_units_to_model(
            [1, 2, 3.4], units=common.UomType.HZ
        ) == common_types.DoubleArrayWithUnits(value="1.0 2.0 3.4", count=3, units=common_types.UomType.HZ)

    def test_translate_complex(self):
        self.assertEqual(translate_common.translate_complex(common_types.Complex(re=1.0, im=2.0)), 1.0 + 1j * 2.0)
        self.assertEqual(
            translate_common.translate_complex_to_model(1.0 + 1j * 2.0), common_types.Complex(re=1.0, im=2.0)
        )

    def test_translate_datum_type(self):
        self.assertEqual(
            translate_common.translate_datum(
                common_types.DatumType(
                    coordinate_reference_system="TEST",
                    geodetic_reference_frame=common_types.GeodeticReferenceFrameType.WGS84,
                )
            ),
            common.DatumType("TEST", common.GeodeticReferenceFrameType.WGS84),
        )
        self.assertEqual(
            translate_common.translate_datum_to_model(
                common.DatumType("TEST", common.GeodeticReferenceFrameType.WGS84)
            ),
            common_types.DatumType(
                coordinate_reference_system="TEST",
                geodetic_reference_frame=common_types.GeodeticReferenceFrameType.WGS84,
            ),
        )

    def test_translate_state_type(self):
        model = common_types.StateType(
            azimuth_time="2021-01-01T00:00:00.000000",
            value=common_types.FloatWithUnit(value=3.2, units=common_types.UomType.S),
        )
        translated = PreciseDateTime.from_numeric_datetime(2021), 3.2
        self.assertEqual(translate_common.translate_state_type(model), translated)
        self.assertEqual(translate_common.translate_state_type_to_model(translated, common.UomType.S), model)

    def test_translate_time_with_pol(self):
        translated = (
            common.PolarisationType.HV,
            PreciseDateTime.from_numeric_datetime(2021),
        )

        model = common_types.TimeTypeWithPolarisation(
            value="2021-01-01T00:00:00.000000",
            polarisation=common_types.PolarisationType.HV,
        )
        self.assertEqual(translate_common.translate_time_with_pol(model), translated)
        self.assertEqual(translate_common.translate_time_with_pol_to_model(translated), model)

    def test_translate_slant_range_polynomial(self):
        translated = common.SlantRangePolynomialType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
            t0=4.5,
            polynomial=[1.3, 4.5],
        )
        model = common_types.SlantRangePolynomialType(
            azimuth_time="2021-01-01T00:00:00.000000",
            t0=common_types.DoubleWithUnit(value=4.5, units=common_types.UomType.S),
            polynomial=common_types.DoubleArray(value="1.3 4.5", count=2),
        )
        self.assertEqual(translate_common.translate_slant_range_polynomial(model), translated)
        self.assertEqual(translate_common.translate_slant_range_polynomial_to_model(translated), model)

    def test_translate_height_model(self):
        translated = common.HeightModelType(
            value=common.HeightModelBaseType.COPERNICUS_DEM,
            version="Copernicus version",
        )
        model = common_types.HeightModelType(
            value=common_types.HeightModelBaseType.COPERNICUS_DEM,
            version="Copernicus version",
        )
        self.assertEqual(
            translate_common.translate_height_model(model),
            translated,
        )
        self.assertEqual(translate_common.translate_height_model_to_model(translated), model)

    def test_translate_layer_list(self):
        translated = [common.LayerType.ACM, common.LayerType.AZIMUTH_SHIFTS_M]
        model = common_types.LayerListType(
            layer=[common_types.LayerType.ACM, common_types.LayerType.AZIMUTH_SHIFTS_M],
            count=2,
        )
        self.assertEqual(
            translate_common.translate_layer_list(model),
            translated,
        )
        self.assertEqual(translate_common.translate_layer_list_to_model(translated), model)

    def test_translate_polarisation_combination_method(self):
        model_to_common_pairs = [
            (common_types.PolarisationCombinationMethodType.AVERAGE, common.PolarisationCombinationMethodType.AVERAGE),
            (common_types.PolarisationCombinationMethodType.HV, common.PolarisationCombinationMethodType.HV),
            (common_types.PolarisationCombinationMethodType.VH, common.PolarisationCombinationMethodType.VH),
            (common_types.PolarisationCombinationMethodType.NONE, None),
        ]
        self.assertEqual(len(model_to_common_pairs), len(common_types.PolarisationCombinationMethodType))

        for model_member, common_member in model_to_common_pairs:
            with self.subTest(member=model_member.name, direction="to common"):
                self.assertEqual(
                    translate_common.translate_polarisation_combination_method(model_member), common_member
                )

        common_to_model_pairs = [
            (common.PolarisationCombinationMethodType.AVERAGE, common_types.PolarisationCombinationMethodType.AVERAGE),
            (common.PolarisationCombinationMethodType.HV, common_types.PolarisationCombinationMethodType.HV),
            (common.PolarisationCombinationMethodType.VH, common_types.PolarisationCombinationMethodType.VH),
            (None, common_types.PolarisationCombinationMethodType.NONE),
        ]
        self.assertEqual(len(common_to_model_pairs), len(common_types.PolarisationCombinationMethodType))

        for common_member, model_member in common_to_model_pairs:
            with self.subTest(member=model_member.name, direction="to model"):
                self.assertEqual(
                    translate_common.translate_polarisation_combination_method_to_model(common_member), model_member
                )


if __name__ == "__main__":
    unittest.main()
