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
from bps.common.io import common
from bps.transcoder.io import translate_common
from bps.transcoder.io.v445 import common as v445
from bps.transcoder.io.v450 import common as v450

VERSIONS = (v450, v445)
LATEST_VERSION = v450


class CommonTranslateTestCase(unittest.TestCase):
    """Test of common types translation"""

    def _assert_enum_translation(self, model_enum, model_enum_latest, common_enum, translate, translate_to_model):
        enum_tuples = [
            (model_member, model_enum_latest[model_member.name], common_enum[model_member.name])
            for model_member in model_enum
        ]
        self.assertEqual(len(enum_tuples), len(model_enum))
        self.assertEqual(len(enum_tuples), len(common_enum))

        for model_member, model_member_latest, common_member in enum_tuples:
            with self.subTest(enum=model_enum.__name__, member=model_member.name):
                self.assertEqual(translate(model_member), common_member)
                self.assertEqual(translate_to_model(common_member), model_member_latest)

    def test_translate_all_enums(self):
        enum_translations = (
            (
                (version.ProcessingModeType for version in VERSIONS),
                LATEST_VERSION.ProcessingModeType,
                common.ProcessingModeType,
                translate_common.translate_processing_mode,
                translate_common.translate_processing_mode_to_model,
            ),
            (
                (version.OrbitAttitudeSourceType for version in VERSIONS),
                LATEST_VERSION.OrbitAttitudeSourceType,
                common.OrbitAttitudeSourceType,
                translate_common.translate_orbit_attitude_source,
                translate_common.translate_orbit_attitude_source_to_model,
            ),
            (
                (version.RfiMaskType for version in VERSIONS),
                LATEST_VERSION.RfiMaskType,
                common.RfiMaskType,
                translate_common.translate_rfi_mask_type,
                translate_common.translate_rfi_mask_type_to_model,
            ),
            (
                (version.RfiMitigationMethodType for version in VERSIONS),
                LATEST_VERSION.RfiMitigationMethodType,
                common.RfiMitigationMethodType,
                translate_common.translate_rfi_mitigation_method_type,
                translate_common.translate_rfi_mitigation_method_type_to_model,
            ),
            (
                (version.RfiMaskGenerationMethodType for version in VERSIONS),
                LATEST_VERSION.RfiMaskGenerationMethodType,
                common.RfiMaskGenerationMethodType,
                translate_common.translate_rfi_mask_generation_method_type,
                translate_common.translate_rfi_mask_generation_method_type_to_model,
            ),
            (
                (version.RfiFmmitigationMethodType for version in VERSIONS),
                LATEST_VERSION.RfiFmmitigationMethodType,
                common.RfiFmmitigationMethodType,
                translate_common.translate_rfi_fmmitigation_method_type,
                translate_common.translate_rfi_fmmitigation_method_type_to_model,
            ),
            (
                (version.RangeCompressionMethodType for version in VERSIONS),
                LATEST_VERSION.RangeCompressionMethodType,
                common.RangeCompressionMethodType,
                translate_common.translate_range_compression_method_type,
                translate_common.translate_range_compression_method_type_to_model,
            ),
            (
                (version.RangeReferenceFunctionType for version in VERSIONS),
                LATEST_VERSION.RangeReferenceFunctionType,
                common.RangeReferenceFunctionType,
                translate_common.translate_range_reference_function_type,
                translate_common.translate_range_reference_function_type_to_model,
            ),
            (
                (version.DcMethodType for version in VERSIONS),
                LATEST_VERSION.DcMethodType,
                common.DcMethodType,
                translate_common.translate_dc_method_type,
                translate_common.translate_dc_method_type_to_model,
            ),
            (
                (version.WeightingWindowType for version in VERSIONS),
                LATEST_VERSION.WeightingWindowType,
                common.WeightingWindowType,
                translate_common.translate_weighting_window_type,
                translate_common.translate_weighting_window_type_to_model,
            ),
            (
                (version.BistaticDelayCorrectionMethodType for version in VERSIONS),
                LATEST_VERSION.BistaticDelayCorrectionMethodType,
                common.BistaticDelayCorrectionMethodType,
                translate_common.translate_bistatic_delay_correction_method_type,
                translate_common.translate_bistatic_delay_correction_method_type_to_model,
            ),
            (
                (version.PolarisationType for version in VERSIONS),
                LATEST_VERSION.PolarisationType,
                common.PolarisationType,
                translate_common.translate_polarisation_type,
                translate_common.translate_polarisation_type_to_model,
            ),
            (
                (version.IonosphereHeightEstimationMethodType for version in VERSIONS),
                LATEST_VERSION.IonosphereHeightEstimationMethodType,
                common.IonosphereHeightEstimationMethodType,
                translate_common.translate_ionosphere_height_estimation_method_type,
                translate_common.translate_ionosphere_height_estimation_method_type_to_model,
            ),
            (
                (version.AutofocusMethodType for version in VERSIONS),
                LATEST_VERSION.AutofocusMethodType,
                common.AutofocusMethodType,
                translate_common.translate_autofocus_method_type,
                translate_common.translate_autofocus_method_type_to_model,
            ),
            (
                (version.UomType for version in VERSIONS),
                LATEST_VERSION.UomType,
                common.UomType,
                translate_common.translate_uom_type,
                translate_common.translate_uom_type_to_model,
            ),
            (
                (version.ProjectionType for version in VERSIONS),
                LATEST_VERSION.ProjectionType,
                common.ProjectionType,
                translate_common.translate_projection_type,
                translate_common.translate_projection_type_to_model,
            ),
            (
                (version.GeodeticReferenceFrameType for version in VERSIONS),
                LATEST_VERSION.GeodeticReferenceFrameType,
                common.GeodeticReferenceFrameType,
                translate_common.translate_geodetic_reference_frame_type,
                translate_common.translate_geodetic_reference_frame_type_to_model,
            ),
            (
                (version.PixelRepresentationType for version in VERSIONS),
                LATEST_VERSION.PixelRepresentationType,
                common.PixelRepresentationType,
                translate_common.translate_pixel_representation_type,
                translate_common.translate_pixel_representation_type_to_model,
            ),
            (
                (version.PixelTypeType for version in VERSIONS),
                LATEST_VERSION.PixelTypeType,
                common.PixelTypeType,
                translate_common.translate_pixel_type_type,
                translate_common.translate_pixel_type_type_to_model,
            ),
            (
                (version.PixelQuantityType for version in VERSIONS),
                LATEST_VERSION.PixelQuantityType,
                common.PixelQuantityType,
                translate_common.translate_pixel_quantity_type,
                translate_common.translate_pixel_quantity_type_to_model,
            ),
            (
                (version.MissionType for version in VERSIONS),
                LATEST_VERSION.MissionType,
                common.MissionType,
                translate_common.translate_mission_type,
                translate_common.translate_mission_type_to_model,
            ),
            (
                (version.SwathType for version in VERSIONS),
                LATEST_VERSION.SwathType,
                common.SwathType,
                translate_common.translate_swath_type,
                translate_common.translate_swath_type_to_model,
            ),
            (
                (version.ProductType for version in VERSIONS),
                LATEST_VERSION.ProductType,
                common.ProductType,
                translate_common.translate_product_type,
                translate_common.translate_product_type_to_model,
            ),
            (
                (version.MissionPhaseIdtype for version in VERSIONS),
                LATEST_VERSION.MissionPhaseIdtype,
                common.MissionPhaseIdtype,
                translate_common.translate_mission_phase_id,
                translate_common.translate_mission_phase_id_to_model,
            ),
            (
                (version.SensorModeType for version in VERSIONS),
                LATEST_VERSION.SensorModeType,
                common.SensorModeType,
                translate_common.translate_sensor_mode,
                translate_common.translate_sensor_mode_to_model,
            ),
            (
                (version.OrbitPassType for version in VERSIONS),
                LATEST_VERSION.OrbitPassType,
                common.OrbitPassType,
                translate_common.translate_orbit_pass,
                translate_common.translate_orbit_pass_to_model,
            ),
            (
                (version.ProductCompositionType for version in VERSIONS),
                LATEST_VERSION.ProductCompositionType,
                common.ProductCompositionType,
                translate_common.translate_product_composition,
                translate_common.translate_product_composition_to_model,
            ),
            (
                (version.DataFormatModeType for version in VERSIONS),
                LATEST_VERSION.DataFormatModeType,
                common.DataFormatModeType,
                translate_common.translate_data_format_mode,
                translate_common.translate_data_format_mode_to_model,
            ),
            (
                (version.InternalCalibrationSourceType for version in VERSIONS),
                LATEST_VERSION.InternalCalibrationSourceType,
                common.InternalCalibrationSourceType,
                translate_common.translate_internal_calibration_source,
                translate_common.translate_internal_calibration_source_to_model,
            ),
            (
                (version.HeightModelBaseType for version in VERSIONS),
                LATEST_VERSION.HeightModelBaseType,
                common.HeightModelBaseType,
                translate_common.translate_height_model_base,
                translate_common.translate_height_model_base_to_model,
            ),
            (
                (version.PrimaryImageSelectionInformationType for version in VERSIONS),
                LATEST_VERSION.PrimaryImageSelectionInformationType,
                common.PrimaryImageSelectionInformationType,
                translate_common.translate_primary_image_selection_information,
                translate_common.translate_primary_image_selection_information_to_model,
            ),
            (
                (version.PrimaryImageSelectionMethodType for version in VERSIONS),
                LATEST_VERSION.PrimaryImageSelectionMethodType,
                common.PrimaryImageSelectionMethodType,
                translate_common.translate_primary_image_selection_method,
                translate_common.translate_primary_image_selection_method_to_model,
            ),
            (
                (version.SkpPhaseCalibrationPostprocessingType for version in VERSIONS),
                LATEST_VERSION.SkpPhaseCalibrationPostprocessingType,
                common.SkpPhaseCalibrationPostprocessingType,
                translate_common.translate_skp_phase_calibration_postprocessing_type,
                translate_common.translate_skp_phase_calibration_postprocessing_type_to_model,
            ),
            (
                (version.CoregistrationMethodType for version in VERSIONS),
                LATEST_VERSION.CoregistrationMethodType,
                common.CoregistrationMethodType,
                translate_common.translate_coregistration_method,
                translate_common.translate_coregistration_method_to_model,
            ),
            (
                (version.ChannelType for version in VERSIONS),
                LATEST_VERSION.ChannelType,
                common.ChannelType,
                translate_common.translate_channel_type,
                translate_common.translate_channel_type_to_model,
            ),
        )

        for model_enums, model_enum_latest, common_enum, translate, translate_to_model in enum_translations:
            for model_enum in model_enums:
                with self.subTest(enum=model_enum.__name__):
                    self._assert_enum_translation(
                        model_enum, model_enum_latest, common_enum, translate, translate_to_model
                    )

    def test_translate_min_max_type(self):
        for version in VERSIONS:
            model_values = version.MinMaxType(min=-2.5, max=7.25)
            common_values = translate_common.translate_min_max_type(model_values)
            assert common_values == common.MinMaxType(min=-2.5, max=7.25)
            assert translate_common.translate_min_max_type_to_model(common_values) == LATEST_VERSION.MinMaxType(
                min=-2.5, max=7.25
            )

    def test_translate_min_max_type_with_unit(self):
        for version in VERSIONS:
            model_values = version.MinMaxTypeWithUnit(
                min=version.FloatWithUnit(value=-2.5, units=version.UomType.M),
                max=version.FloatWithUnit(value=7.25, units=version.UomType.M),
            )
            common_values = translate_common.translate_min_max_type_with_unit(model_values)
            assert common_values == common.MinMaxTypeWithUnit(min=-2.5, max=7.25, units=common.UomType.M)

            latest_model_values = LATEST_VERSION.MinMaxTypeWithUnit(
                min=LATEST_VERSION.FloatWithUnit(value=-2.5, units=LATEST_VERSION.UomType.M),
                max=LATEST_VERSION.FloatWithUnit(value=7.25, units=LATEST_VERSION.UomType.M),
            )
            assert translate_common.translate_min_max_type_with_unit_to_model(common_values) == latest_model_values

            model_values.max.units = version.UomType.RAD
            with self.assertRaises(ValueError):
                translate_common.translate_min_max_type_with_unit(model_values)

    def test_translate_int_array(self):
        for version in VERSIONS:
            model_values = version.IntArray(value="1 -2 3", count=3)
            assert translate_common.translate_int_array(model_values) == [1, -2, 3]

            with self.assertRaises(RuntimeError):
                translate_common.translate_int_array(version.IntArray(value="1 2", count=3))

            with self.assertRaises(RuntimeError):
                translate_common.translate_int_array(version.IntArray(value="1 2", count=3))

        assert translate_common.translate_int_array_to_model([1, -2, 3]) == LATEST_VERSION.IntArray(
            value="1 -2 3", count=3
        )

    def test_translate_float_with_polarisation(self):
        """Translate float with polarisation"""
        for version in VERSIONS:
            float_with_pol = version.FloatWithPolarisation(value=3.4, polarisation=version.PolarisationType.HV)
            assert translate_common.translate_float_with_polarisation(float_with_pol) == (
                3.4,
                common.PolarisationType.HV,
            )
        assert translate_common.translate_float_with_polarisation_to_model(
            (3.4, common.PolarisationType.HV)
        ) == LATEST_VERSION.FloatWithPolarisation(value=3.4, polarisation=LATEST_VERSION.PolarisationType.HV)

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
        for version in VERSIONS:
            assert (
                translate_common.translate_float_with_unit(version.FloatWithUnit(value=3.14, units=version.UomType.RAD))
                == 3.14
            )
        assert translate_common.translate_float_with_unit_to_model(
            3.14, units=common.UomType.RAD
        ) == LATEST_VERSION.FloatWithUnit(value=3.14, units=LATEST_VERSION.UomType.RAD)

    def test_translate_double_with_unit(self):
        """Translate double with unit"""
        for version in VERSIONS:
            assert (
                translate_common.translate_double_with_unit(
                    version.DoubleWithUnit(value=3.14, units=version.UomType.DEG)
                )
                == 3.14
            )
        assert translate_common.translate_double_with_unit_to_model(
            3.14, units=common.UomType.DEG
        ) == LATEST_VERSION.DoubleWithUnit(value=3.14, units=LATEST_VERSION.UomType.DEG)

    def test_translate_float_array(self):
        """Translate a float array"""
        for version in VERSIONS:
            assert translate_common.translate_float_array(version.FloatArray(value="", count=0)) == []
            assert translate_common.translate_float_array(version.FloatArray(value="1 2 3.4", count=3)) == [
                1,
                2,
                3.4,
            ]

        assert translate_common.translate_float_array_to_model([]) == LATEST_VERSION.FloatArray(value="", count=0)
        assert translate_common.translate_float_array_to_model([1, 2, 3.4]) == LATEST_VERSION.FloatArray(
            value="1.0 2.0 3.4", count=3
        )

    def test_translate_float_array_with_units(self):
        """Translate a float array with units"""
        for version in VERSIONS:
            assert (
                translate_common.translate_float_array_with_units(
                    version.FloatArrayWithUnits(value="", count=0, units=version.UomType.HZ)
                )
                == []
            )
        for version in VERSIONS:
            assert translate_common.translate_float_array_with_units(
                version.FloatArrayWithUnits(value="1 2 3.4", count=3, units=version.UomType.HZ)
            ) == [
                1,
                2,
                3.4,
            ]

        assert translate_common.translate_float_array_with_units_to_model(
            [], common.UomType.HZ
        ) == LATEST_VERSION.FloatArrayWithUnits(value="", count=0, units=LATEST_VERSION.UomType.HZ)
        assert translate_common.translate_float_array_with_units_to_model(
            [1, 2, 3.4], units=common.UomType.HZ
        ) == LATEST_VERSION.FloatArrayWithUnits(value="1.0 2.0 3.4", count=3, units=LATEST_VERSION.UomType.HZ)

    def test_translate_double_array(self):
        """Translate a double array"""
        for version in VERSIONS:
            assert translate_common.translate_double_array(version.DoubleArray(value="", count=0)) == []
            assert translate_common.translate_double_array(version.DoubleArray(value="1 2 3.4", count=3)) == [
                1,
                2,
                3.4,
            ]

        assert translate_common.translate_double_array_to_model([]) == LATEST_VERSION.DoubleArray(value="", count=0)
        assert translate_common.translate_double_array_to_model([1, 2, 3.4]) == LATEST_VERSION.DoubleArray(
            value="1.0 2.0 3.4", count=3
        )

    def test_translate_double_array_with_units(self):
        """Translate a double array with units"""
        for version in VERSIONS:
            assert (
                translate_common.translate_double_array_with_units(
                    version.DoubleArrayWithUnits(value="", count=0, units=version.UomType.HZ)
                )
                == []
            )
            assert translate_common.translate_double_array_with_units(
                version.DoubleArrayWithUnits(value="1 2 3.4", count=3, units=version.UomType.C)
            ) == [
                1,
                2,
                3.4,
            ]

        assert translate_common.translate_double_array_with_units_to_model(
            [], common.UomType.HZ
        ) == LATEST_VERSION.DoubleArrayWithUnits(value="", count=0, units=LATEST_VERSION.UomType.HZ)
        assert translate_common.translate_double_array_with_units_to_model(
            [1, 2, 3.4], units=common.UomType.HZ
        ) == LATEST_VERSION.DoubleArrayWithUnits(value="1.0 2.0 3.4", count=3, units=LATEST_VERSION.UomType.HZ)

    def test_translate_complex(self):
        for version in VERSIONS:
            self.assertEqual(translate_common.translate_complex(version.Complex(re=1.0, im=2.0)), 1.0 + 1j * 2.0)
        self.assertEqual(
            translate_common.translate_complex_to_model(1.0 + 1j * 2.0), LATEST_VERSION.Complex(re=1.0, im=2.0)
        )

    def test_translate_datum_type(self):
        for version in VERSIONS:
            self.assertEqual(
                translate_common.translate_datum(
                    version.DatumType(
                        coordinate_reference_system="TEST",
                        geodetic_reference_frame=version.GeodeticReferenceFrameType.WGS84,
                    )
                ),
                common.DatumType("TEST", common.GeodeticReferenceFrameType.WGS84),
            )

        self.assertEqual(
            translate_common.translate_datum_to_model(
                common.DatumType("TEST", common.GeodeticReferenceFrameType.WGS84)
            ),
            v450.DatumType(
                coordinate_reference_system="TEST",
                geodetic_reference_frame=LATEST_VERSION.GeodeticReferenceFrameType.WGS84,
            ),
        )

    def test_translate_state_type(self):
        translated = PreciseDateTime.from_numeric_datetime(2021), 3.2

        for version in VERSIONS:
            model = version.StateType(
                azimuth_time="2021-01-01T00:00:00.000000",
                value=version.FloatWithUnit(value=3.2, units=version.UomType.S),
            )
            self.assertEqual(translate_common.translate_state_type(model), translated)
        self.assertEqual(
            translate_common.translate_state_type_to_model(translated, common.UomType.S),
            LATEST_VERSION.StateType(
                azimuth_time="2021-01-01T00:00:00.000000",
                value=LATEST_VERSION.FloatWithUnit(value=3.2, units=LATEST_VERSION.UomType.S),
            ),
        )

    def test_translate_time_with_pol(self):
        translated = (
            common.PolarisationType.HV,
            PreciseDateTime.from_numeric_datetime(2021),
        )

        for version in VERSIONS:
            model = version.TimeTypeWithPolarisation(
                value="2021-01-01T00:00:00.000000",
                polarisation=version.PolarisationType.HV,
            )
            self.assertEqual(translate_common.translate_time_with_pol(model), translated)

        self.assertEqual(
            translate_common.translate_time_with_pol_to_model(translated),
            LATEST_VERSION.TimeTypeWithPolarisation(
                value="2021-01-01T00:00:00.000000",
                polarisation=LATEST_VERSION.PolarisationType.HV,
            ),
        )

    def test_translate_slant_range_polynomial(self):
        translated = common.SlantRangePolynomialType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
            t0=4.5,
            polynomial=[1.3, 4.5],
        )
        for version in VERSIONS:
            model = version.SlantRangePolynomialType(
                azimuth_time="2021-01-01T00:00:00.000000",
                t0=version.DoubleWithUnit(value=4.5, units=version.UomType.S),
                polynomial=version.DoubleArray(value="1.3 4.5", count=2),
            )
            self.assertEqual(translate_common.translate_slant_range_polynomial(model), translated)

        self.assertEqual(
            translate_common.translate_slant_range_polynomial_to_model(translated),
            LATEST_VERSION.SlantRangePolynomialType(
                azimuth_time="2021-01-01T00:00:00.000000",
                t0=LATEST_VERSION.DoubleWithUnit(value=4.5, units=LATEST_VERSION.UomType.S),
                polynomial=LATEST_VERSION.DoubleArray(value="1.3 4.5", count=2),
            ),
        )

    def test_translate_height_model(self):
        translated = common.HeightModelType(
            value=common.HeightModelBaseType.COPERNICUS_DEM,
            version="Copernicus version",
        )

        for version in VERSIONS:
            model = version.HeightModelType(
                value=version.HeightModelBaseType.COPERNICUS_DEM,
                version="Copernicus version",
            )
            self.assertEqual(
                translate_common.translate_height_model(model),
                translated,
            )
        self.assertEqual(
            translate_common.translate_height_model_to_model(translated),
            LATEST_VERSION.HeightModelType(
                value=LATEST_VERSION.HeightModelBaseType.COPERNICUS_DEM,
                version="Copernicus version",
            ),
        )

    def test_translate_layer_list(self):
        translated = [common.LayerType.ACM, common.LayerType.AZIMUTH_SHIFTS_M]

        for version in VERSIONS:
            model = version.LayerListType(
                layer=[version.LayerType.ACM, version.LayerType.AZIMUTH_SHIFTS_M],
                count=2,
            )
            self.assertEqual(
                translate_common.translate_layer_list(model),
                translated,
            )
        self.assertEqual(
            translate_common.translate_layer_list_to_model(translated),
            LATEST_VERSION.LayerListType(
                layer=[LATEST_VERSION.LayerType.ACM, LATEST_VERSION.LayerType.AZIMUTH_SHIFTS_M],
                count=2,
            ),
        )

    def test_translate_polarisation_combination_method(self):
        for version in VERSIONS:
            model_to_common_pairs = [
                (version.PolarisationCombinationMethodType.AVERAGE, common.PolarisationCombinationMethodType.AVERAGE),
                (version.PolarisationCombinationMethodType.HV, common.PolarisationCombinationMethodType.HV),
                (version.PolarisationCombinationMethodType.VH, common.PolarisationCombinationMethodType.VH),
                (version.PolarisationCombinationMethodType.NONE, None),
            ]
            self.assertEqual(len(model_to_common_pairs), len(LATEST_VERSION.PolarisationCombinationMethodType))

            for model_member, common_member in model_to_common_pairs:
                with self.subTest(member=model_member.name, direction="to common"):
                    self.assertEqual(
                        translate_common.translate_polarisation_combination_method(model_member), common_member
                    )

        common_to_model_pairs = [
            (
                common.PolarisationCombinationMethodType.AVERAGE,
                LATEST_VERSION.PolarisationCombinationMethodType.AVERAGE,
            ),
            (common.PolarisationCombinationMethodType.HV, LATEST_VERSION.PolarisationCombinationMethodType.HV),
            (common.PolarisationCombinationMethodType.VH, LATEST_VERSION.PolarisationCombinationMethodType.VH),
            (None, LATEST_VERSION.PolarisationCombinationMethodType.NONE),
        ]
        self.assertEqual(len(common_to_model_pairs), len(LATEST_VERSION.PolarisationCombinationMethodType))

        for common_member, model_member in common_to_model_pairs:
            with self.subTest(member=model_member.name, direction="to model"):
                self.assertEqual(
                    translate_common.translate_polarisation_combination_method_to_model(common_member), model_member
                )

    def test_translate_layer_type(self):
        enum_tuples = [(model_member, common.LayerType[model_member.name]) for model_member in v450.LayerType]
        self.assertEqual(len(enum_tuples), len(v450.LayerType))
        self.assertEqual(len(enum_tuples), len(common.LayerType))

        for model_member, common_member in enum_tuples:
            with self.subTest(enum=common.LayerType.__name__, member=model_member.name):
                self.assertEqual(translate_common.translate_lut_layer(model_member), common_member)
                self.assertEqual(translate_common.translate_lut_layer_to_model(common_member), model_member)

        enum_tuples = [
            (model_member, common.LayerType[model_member.name], LATEST_VERSION.LayerType[model_member.name])
            for model_member in v445.LayerType
        ]
        self.assertEqual(len(enum_tuples), len(v445.LayerType))
        self.assertEqual(
            len(enum_tuples), len(common.LayerType) - 1
        )  # v445.LayerType has one less member than common.LayerType

        for model_member, common_member, model_member_latest in enum_tuples:
            with self.subTest(enum=common.LayerType.__name__, member=model_member.name):
                self.assertEqual(translate_common.translate_lut_layer(model_member), common_member)
                self.assertEqual(translate_common.translate_lut_layer_to_model(common_member), model_member_latest)


if __name__ == "__main__":
    unittest.main()
