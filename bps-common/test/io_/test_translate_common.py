# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
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

    def test_translate_processing_mode(self):
        """Translate nominal/parc enum"""
        assert (
            translate_common.translate_processing_mode(common_types.ProcessingModeType.NOMINAL)
            == common.ProcessingModeType.NOMINAL
        )
        assert (
            translate_common.translate_processing_mode(common_types.ProcessingModeType.PARC)
            == common.ProcessingModeType.PARC
        )

        assert (
            translate_common.translate_processing_mode_to_model(common.ProcessingModeType.NOMINAL)
            == common_types.ProcessingModeType.NOMINAL
        )
        assert (
            translate_common.translate_processing_mode_to_model(common.ProcessingModeType.PARC)
            == common_types.ProcessingModeType.PARC
        )

    def test_orbit_attitude_source(self):
        """Translate downlink/auxiliary enum"""
        assert (
            translate_common.translate_orbit_attitude_source(common_types.OrbitAttitudeSourceType.AUXILIARY)
            == common.OrbitAttitudeSourceType.AUXILIARY
        )
        assert (
            translate_common.translate_orbit_attitude_source(common_types.OrbitAttitudeSourceType.DOWNLINK)
            == common.OrbitAttitudeSourceType.DOWNLINK
        )

        assert (
            translate_common.translate_orbit_attitude_source_to_model(common.OrbitAttitudeSourceType.AUXILIARY)
            == common_types.OrbitAttitudeSourceType.AUXILIARY
        )
        assert (
            translate_common.translate_orbit_attitude_source_to_model(common.OrbitAttitudeSourceType.DOWNLINK)
            == common_types.OrbitAttitudeSourceType.DOWNLINK
        )

    def test_translate_rfi_mask_type(self):
        """Translate single/multliple rfi mask enum"""
        assert translate_common.translate_rfi_mask_type(common_types.RfiMaskType.SINGLE) == common.RfiMaskType.SINGLE
        assert (
            translate_common.translate_rfi_mask_type(common_types.RfiMaskType.MULTIPLE) == common.RfiMaskType.MULTIPLE
        )

        assert (
            translate_common.translate_rfi_mask_type_to_model(common.RfiMaskType.SINGLE)
            == common_types.RfiMaskType.SINGLE
        )
        assert (
            translate_common.translate_rfi_mask_type_to_model(common.RfiMaskType.MULTIPLE)
            == common_types.RfiMaskType.MULTIPLE
        )

    def test_translate_rfi_mitigation_method_type(self):
        """Translate time/frequency/... enum"""
        assert (
            translate_common.translate_rfi_mitigation_method_type(common_types.RfiMitigationMethodType.TIME)
            == common.RfiMitigationMethodType.TIME
        )
        assert (
            translate_common.translate_rfi_mitigation_method_type(common_types.RfiMitigationMethodType.FREQUENCY)
            == common.RfiMitigationMethodType.FREQUENCY
        )
        assert (
            translate_common.translate_rfi_mitigation_method_type(
                common_types.RfiMitigationMethodType.TIME_AND_FREQUENCY
            )
            == common.RfiMitigationMethodType.TIME_AND_FREQUENCY
        )
        assert (
            translate_common.translate_rfi_mitigation_method_type(
                common_types.RfiMitigationMethodType.FREQUENCY_AND_TIME
            )
            == common.RfiMitigationMethodType.FREQUENCY_AND_TIME
        )

        assert (
            translate_common.translate_rfi_mitigation_method_type_to_model(common.RfiMitigationMethodType.TIME)
            == common_types.RfiMitigationMethodType.TIME
        )
        assert (
            translate_common.translate_rfi_mitigation_method_type_to_model(common.RfiMitigationMethodType.FREQUENCY)
            == common_types.RfiMitigationMethodType.FREQUENCY
        )
        assert (
            translate_common.translate_rfi_mitigation_method_type_to_model(
                common.RfiMitigationMethodType.TIME_AND_FREQUENCY
            )
            == common_types.RfiMitigationMethodType.TIME_AND_FREQUENCY
        )
        assert (
            translate_common.translate_rfi_mitigation_method_type_to_model(
                common.RfiMitigationMethodType.FREQUENCY_AND_TIME
            )
            == common_types.RfiMitigationMethodType.FREQUENCY_AND_TIME
        )

    def test_translate_rfi_mask_generation_method_type(self):
        """Translate and/or enum"""
        assert (
            translate_common.translate_rfi_mask_generation_method_type(common_types.RfiMaskGenerationMethodType.AND)
            == common.RfiMaskGenerationMethodType.AND
        )
        assert (
            translate_common.translate_rfi_mask_generation_method_type(common_types.RfiMaskGenerationMethodType.OR)
            == common.RfiMaskGenerationMethodType.OR
        )

        assert (
            translate_common.translate_rfi_mask_generation_method_type_to_model(common.RfiMaskGenerationMethodType.AND)
            == common_types.RfiMaskGenerationMethodType.AND
        )
        assert (
            translate_common.translate_rfi_mask_generation_method_type_to_model(common.RfiMaskGenerationMethodType.OR)
            == common_types.RfiMaskGenerationMethodType.OR
        )

    def test_translate_range_compression_method_type(self):
        """Translate method enum"""
        assert (
            translate_common.translate_range_compression_method_type(
                common_types.RangeCompressionMethodType.MATCHED_FILTER
            )
            == common.RangeCompressionMethodType.MATCHED_FILTER
        )
        assert (
            translate_common.translate_range_compression_method_type(
                common_types.RangeCompressionMethodType.INVERSE_FILTER
            )
            == common.RangeCompressionMethodType.INVERSE_FILTER
        )

        assert (
            translate_common.translate_range_compression_method_type_to_model(
                common.RangeCompressionMethodType.MATCHED_FILTER
            )
            == common_types.RangeCompressionMethodType.MATCHED_FILTER
        )
        assert (
            translate_common.translate_range_compression_method_type_to_model(
                common.RangeCompressionMethodType.INVERSE_FILTER
            )
            == common_types.RangeCompressionMethodType.INVERSE_FILTER
        )

    def test_translate_range_reference_function_type(self):
        """Translate range reference source nominal/internal/replica enum"""
        assert (
            translate_common.translate_range_reference_function_type(common_types.RangeReferenceFunctionType.INTERNAL)
            == common.RangeReferenceFunctionType.INTERNAL
        )
        assert (
            translate_common.translate_range_reference_function_type(common_types.RangeReferenceFunctionType.NOMINAL)
            == common.RangeReferenceFunctionType.NOMINAL
        )
        assert (
            translate_common.translate_range_reference_function_type(common_types.RangeReferenceFunctionType.REPLICA)
            == common.RangeReferenceFunctionType.REPLICA
        )

        assert (
            translate_common.translate_range_reference_function_type_to_model(
                common.RangeReferenceFunctionType.INTERNAL
            )
            == common_types.RangeReferenceFunctionType.INTERNAL
        )
        assert (
            translate_common.translate_range_reference_function_type_to_model(common.RangeReferenceFunctionType.NOMINAL)
            == common_types.RangeReferenceFunctionType.NOMINAL
        )
        assert (
            translate_common.translate_range_reference_function_type_to_model(common.RangeReferenceFunctionType.REPLICA)
            == common_types.RangeReferenceFunctionType.REPLICA
        )

    def test_translate_dc_method_type(self):
        """Translate dc method"""
        assert (
            translate_common.translate_dc_method_type(common_types.DcMethodType.COMBINED)
            == common.DcMethodType.COMBINED
        )
        assert translate_common.translate_dc_method_type(common_types.DcMethodType.FIXED) == common.DcMethodType.FIXED
        assert (
            translate_common.translate_dc_method_type(common_types.DcMethodType.GEOMETRY)
            == common.DcMethodType.GEOMETRY
        )

        assert (
            translate_common.translate_dc_method_type_to_model(common.DcMethodType.COMBINED)
            == common_types.DcMethodType.COMBINED
        )
        assert (
            translate_common.translate_dc_method_type_to_model(common.DcMethodType.FIXED)
            == common_types.DcMethodType.FIXED
        )
        assert (
            translate_common.translate_dc_method_type_to_model(common.DcMethodType.GEOMETRY)
            == common_types.DcMethodType.GEOMETRY
        )

    def test_translate_weighting_window_type(self):
        """Translate weighting window"""
        assert (
            translate_common.translate_weighting_window_type(common_types.WeightingWindowType.HAMMING)
            == common.WeightingWindowType.HAMMING
        )
        assert (
            translate_common.translate_weighting_window_type(common_types.WeightingWindowType.KAISER)
            == common.WeightingWindowType.KAISER
        )
        assert (
            translate_common.translate_weighting_window_type(common_types.WeightingWindowType.NONE)
            == common.WeightingWindowType.NONE
        )

        assert (
            translate_common.translate_weighting_window_type_to_model(common.WeightingWindowType.HAMMING)
            == common_types.WeightingWindowType.HAMMING
        )
        assert (
            translate_common.translate_weighting_window_type_to_model(common.WeightingWindowType.KAISER)
            == common_types.WeightingWindowType.KAISER
        )
        assert (
            translate_common.translate_weighting_window_type_to_model(common.WeightingWindowType.NONE)
            == common_types.WeightingWindowType.NONE
        )

    def test_translate_bistatic_delay_correction_method_type(self):
        """Translate bistatic delay correction"""
        assert (
            translate_common.translate_bistatic_delay_correction_method_type(
                common_types.BistaticDelayCorrectionMethodType.BULK
            )
            == common.BistaticDelayCorrectionMethodType.BULK
        )
        assert (
            translate_common.translate_bistatic_delay_correction_method_type(
                common_types.BistaticDelayCorrectionMethodType.FULL
            )
            == common.BistaticDelayCorrectionMethodType.FULL
        )

        assert (
            translate_common.translate_bistatic_delay_correction_method_type_to_model(
                common.BistaticDelayCorrectionMethodType.BULK
            )
            == common_types.BistaticDelayCorrectionMethodType.BULK
        )
        assert (
            translate_common.translate_bistatic_delay_correction_method_type_to_model(
                common.BistaticDelayCorrectionMethodType.FULL
            )
            == common_types.BistaticDelayCorrectionMethodType.FULL
        )

    def test_translate_polarisation_type(self):
        """Translate polarisation type"""
        assert (
            translate_common.translate_polarisation_type(common_types.PolarisationType.HH) == common.PolarisationType.HH
        )
        assert (
            translate_common.translate_polarisation_type(common_types.PolarisationType.HV) == common.PolarisationType.HV
        )
        assert (
            translate_common.translate_polarisation_type(common_types.PolarisationType.VH) == common.PolarisationType.VH
        )
        assert (
            translate_common.translate_polarisation_type(common_types.PolarisationType.VV) == common.PolarisationType.VV
        )
        assert (
            translate_common.translate_polarisation_type(common_types.PolarisationType.XX) == common.PolarisationType.XX
        )

        assert (
            translate_common.translate_polarisation_type_to_model(common.PolarisationType.HH)
            == common_types.PolarisationType.HH
        )
        assert (
            translate_common.translate_polarisation_type_to_model(common.PolarisationType.HV)
            == common_types.PolarisationType.HV
        )
        assert (
            translate_common.translate_polarisation_type_to_model(common.PolarisationType.VH)
            == common_types.PolarisationType.VH
        )
        assert (
            translate_common.translate_polarisation_type_to_model(common.PolarisationType.VV)
            == common_types.PolarisationType.VV
        )
        assert (
            translate_common.translate_polarisation_type_to_model(common.PolarisationType.XX)
            == common_types.PolarisationType.XX
        )

    def test_translate_float_with_polarisation(self):
        """Translate float with polarisation"""
        assert translate_common.translate_float_with_polarisation(
            common_types.FloatWithPolarisation(value=3.4, polarisation=common_types.PolarisationType.HV)
        ) == (3.4, common.PolarisationType.HV)

        assert translate_common.translate_float_with_polarisation_to_model(
            (3.4, common.PolarisationType.HV)
        ) == common_types.FloatWithPolarisation(value=3.4, polarisation=common_types.PolarisationType.HV)

    def test_translate_ionosphere_height_estimation_method_type(self):
        """Translate height estimation method"""
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type(
                common_types.IonosphereHeightEstimationMethodType.AUTOMATIC
            )
            == common.IonosphereHeightEstimationMethodType.AUTOMATIC
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type(
                common_types.IonosphereHeightEstimationMethodType.FEATURE_TRACKING
            )
            == common.IonosphereHeightEstimationMethodType.FEATURE_TRACKING
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type(
                common_types.IonosphereHeightEstimationMethodType.FIXED
            )
            == common.IonosphereHeightEstimationMethodType.FIXED
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type(
                common_types.IonosphereHeightEstimationMethodType.MODEL
            )
            == common.IonosphereHeightEstimationMethodType.MODEL
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type(
                common_types.IonosphereHeightEstimationMethodType.SQUINT_SENSITIVITY
            )
            == common.IonosphereHeightEstimationMethodType.SQUINT_SENSITIVITY
        )

        assert (
            translate_common.translate_ionosphere_height_estimation_method_type_to_model(
                common.IonosphereHeightEstimationMethodType.AUTOMATIC
            )
            == common_types.IonosphereHeightEstimationMethodType.AUTOMATIC
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type_to_model(
                common.IonosphereHeightEstimationMethodType.FEATURE_TRACKING
            )
            == common_types.IonosphereHeightEstimationMethodType.FEATURE_TRACKING
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type_to_model(
                common.IonosphereHeightEstimationMethodType.FIXED
            )
            == common_types.IonosphereHeightEstimationMethodType.FIXED
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type_to_model(
                common.IonosphereHeightEstimationMethodType.MODEL
            )
            == common_types.IonosphereHeightEstimationMethodType.MODEL
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type_to_model(
                common.IonosphereHeightEstimationMethodType.SQUINT_SENSITIVITY
            )
            == common_types.IonosphereHeightEstimationMethodType.SQUINT_SENSITIVITY
        )
        assert (
            translate_common.translate_ionosphere_height_estimation_method_type_to_model(
                common.IonosphereHeightEstimationMethodType.NA
            )
            == common_types.IonosphereHeightEstimationMethodType.NA
        )

    def test_translate_autofocus_method_type(self):
        """Translate autofocus method"""
        assert (
            translate_common.translate_autofocus_method_type(common_types.AutofocusMethodType.MAP_DRIFT)
            == common.AutofocusMethodType.MAP_DRIFT
        )

        assert (
            translate_common.translate_autofocus_method_type_to_model(common.AutofocusMethodType.MAP_DRIFT)
            == common_types.AutofocusMethodType.MAP_DRIFT
        )

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

    def test_translate_uom_type(self):
        """Translate unit of measure method"""
        assert translate_common.translate_uom_type(common_types.UomType.S) == common.UomType.S
        assert translate_common.translate_uom_type(common_types.UomType.M) == common.UomType.M
        assert translate_common.translate_uom_type(common_types.UomType.SAMPLES) == common.UomType.SAMPLES
        assert translate_common.translate_uom_type(common_types.UomType.LINES) == common.UomType.LINES
        assert translate_common.translate_uom_type(common_types.UomType.DEG) == common.UomType.DEG
        assert translate_common.translate_uom_type(common_types.UomType.RAD) == common.UomType.RAD
        assert translate_common.translate_uom_type(common_types.UomType.HZ) == common.UomType.HZ
        assert translate_common.translate_uom_type(common_types.UomType.HZ_S) == common.UomType.HZ_S
        assert translate_common.translate_uom_type(common_types.UomType.D_B) == common.UomType.D_B
        assert translate_common.translate_uom_type(common_types.UomType.MBPS) == common.UomType.MBPS
        assert translate_common.translate_uom_type(common_types.UomType.C) == common.UomType.C
        assert translate_common.translate_uom_type(common_types.UomType.K) == common.UomType.K
        assert translate_common.translate_uom_type(common_types.UomType.DEG_S) == common.UomType.DEG_S
        assert translate_common.translate_uom_type(common_types.UomType.MM_KM) == common.UomType.MM_KM
        assert translate_common.translate_uom_type(common_types.UomType.DEG_M) == common.UomType.DEG_M
        assert translate_common.translate_uom_type(common_types.UomType.VALUE_1_M) == common.UomType.VALUE_1_M
        assert translate_common.translate_uom_type(common_types.UomType.T_HA) == common.UomType.T_HA
        assert translate_common.translate_uom_type(common_types.UomType.RAD_N_T_2) == common.UomType.RAD_N_T_2

        assert translate_common.translate_uom_type_to_model(common.UomType.S) == common_types.UomType.S
        assert translate_common.translate_uom_type_to_model(common.UomType.M) == common_types.UomType.M
        assert translate_common.translate_uom_type_to_model(common.UomType.SAMPLES) == common_types.UomType.SAMPLES
        assert translate_common.translate_uom_type_to_model(common.UomType.LINES) == common_types.UomType.LINES
        assert translate_common.translate_uom_type_to_model(common.UomType.DEG) == common_types.UomType.DEG
        assert translate_common.translate_uom_type_to_model(common.UomType.RAD) == common_types.UomType.RAD
        assert translate_common.translate_uom_type_to_model(common.UomType.HZ) == common_types.UomType.HZ
        assert translate_common.translate_uom_type_to_model(common.UomType.HZ_S) == common_types.UomType.HZ_S
        assert translate_common.translate_uom_type_to_model(common.UomType.D_B) == common_types.UomType.D_B
        assert translate_common.translate_uom_type_to_model(common.UomType.MBPS) == common_types.UomType.MBPS
        assert translate_common.translate_uom_type_to_model(common.UomType.C) == common_types.UomType.C
        assert translate_common.translate_uom_type_to_model(common.UomType.K) == common_types.UomType.K
        assert translate_common.translate_uom_type_to_model(common.UomType.DEG_S) == common_types.UomType.DEG_S
        assert translate_common.translate_uom_type_to_model(common.UomType.MM_KM) == common_types.UomType.MM_KM
        assert translate_common.translate_uom_type_to_model(common.UomType.DEG_M) == common_types.UomType.DEG_M
        assert translate_common.translate_uom_type_to_model(common.UomType.VALUE_1_M) == common_types.UomType.VALUE_1_M
        assert translate_common.translate_uom_type_to_model(common.UomType.T_HA) == common_types.UomType.T_HA
        assert translate_common.translate_uom_type_to_model(common.UomType.RAD_N_T_2) == common_types.UomType.RAD_N_T_2

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

    def test_translate_projection_type(self):
        assert (
            translate_common.translate_projection_type(common_types.ProjectionType.SLANT_RANGE)
            == common.ProjectionType.SLANT_RANGE
        )

        assert (
            translate_common.translate_projection_type_to_model(common.ProjectionType.SLANT_RANGE)
            == common_types.ProjectionType.SLANT_RANGE
        )
        assert (
            translate_common.translate_projection_type(common_types.ProjectionType.GROUND_RANGE)
            == common.ProjectionType.GROUND_RANGE
        )

        assert (
            translate_common.translate_projection_type_to_model(common.ProjectionType.GROUND_RANGE)
            == common_types.ProjectionType.GROUND_RANGE
        )
        assert (
            translate_common.translate_projection_type(common_types.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG)
            == common.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG
        )

        assert (
            translate_common.translate_projection_type_to_model(common.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG)
            == common_types.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG
        )
        assert translate_common.translate_projection_type(common_types.ProjectionType.DGG) == common.ProjectionType.DGG

        assert (
            translate_common.translate_projection_type_to_model(common.ProjectionType.DGG)
            == common_types.ProjectionType.DGG
        )

    def test_translate_geodetic_reference_frame_type(self):
        assert (
            translate_common.translate_geodetic_reference_frame_type(common_types.GeodeticReferenceFrameType.WGS84)
            == common.GeodeticReferenceFrameType.WGS84
        )

        assert (
            translate_common.translate_geodetic_reference_frame_type_to_model(common.GeodeticReferenceFrameType.WGS84)
            == common_types.GeodeticReferenceFrameType.WGS84
        )

        assert (
            translate_common.translate_geodetic_reference_frame_type(common_types.GeodeticReferenceFrameType.ITRF2000)
            == common.GeodeticReferenceFrameType.ITRF2000
        )
        assert (
            translate_common.translate_geodetic_reference_frame_type_to_model(
                common.GeodeticReferenceFrameType.ITRF2000
            )
            == common_types.GeodeticReferenceFrameType.ITRF2000
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

    def test_translate_pixel_representation(self):
        self.assertEqual(
            translate_common.translate_pixel_representation_type(common_types.PixelRepresentationType.I_Q),
            common.PixelRepresentationType.I_Q,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(common.PixelRepresentationType.I_Q),
            common_types.PixelRepresentationType.I_Q,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(common_types.PixelRepresentationType.ABS_PHASE),
            common.PixelRepresentationType.ABS_PHASE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(common.PixelRepresentationType.ABS_PHASE),
            common_types.PixelRepresentationType.ABS_PHASE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(common_types.PixelRepresentationType.ABS),
            common.PixelRepresentationType.ABS,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(common.PixelRepresentationType.ABS),
            common_types.PixelRepresentationType.ABS,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.PROBABILITY_OF_CHANGE
            ),
            common.PixelRepresentationType.PROBABILITY_OF_CHANGE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.PROBABILITY_OF_CHANGE
            ),
            common_types.PixelRepresentationType.PROBABILITY_OF_CHANGE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.COMPUTED_FOREST_MASK
            ),
            common.PixelRepresentationType.COMPUTED_FOREST_MASK,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.COMPUTED_FOREST_MASK
            ),
            common_types.PixelRepresentationType.COMPUTED_FOREST_MASK,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.FOREST_DISTURBANCE
            ),
            common.PixelRepresentationType.FOREST_DISTURBANCE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.FOREST_DISTURBANCE
            ),
            common_types.PixelRepresentationType.FOREST_DISTURBANCE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(common_types.PixelRepresentationType.HEAT_MAP),
            common.PixelRepresentationType.HEAT_MAP,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(common.PixelRepresentationType.HEAT_MAP),
            common_types.PixelRepresentationType.HEAT_MAP,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(common_types.PixelRepresentationType.FOREST_HEIGHT_M),
            common.PixelRepresentationType.FOREST_HEIGHT_M,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.FOREST_HEIGHT_M
            ),
            common_types.PixelRepresentationType.FOREST_HEIGHT_M,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.FOREST_HEIGHT_QUALITY
            ),
            common.PixelRepresentationType.FOREST_HEIGHT_QUALITY,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.FOREST_HEIGHT_QUALITY
            ),
            common_types.PixelRepresentationType.FOREST_HEIGHT_QUALITY,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(common_types.PixelRepresentationType.FOREST_MASK),
            common.PixelRepresentationType.FOREST_MASK,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(common.PixelRepresentationType.FOREST_MASK),
            common_types.PixelRepresentationType.FOREST_MASK,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.GROUND_CANCELLED_BACKSCATTER
            ),
            common.PixelRepresentationType.GROUND_CANCELLED_BACKSCATTER,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.GROUND_CANCELLED_BACKSCATTER
            ),
            common_types.PixelRepresentationType.GROUND_CANCELLED_BACKSCATTER,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_T_HA
            ),
            common.PixelRepresentationType.ABOVE_GROUND_BIOMASS_T_HA,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.ABOVE_GROUND_BIOMASS_T_HA
            ),
            common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_T_HA,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_QUALITY_T_HA
            ),
            common.PixelRepresentationType.ABOVE_GROUND_BIOMASS_QUALITY_T_HA,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.ABOVE_GROUND_BIOMASS_QUALITY_T_HA
            ),
            common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_QUALITY_T_HA,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type(
                common_types.PixelRepresentationType.ACQUISITION_ID_IMAGE
            ),
            common.PixelRepresentationType.ACQUISITION_ID_IMAGE,
        )
        self.assertEqual(
            translate_common.translate_pixel_representation_type_to_model(
                common.PixelRepresentationType.ACQUISITION_ID_IMAGE
            ),
            common_types.PixelRepresentationType.ACQUISITION_ID_IMAGE,
        )

    def test_translate_pixel_type(self):
        self.assertEqual(
            translate_common.translate_pixel_type_type(common_types.PixelTypeType.VALUE_32_BIT_FLOAT),
            common.PixelTypeType.VALUE_32_BIT_FLOAT,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type_to_model(common.PixelTypeType.VALUE_32_BIT_FLOAT),
            common_types.PixelTypeType.VALUE_32_BIT_FLOAT,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type(common_types.PixelTypeType.VALUE_16_BIT_SIGNED_INTEGER),
            common.PixelTypeType.VALUE_16_BIT_SIGNED_INTEGER,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type_to_model(common.PixelTypeType.VALUE_16_BIT_SIGNED_INTEGER),
            common_types.PixelTypeType.VALUE_16_BIT_SIGNED_INTEGER,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type(common_types.PixelTypeType.VALUE_16_BIT_UNSIGNED_INTEGER),
            common.PixelTypeType.VALUE_16_BIT_UNSIGNED_INTEGER,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type_to_model(common.PixelTypeType.VALUE_16_BIT_UNSIGNED_INTEGER),
            common_types.PixelTypeType.VALUE_16_BIT_UNSIGNED_INTEGER,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type(common_types.PixelTypeType.VALUE_8_BIT_UNSIGNED_INTEGER),
            common.PixelTypeType.VALUE_8_BIT_UNSIGNED_INTEGER,
        )
        self.assertEqual(
            translate_common.translate_pixel_type_type_to_model(common.PixelTypeType.VALUE_8_BIT_UNSIGNED_INTEGER),
            common_types.PixelTypeType.VALUE_8_BIT_UNSIGNED_INTEGER,
        )

    def test_translate_pixel_quantity(self):
        self.assertEqual(
            translate_common.translate_pixel_quantity_type(common_types.PixelQuantityType.BETA_NOUGHT),
            common.PixelQuantityType.BETA_NOUGHT,
        )
        self.assertEqual(
            translate_common.translate_pixel_quantity_type_to_model(common.PixelQuantityType.BETA_NOUGHT),
            common_types.PixelQuantityType.BETA_NOUGHT,
        )
        self.assertEqual(
            translate_common.translate_pixel_quantity_type(common_types.PixelQuantityType.SIGMA_NOUGHT),
            common.PixelQuantityType.SIGMA_NOUGHT,
        )
        self.assertEqual(
            translate_common.translate_pixel_quantity_type_to_model(common.PixelQuantityType.SIGMA_NOUGHT),
            common_types.PixelQuantityType.SIGMA_NOUGHT,
        )
        self.assertEqual(
            translate_common.translate_pixel_quantity_type(common_types.PixelQuantityType.GAMMA_NOUGHT),
            common.PixelQuantityType.GAMMA_NOUGHT,
        )
        self.assertEqual(
            translate_common.translate_pixel_quantity_type_to_model(common.PixelQuantityType.GAMMA_NOUGHT),
            common_types.PixelQuantityType.GAMMA_NOUGHT,
        )

    def test_translate_mission_type(self):
        self.assertEqual(
            translate_common.translate_mission_type(common_types.MissionType.BIOMASS),
            common.MissionType.BIOMASS,
        )
        self.assertEqual(
            translate_common.translate_mission_type_to_model(common.MissionType.BIOMASS),
            common_types.MissionType.BIOMASS,
        )

    def test_translate_swath_type(self):
        self.assertEqual(
            translate_common.translate_swath_type(common_types.SwathType.S1),
            common.SwathType.S1,
        )
        self.assertEqual(
            translate_common.translate_swath_type_to_model(common.SwathType.S1),
            common_types.SwathType.S1,
        )
        self.assertEqual(
            translate_common.translate_swath_type(common_types.SwathType.S2),
            common.SwathType.S2,
        )
        self.assertEqual(
            translate_common.translate_swath_type_to_model(common.SwathType.S2),
            common_types.SwathType.S2,
        )
        self.assertEqual(
            translate_common.translate_swath_type(common_types.SwathType.S3),
            common.SwathType.S3,
        )
        self.assertEqual(
            translate_common.translate_swath_type_to_model(common.SwathType.S3),
            common_types.SwathType.S3,
        )

    def test_translate_product_type(self):
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.SCS),
            common.ProductType.SCS,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.SCS),
            common_types.ProductType.SCS,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.DGM),
            common.ProductType.DGM,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.DGM),
            common_types.ProductType.DGM,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.STA),
            common.ProductType.STA,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.STA),
            common_types.ProductType.STA,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.FH_L2_A),
            common.ProductType.FH_L2_A,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.FH_L2_A),
            common_types.ProductType.FH_L2_A,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.FH_L2_B),
            common.ProductType.FH_L2_B,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.FH_L2_B),
            common_types.ProductType.FH_L2_B,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.FD_L2_A),
            common.ProductType.FD_L2_A,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.FD_L2_A),
            common_types.ProductType.FD_L2_A,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.FD_L2_B),
            common.ProductType.FD_L2_B,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.FD_L2_B),
            common_types.ProductType.FD_L2_B,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.GN_L2_A),
            common.ProductType.GN_L2_A,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.GN_L2_A),
            common_types.ProductType.GN_L2_A,
        )
        self.assertEqual(
            translate_common.translate_product_type(common_types.ProductType.AGB_L2_B),
            common.ProductType.AGB_L2_B,
        )
        self.assertEqual(
            translate_common.translate_product_type_to_model(common.ProductType.AGB_L2_B),
            common_types.ProductType.AGB_L2_B,
        )

    def test_translate_mission_phase(self):
        self.assertEqual(
            translate_common.translate_mission_phase_id(common_types.MissionPhaseIdtype.COM),
            common.MissionPhaseIdtype.COM,
        )
        self.assertEqual(
            translate_common.translate_mission_phase_id_to_model(common.MissionPhaseIdtype.COM),
            common_types.MissionPhaseIdtype.COM,
        )
        self.assertEqual(
            translate_common.translate_mission_phase_id(common_types.MissionPhaseIdtype.TOM),
            common.MissionPhaseIdtype.TOM,
        )
        self.assertEqual(
            translate_common.translate_mission_phase_id_to_model(common.MissionPhaseIdtype.TOM),
            common_types.MissionPhaseIdtype.TOM,
        )
        self.assertEqual(
            translate_common.translate_mission_phase_id(common_types.MissionPhaseIdtype.INT),
            common.MissionPhaseIdtype.INT,
        )
        self.assertEqual(
            translate_common.translate_mission_phase_id_to_model(common.MissionPhaseIdtype.INT),
            common_types.MissionPhaseIdtype.INT,
        )

    def test_translate_sensor_mode(self):
        self.assertEqual(
            translate_common.translate_sensor_mode(common_types.SensorModeType.MEASUREMENT),
            common.SensorModeType.MEASUREMENT,
        )
        self.assertEqual(
            translate_common.translate_sensor_mode_to_model(common.SensorModeType.MEASUREMENT),
            common_types.SensorModeType.MEASUREMENT,
        )
        self.assertEqual(
            translate_common.translate_sensor_mode(common_types.SensorModeType.RX_ONLY),
            common.SensorModeType.RX_ONLY,
        )
        self.assertEqual(
            translate_common.translate_sensor_mode_to_model(common.SensorModeType.RX_ONLY),
            common_types.SensorModeType.RX_ONLY,
        )
        self.assertEqual(
            translate_common.translate_sensor_mode(common_types.SensorModeType.EXTERNAL_CALIBRATION),
            common.SensorModeType.EXTERNAL_CALIBRATION,
        )
        self.assertEqual(
            translate_common.translate_sensor_mode_to_model(common.SensorModeType.EXTERNAL_CALIBRATION),
            common_types.SensorModeType.EXTERNAL_CALIBRATION,
        )

    def test_translate_orbit_pass(self):
        self.assertEqual(
            translate_common.translate_orbit_pass(common_types.OrbitPassType.ASCENDING),
            common.OrbitPassType.ASCENDING,
        )
        self.assertEqual(
            translate_common.translate_orbit_pass_to_model(common.OrbitPassType.ASCENDING),
            common_types.OrbitPassType.ASCENDING,
        )
        self.assertEqual(
            translate_common.translate_orbit_pass(common_types.OrbitPassType.DESCENDING),
            common.OrbitPassType.DESCENDING,
        )
        self.assertEqual(
            translate_common.translate_orbit_pass_to_model(common.OrbitPassType.DESCENDING),
            common_types.OrbitPassType.DESCENDING,
        )

    def test_translate_product_composition(self):
        self.assertEqual(
            translate_common.translate_product_composition(common_types.ProductCompositionType.NOMINAL),
            common.ProductCompositionType.NOMINAL,
        )
        self.assertEqual(
            translate_common.translate_product_composition_to_model(common.ProductCompositionType.NOMINAL),
            common_types.ProductCompositionType.NOMINAL,
        )
        self.assertEqual(
            translate_common.translate_product_composition(common_types.ProductCompositionType.MERGED),
            common.ProductCompositionType.MERGED,
        )
        self.assertEqual(
            translate_common.translate_product_composition_to_model(common.ProductCompositionType.MERGED),
            common_types.ProductCompositionType.MERGED,
        )
        self.assertEqual(
            translate_common.translate_product_composition(common_types.ProductCompositionType.PARTIAL),
            common.ProductCompositionType.PARTIAL,
        )
        self.assertEqual(
            translate_common.translate_product_composition_to_model(common.ProductCompositionType.PARTIAL),
            common_types.ProductCompositionType.PARTIAL,
        )
        self.assertEqual(
            translate_common.translate_product_composition(common_types.ProductCompositionType.INCOMPLETE),
            common.ProductCompositionType.INCOMPLETE,
        )
        self.assertEqual(
            translate_common.translate_product_composition_to_model(common.ProductCompositionType.INCOMPLETE),
            common_types.ProductCompositionType.INCOMPLETE,
        )
        self.assertEqual(
            translate_common.translate_product_composition(common_types.ProductCompositionType.NOT_FRAMED),
            common.ProductCompositionType.NOT_FRAMED,
        )
        self.assertEqual(
            translate_common.translate_product_composition_to_model(common.ProductCompositionType.NOT_FRAMED),
            common_types.ProductCompositionType.NOT_FRAMED,
        )

    def test_translate_state_type(self):
        model = common_types.StateType(
            azimuth_time="2021-01-01T00:00:00.000000",
            value=common_types.FloatWithUnit(value=3.2, units=common_types.UomType.S),
        )
        translated = PreciseDateTime.from_numeric_datetime(2021), 3.2
        self.assertEqual(translate_common.translate_state_type(model), translated)
        self.assertEqual(translate_common.translate_state_type_to_model(translated, common.UomType.S), model)

    def test_translate_data_format_mode(self):
        self.assertEqual(
            translate_common.translate_data_format_mode(common_types.DataFormatModeType.BAQ_4_BIT),
            common.DataFormatModeType.BAQ_4_BIT,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode_to_model(common.DataFormatModeType.BAQ_4_BIT),
            common_types.DataFormatModeType.BAQ_4_BIT,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode(common_types.DataFormatModeType.BAQ_5_BIT),
            common.DataFormatModeType.BAQ_5_BIT,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode_to_model(common.DataFormatModeType.BAQ_5_BIT),
            common_types.DataFormatModeType.BAQ_5_BIT,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode(common_types.DataFormatModeType.BAQ_6_BIT),
            common.DataFormatModeType.BAQ_6_BIT,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode_to_model(common.DataFormatModeType.BAQ_6_BIT),
            common_types.DataFormatModeType.BAQ_6_BIT,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode(common_types.DataFormatModeType.BYPASS),
            common.DataFormatModeType.BYPASS,
        )
        self.assertEqual(
            translate_common.translate_data_format_mode_to_model(common.DataFormatModeType.BYPASS),
            common_types.DataFormatModeType.BYPASS,
        )

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

    def test_translate_internal_calibrationsource(self):
        self.assertEqual(
            translate_common.translate_internal_calibration_source(
                common_types.InternalCalibrationSourceType.EXTRACTED
            ),
            common.InternalCalibrationSourceType.EXTRACTED,
        )
        self.assertEqual(
            translate_common.translate_internal_calibration_source_to_model(common.InternalCalibrationSourceType.MODEL),
            common_types.InternalCalibrationSourceType.MODEL,
        )
        self.assertEqual(
            translate_common.translate_internal_calibration_source(
                common_types.InternalCalibrationSourceType.EXTRACTED
            ),
            common.InternalCalibrationSourceType.EXTRACTED,
        )
        self.assertEqual(
            translate_common.translate_internal_calibration_source_to_model(common.InternalCalibrationSourceType.MODEL),
            common_types.InternalCalibrationSourceType.MODEL,
        )

    def test_translate_height_model_base(self):
        self.assertEqual(
            translate_common.translate_height_model_base(common_types.HeightModelBaseType.ELLIPSOID),
            common.HeightModelBaseType.ELLIPSOID,
        )
        self.assertEqual(
            translate_common.translate_height_model_base_to_model(common.HeightModelBaseType.ELLIPSOID),
            common_types.HeightModelBaseType.ELLIPSOID,
        )
        self.assertEqual(
            translate_common.translate_height_model_base(common_types.HeightModelBaseType.COPERNICUS_DEM),
            common.HeightModelBaseType.COPERNICUS_DEM,
        )
        self.assertEqual(
            translate_common.translate_height_model_base_to_model(common.HeightModelBaseType.COPERNICUS_DEM),
            common_types.HeightModelBaseType.COPERNICUS_DEM,
        )
        self.assertEqual(
            translate_common.translate_height_model_base(common_types.HeightModelBaseType.SRTM),
            common.HeightModelBaseType.SRTM,
        )
        self.assertEqual(
            translate_common.translate_height_model_base_to_model(common.HeightModelBaseType.SRTM),
            common_types.HeightModelBaseType.SRTM,
        )
        self.assertEqual(
            translate_common.translate_height_model_base(common_types.HeightModelBaseType.BIOMASS_DTM),
            common.HeightModelBaseType.BIOMASS_DTM,
        )
        self.assertEqual(
            translate_common.translate_height_model_base_to_model(common.HeightModelBaseType.BIOMASS_DTM),
            common_types.HeightModelBaseType.BIOMASS_DTM,
        )

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

    def test_translate_layer(self):
        """Translate layer"""
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.FARADAY_ROTATION_PLANE_RAD),
            common.LayerType.FARADAY_ROTATION_PLANE_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.FARADAY_ROTATION_RAD),
            common.LayerType.FARADAY_ROTATION_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.FARADAY_ROTATION_STD_RAD),
            common.LayerType.FARADAY_ROTATION_STD_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.PHASE_SCREEN_RAD),
            common.LayerType.PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.TEC_TECU),
            common.LayerType.TEC_TECU,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RANGE_SHIFTS_M),
            common.LayerType.RANGE_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.AZIMUTH_SHIFTS_M),
            common.LayerType.AZIMUTH_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.AUTOFOCUS_PHASE_SCREEN_RAD),
            common.LayerType.AUTOFOCUS_PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.AUTOFOCUS_PHASE_SCREEN_STD_RAD),
            common.LayerType.AUTOFOCUS_PHASE_SCREEN_STD_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_TIME_DOMAIN_MASK_HH),
            common.LayerType.RFI_TIME_DOMAIN_MASK_HH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_TIME_DOMAIN_MASK_HV),
            common.LayerType.RFI_TIME_DOMAIN_MASK_HV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_TIME_DOMAIN_MASK_VH),
            common.LayerType.RFI_TIME_DOMAIN_MASK_VH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_TIME_DOMAIN_MASK_VV),
            common.LayerType.RFI_TIME_DOMAIN_MASK_VV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HH),
            common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HV),
            common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VH),
            common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VV),
            common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.SIGMA_NOUGHT_LUT),
            common.LayerType.SIGMA_NOUGHT_LUT,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.GAMMA_NOUGHT_LUT),
            common.LayerType.GAMMA_NOUGHT_LUT,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.DENOISING_MAP_HH),
            common.LayerType.DENOISING_MAP_HH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.DENOISING_MAP_HV),
            common.LayerType.DENOISING_MAP_HV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.DENOISING_MAP_XX),
            common.LayerType.DENOISING_MAP_XX,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.DENOISING_MAP_VH),
            common.LayerType.DENOISING_MAP_VH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.DENOISING_MAP_VV),
            common.LayerType.DENOISING_MAP_VV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.LATITUDE_DEG),
            common.LayerType.LATITUDE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.LONGITUDE_DEG),
            common.LayerType.LONGITUDE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.HEIGHT_M),
            common.LayerType.HEIGHT_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.INCIDENCE_ANGLE_DEG),
            common.LayerType.INCIDENCE_ANGLE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.ELEVATION_ANGLE_DEG),
            common.LayerType.ELEVATION_ANGLE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RANGE_TERRAIN_SLOPE_DEG),
            common.LayerType.RANGE_TERRAIN_SLOPE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.AZIMUTH_TERRAIN_SLOPE_DEG),
            common.LayerType.AZIMUTH_TERRAIN_SLOPE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.FNF),
            common.LayerType.FNF,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.ACM),
            common.LayerType.ACM,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.NUMBER_OF_AVERAGES),
            common.LayerType.NUMBER_OF_AVERAGES,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.AZIMUTH_COREGISTRATION_SHIFTS_M),
            common.LayerType.AZIMUTH_COREGISTRATION_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.RANGE_COREGISTRATION_SHIFTS_M),
            common.LayerType.RANGE_COREGISTRATION_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.COREGISTRATION_SHIFTS_QUALITY),
            common.LayerType.COREGISTRATION_SHIFTS_QUALITY,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.WAVENUMBERS_RAD_M),
            common.LayerType.WAVENUMBERS_RAD_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.FLATTENING_PHASE_SCREEN_RAD),
            common.LayerType.FLATTENING_PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.SKP_CALIBRATION_PHASE_SCREEN_RAD),
            common.LayerType.SKP_CALIBRATION_PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer(common_types.LayerType.SKP_CALIBRATION_PHASE_SCREEN_QUALITY),
            common.LayerType.SKP_CALIBRATION_PHASE_SCREEN_QUALITY,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.FARADAY_ROTATION_PLANE_RAD),
            common_types.LayerType.FARADAY_ROTATION_PLANE_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.FARADAY_ROTATION_RAD),
            common_types.LayerType.FARADAY_ROTATION_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.FARADAY_ROTATION_STD_RAD),
            common_types.LayerType.FARADAY_ROTATION_STD_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.PHASE_SCREEN_RAD),
            common_types.LayerType.PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.TEC_TECU),
            common_types.LayerType.TEC_TECU,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RANGE_SHIFTS_M),
            common_types.LayerType.RANGE_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.AZIMUTH_SHIFTS_M),
            common_types.LayerType.AZIMUTH_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.AUTOFOCUS_PHASE_SCREEN_RAD),
            common_types.LayerType.AUTOFOCUS_PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.AUTOFOCUS_PHASE_SCREEN_STD_RAD),
            common_types.LayerType.AUTOFOCUS_PHASE_SCREEN_STD_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_TIME_DOMAIN_MASK_HH),
            common_types.LayerType.RFI_TIME_DOMAIN_MASK_HH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_TIME_DOMAIN_MASK_HV),
            common_types.LayerType.RFI_TIME_DOMAIN_MASK_HV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_TIME_DOMAIN_MASK_VH),
            common_types.LayerType.RFI_TIME_DOMAIN_MASK_VH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_TIME_DOMAIN_MASK_VV),
            common_types.LayerType.RFI_TIME_DOMAIN_MASK_VV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HH),
            common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HV),
            common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VH),
            common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VV),
            common_types.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.SIGMA_NOUGHT_LUT),
            common_types.LayerType.SIGMA_NOUGHT_LUT,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.GAMMA_NOUGHT_LUT),
            common_types.LayerType.GAMMA_NOUGHT_LUT,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.DENOISING_MAP_HH),
            common_types.LayerType.DENOISING_MAP_HH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.DENOISING_MAP_HV),
            common_types.LayerType.DENOISING_MAP_HV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.DENOISING_MAP_XX),
            common_types.LayerType.DENOISING_MAP_XX,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.DENOISING_MAP_VH),
            common_types.LayerType.DENOISING_MAP_VH,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.DENOISING_MAP_VV),
            common_types.LayerType.DENOISING_MAP_VV,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.LATITUDE_DEG),
            common_types.LayerType.LATITUDE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.LONGITUDE_DEG),
            common_types.LayerType.LONGITUDE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.HEIGHT_M),
            common_types.LayerType.HEIGHT_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.INCIDENCE_ANGLE_DEG),
            common_types.LayerType.INCIDENCE_ANGLE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.ELEVATION_ANGLE_DEG),
            common_types.LayerType.ELEVATION_ANGLE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RANGE_TERRAIN_SLOPE_DEG),
            common_types.LayerType.RANGE_TERRAIN_SLOPE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.AZIMUTH_TERRAIN_SLOPE_DEG),
            common_types.LayerType.AZIMUTH_TERRAIN_SLOPE_DEG,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.FNF),
            common_types.LayerType.FNF,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.ACM),
            common_types.LayerType.ACM,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.NUMBER_OF_AVERAGES),
            common_types.LayerType.NUMBER_OF_AVERAGES,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.AZIMUTH_COREGISTRATION_SHIFTS_M),
            common_types.LayerType.AZIMUTH_COREGISTRATION_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.RANGE_COREGISTRATION_SHIFTS_M),
            common_types.LayerType.RANGE_COREGISTRATION_SHIFTS_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.COREGISTRATION_SHIFTS_QUALITY),
            common_types.LayerType.COREGISTRATION_SHIFTS_QUALITY,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.WAVENUMBERS_RAD_M),
            common_types.LayerType.WAVENUMBERS_RAD_M,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.FLATTENING_PHASE_SCREEN_RAD),
            common_types.LayerType.FLATTENING_PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.SKP_CALIBRATION_PHASE_SCREEN_RAD),
            common_types.LayerType.SKP_CALIBRATION_PHASE_SCREEN_RAD,
        )
        self.assertEqual(
            translate_common.translate_lut_layer_to_model(common.LayerType.SKP_CALIBRATION_PHASE_SCREEN_QUALITY),
            common_types.LayerType.SKP_CALIBRATION_PHASE_SCREEN_QUALITY,
        )

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
        self.assertEqual(
            translate_common.translate_polarisation_combination_method(
                common_types.PolarisationCombinationMethodType.AVERAGE
            ),
            common.PolarisationCombinationMethodType.AVERAGE,
        )
        self.assertEqual(
            translate_common.translate_polarisation_combination_method(
                common_types.PolarisationCombinationMethodType.HV
            ),
            common.PolarisationCombinationMethodType.HV,
        )
        self.assertEqual(
            translate_common.translate_polarisation_combination_method(
                common_types.PolarisationCombinationMethodType.VH
            ),
            common.PolarisationCombinationMethodType.VH,
        )
        self.assertEqual(
            translate_common.translate_polarisation_combination_method(
                common_types.PolarisationCombinationMethodType.NONE
            ),
            None,
        )

        self.assertEqual(
            translate_common.translate_polarisation_combination_method_to_model(
                common.PolarisationCombinationMethodType.AVERAGE
            ),
            common_types.PolarisationCombinationMethodType.AVERAGE,
        )
        self.assertEqual(
            translate_common.translate_polarisation_combination_method_to_model(
                common.PolarisationCombinationMethodType.HV
            ),
            common_types.PolarisationCombinationMethodType.HV,
        )
        self.assertEqual(
            translate_common.translate_polarisation_combination_method_to_model(
                common.PolarisationCombinationMethodType.VH
            ),
            common_types.PolarisationCombinationMethodType.VH,
        )
        self.assertEqual(
            translate_common.translate_polarisation_combination_method_to_model(None),
            common_types.PolarisationCombinationMethodType.NONE,
        )

    def test_translate_primary_image_selection_information(self):
        self.assertEqual(
            translate_common.translate_primary_image_selection_information(
                common_types.PrimaryImageSelectionInformationType.GEOMETRY
            ),
            common.PrimaryImageSelectionInformationType.GEOMETRY,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information(
                common_types.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION
            ),
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information(
                common_types.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION
            ),
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information(
                common_types.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS
            ),
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information(
                common_types.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE
            ),
            common.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE,
        )

        self.assertEqual(
            translate_common.translate_primary_image_selection_information_to_model(
                common.PrimaryImageSelectionInformationType.GEOMETRY,
            ),
            common_types.PrimaryImageSelectionInformationType.GEOMETRY,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information_to_model(
                common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
            ),
            common_types.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information_to_model(
                common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
            ),
            common_types.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information_to_model(
                common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
            ),
            common_types.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
        )
        self.assertEqual(
            translate_common.translate_primary_image_selection_information_to_model(
                common.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE,
            ),
            common_types.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE,
        )

    def test_translate_coregistration_method(self):
        self.assertEqual(
            translate_common.translate_coregistration_method(common_types.CoregistrationMethodType.GEOMETRY),
            common.CoregistrationMethodType.GEOMETRY,
        )
        self.assertEqual(
            translate_common.translate_coregistration_method(common_types.CoregistrationMethodType.GEOMETRY_AND_DATA),
            common.CoregistrationMethodType.GEOMETRY_AND_DATA,
        )
        self.assertEqual(
            translate_common.translate_coregistration_method(common_types.CoregistrationMethodType.AUTOMATIC),
            common.CoregistrationMethodType.AUTOMATIC,
        )

        self.assertEqual(
            translate_common.translate_coregistration_method_to_model(
                common.CoregistrationMethodType.GEOMETRY,
            ),
            common_types.CoregistrationMethodType.GEOMETRY,
        )
        self.assertEqual(
            translate_common.translate_coregistration_method_to_model(
                common.CoregistrationMethodType.GEOMETRY_AND_DATA,
            ),
            common_types.CoregistrationMethodType.GEOMETRY_AND_DATA,
        )
        self.assertEqual(
            translate_common.translate_coregistration_method_to_model(
                common.CoregistrationMethodType.AUTOMATIC,
            ),
            common_types.CoregistrationMethodType.AUTOMATIC,
        )

    def test_translate_coregistration_execution_policy(self):
        self.assertEqual(
            translate_common.translate_coregistration_execution_policy(
                common_types.CoregistrationExecutionPolicyType.NOMINAL
            ),
            common.CoregistrationExecutionPolicyType.NOMINAL,
        )
        self.assertEqual(
            translate_common.translate_coregistration_execution_policy(
                common_types.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY
            ),
            common.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY,
        )
        self.assertEqual(
            translate_common.translate_coregistration_execution_policy(
                common_types.CoregistrationExecutionPolicyType.WARPING_ONLY
            ),
            common.CoregistrationExecutionPolicyType.WARPING_ONLY,
        )

        self.assertEqual(
            translate_common.translate_coregistration_execution_policy_to_model(
                common.CoregistrationExecutionPolicyType.NOMINAL,
            ),
            common_types.CoregistrationExecutionPolicyType.NOMINAL,
        )
        self.assertEqual(
            translate_common.translate_coregistration_execution_policy_to_model(
                common.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY,
            ),
            common_types.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY,
        )
        self.assertEqual(
            translate_common.translate_coregistration_execution_policy_to_model(
                common.CoregistrationExecutionPolicyType.WARPING_ONLY,
            ),
            common_types.CoregistrationExecutionPolicyType.WARPING_ONLY,
        )


if __name__ == "__main__":
    unittest.main()
