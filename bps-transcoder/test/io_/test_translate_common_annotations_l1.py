# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common annotations l1 translate test
------------------------------------
"""

import copy
import unittest

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.transcoder.io import common_annotation_l1, translate_common_annotation_l1
from bps.transcoder.io.v445 import common as common_v445
from bps.transcoder.io.v445 import common_annotation_models_l1 as v445
from bps.transcoder.io.v450 import common as common_v450
from bps.transcoder.io.v450 import common_annotation_models_l1 as v450

VERSIONS = ((v445, common_v445), (v450, common_v450))
LATEST_VERSION = v450
LATEST_COMMON_VERSION = common_v450


class CommonAnnotationL1TranslateTestCase(unittest.TestCase):
    """Common annotation"""

    def test_translate_noise_gain_list(self):
        """Noise gain translation"""
        translated = [(3.0, common.PolarisationType.HH), (5.0, common.PolarisationType.HV)]
        for version, common_version in VERSIONS:
            model = version.NoiseGainListType(
                noise_gain=[
                    common_version.FloatWithPolarisation(value=3, polarisation=common_version.PolarisationType.HH),
                    common_version.FloatWithPolarisation(value=5, polarisation=common_version.PolarisationType.HV),
                ],
                count=2,
            )
            assert translate_common_annotation_l1.translate_noise_gain_list(model) == translated

        assert translate_common_annotation_l1.translate_noise_gain_list_to_model(
            translated
        ) == LATEST_VERSION.NoiseGainListType(
            noise_gain=[
                LATEST_COMMON_VERSION.FloatWithPolarisation(
                    value=3, polarisation=LATEST_COMMON_VERSION.PolarisationType.HH
                ),
                LATEST_COMMON_VERSION.FloatWithPolarisation(
                    value=5, polarisation=LATEST_COMMON_VERSION.PolarisationType.HV
                ),
            ],
            count=2,
        )

    def test_translate_processing_gain_list(self):
        """Processing gain translation"""
        translated = [(3.0, common.PolarisationType.HH), (5.0, common.PolarisationType.HV)]

        for version, common_version in VERSIONS:
            model = version.ProcessingGainListType(
                processing_gain=[
                    common_version.FloatWithPolarisation(value=3, polarisation=common_version.PolarisationType.HH),
                    common_version.FloatWithPolarisation(value=5, polarisation=common_version.PolarisationType.HV),
                ],
                count=2,
            )

            assert translate_common_annotation_l1.translate_processing_gain_list(model) == translated
        assert translate_common_annotation_l1.translate_processing_gain_list_to_model(
            translated
        ) == LATEST_VERSION.ProcessingGainListType(
            processing_gain=[
                LATEST_COMMON_VERSION.FloatWithPolarisation(
                    value=3, polarisation=LATEST_COMMON_VERSION.PolarisationType.HH
                ),
                LATEST_COMMON_VERSION.FloatWithPolarisation(
                    value=5, polarisation=LATEST_COMMON_VERSION.PolarisationType.HV
                ),
            ],
            count=2,
        )

    def test_translate_spectrum_processing_parameters_type(self):
        """Transalte spectrum proc params"""
        translated = common_annotation_l1.SpectrumProcessingParametersType(
            window_type=common.WeightingWindowType.HAMMING,
            window_coefficient=0.2,
            total_bandwidth=1.2,
            processing_bandwidth=1.1,
            look_bandwidth=0.9,
            number_of_looks=3,
            look_overlap=0.5,
        )
        for version, common_version in VERSIONS:
            model = version.SpectrumProcessingParametersType(
                window_type=version.WeightingWindowType.HAMMING,
                window_coefficient=0.2,
                total_bandwidth=common_version.DoubleWithUnit(value=1.2, units=common_version.UomType.HZ),
                processing_bandwidth=common_version.DoubleWithUnit(value=1.1, units=common_version.UomType.HZ),
                look_bandwidth=common_version.DoubleWithUnit(value=0.9, units=common_version.UomType.HZ),
                number_of_looks=3,
                look_overlap=common_version.DoubleWithUnit(value=0.5, units=common_version.UomType.HZ),
            )

            assert translate_common_annotation_l1.translate_spectrum_processing_parameters_type(model) == translated
        assert translate_common_annotation_l1.translate_spectrum_processing_parameters_type_to_model(
            translated
        ) == LATEST_VERSION.SpectrumProcessingParametersType(
            window_type=LATEST_COMMON_VERSION.WeightingWindowType.HAMMING,
            window_coefficient=0.2,
            total_bandwidth=LATEST_COMMON_VERSION.DoubleWithUnit(value=1.2, units=LATEST_COMMON_VERSION.UomType.HZ),
            processing_bandwidth=LATEST_COMMON_VERSION.DoubleWithUnit(
                value=1.1, units=LATEST_COMMON_VERSION.UomType.HZ
            ),
            look_bandwidth=LATEST_COMMON_VERSION.DoubleWithUnit(value=0.9, units=LATEST_COMMON_VERSION.UomType.HZ),
            number_of_looks=3,
            look_overlap=LATEST_COMMON_VERSION.DoubleWithUnit(value=0.5, units=LATEST_COMMON_VERSION.UomType.HZ),
        )

    # TODO update all following tests to test all versions

    def test_translate_processing_parameters(self):
        az_param_model = v450.SpectrumProcessingParametersType(
            window_type=v450.WeightingWindowType.HAMMING,
            window_coefficient=0.2,
            total_bandwidth=common_v450.DoubleWithUnit(value=1.2, units=common_v450.UomType.HZ),
            processing_bandwidth=common_v450.DoubleWithUnit(value=1.1, units=common_v450.UomType.HZ),
            look_bandwidth=common_v450.DoubleWithUnit(value=0.9, units=common_v450.UomType.HZ),
            number_of_looks=3,
            look_overlap=common_v450.DoubleWithUnit(value=0.5, units=common_v450.UomType.HZ),
        )
        az_param_translated = common_annotation_l1.SpectrumProcessingParametersType(
            window_type=common.WeightingWindowType.HAMMING,
            window_coefficient=0.2,
            total_bandwidth=1.2,
            processing_bandwidth=1.1,
            look_bandwidth=0.9,
            number_of_looks=3,
            look_overlap=0.5,
        )
        rg_param_model = copy.copy(az_param_model)
        rg_param_model.look_bandwidth = common_v450.DoubleWithUnit(value=0.5, units=common_v450.UomType.HZ)
        rg_param_translated = copy.copy(az_param_translated)
        rg_param_translated.look_bandwidth = 0.5

        model = v450.ProcessingParametersType(
            processor_version="version",
            product_generation_time="2020-01-01T00:00:00.000000",
            processing_mode=common_v450.ProcessingModeType.NOMINAL,
            orbit_source=common_v450.OrbitAttitudeSourceType.AUXILIARY,
            attitude_source=common_v450.OrbitAttitudeSourceType.AUXILIARY,
            raw_data_correction_flag="true",
            rfi_detection_flag="true",
            rfi_correction_flag="true",
            rfi_mitigation_method=common_v450.RfiMitigationMethodType.FREQUENCY,
            rfi_mask=common_v450.RfiMaskType.MULTIPLE,
            rfi_mask_generation_method=common_v450.RfiMaskGenerationMethodType.AND,
            rfi_fmmitigation_method=common_v450.RfiFmmitigationMethodType.NEAREST_NEIGHBOUR_INTERPOLATION,
            rfi_fmchirp_source=common_v450.RangeReferenceFunctionType.REPLICA,
            internal_calibration_estimation_flag="true",
            internal_calibration_correction_flag="true",
            range_reference_function_source=common_v450.RangeReferenceFunctionType.REPLICA,
            range_compression_method=common_v450.RangeCompressionMethodType.INVERSE_FILTER,
            extended_swath_processing_flag="false",
            dc_method=common_v450.DcMethodType.COMBINED,
            dc_value=common_v450.DoubleWithUnit(value=0.2, units=common_v450.UomType.HZ),
            antenna_pattern_correction1_flag="true",
            antenna_pattern_correction2_flag="true",
            antenna_cross_talk_correction_flag="true",
            range_processing_parameters=rg_param_model,
            azimuth_processing_parameters=az_param_model,
            bistatic_delay_correction_flag="true",
            bistatic_delay_correction_method=common_v450.BistaticDelayCorrectionMethodType.BULK,
            range_spreading_loss_compensation_flag="false",
            processing_gain_list=v450.ProcessingGainListType(
                count=1,
                processing_gain=[
                    common_v450.FloatWithPolarisation(value=1.0, polarisation=common_v450.PolarisationType.HH)
                ],
            ),
            polarimetric_correction_flag="false",
            ionosphere_height_defocusing_flag="false",
            ionosphere_height_estimation_method=common_v450.IonosphereHeightEstimationMethodType.FEATURE_TRACKING,
            faraday_rotation_correction_flag="false",
            ionospheric_phase_screen_correction_flag="false",
            group_delay_correction_flag="false",
            reference_range=common_v450.DoubleWithUnit(value=800000, units=common_v450.UomType.M),
            autofocus_flag="true",
            autofocus_method=common_v450.AutofocusMethodType.MAP_DRIFT,
            detection_flag="true",
            thermal_denoising_flag="false",
            noise_gain_list=v450.NoiseGainListType(
                count=1,
                noise_gain=[common_v450.FloatWithPolarisation(value=2.0, polarisation=common_v450.PolarisationType.HV)],
            ),
            ground_projection_flag="false",
        )

        translated = common_annotation_l1.ProcessingParameters(
            processor_version="version",
            product_generation_time=PreciseDateTime.from_numeric_datetime(
                year=2020,
            ),
            processing_mode=common.ProcessingModeType.NOMINAL,
            orbit_source=common.OrbitAttitudeSourceType.AUXILIARY,
            attitude_source=common.OrbitAttitudeSourceType.AUXILIARY,
            raw_data_correction_flag=True,
            rfi_detection_flag=True,
            rfi_correction_flag=True,
            rfi_mitigation_method=common.RfiMitigationMethodType.FREQUENCY,
            rfi_mask=common.RfiMaskType.MULTIPLE,
            rfi_mask_generation_method=common.RfiMaskGenerationMethodType.AND,
            rfi_fm_mitigation_method=common.RfiFmmitigationMethodType.NEAREST_NEIGHBOUR_INTERPOLATION,
            rfi_fm_chirp_source=common.RangeReferenceFunctionType.REPLICA,
            internal_calibration_estimation_flag=True,
            internal_calibration_correction_flag=True,
            range_reference_function_source=common.RangeReferenceFunctionType.REPLICA,
            range_compression_method=common.RangeCompressionMethodType.INVERSE_FILTER,
            extended_swath_processing_flag=False,
            dc_method=common.DcMethodType.COMBINED,
            dc_value=0.2,
            antenna_pattern_correction1_flag=True,
            antenna_pattern_correction2_flag=True,
            antenna_cross_talk_correction_flag=True,
            range_processing_parameters=rg_param_translated,
            azimuth_processing_parameters=az_param_translated,
            bistatic_delay_correction_flag=True,
            bistatic_delay_correction_method=common.BistaticDelayCorrectionMethodType.BULK,
            range_spreading_loss_compensation_flag=False,
            processing_gain_list=[(1.0, common.PolarisationType.HH)],
            polarimetric_correction_flag=False,
            ionosphere_height_defocusing_flag=False,
            ionosphere_height_estimation_method=common.IonosphereHeightEstimationMethodType.FEATURE_TRACKING,
            faraday_rotation_correction_flag=False,
            ionospheric_phase_screen_correction_flag=False,
            group_delay_correction_flag=False,
            reference_range=800000,
            autofocus_flag=True,
            autofocus_method=common.AutofocusMethodType.MAP_DRIFT,
            detection_flag=True,
            thermal_denoising_flag=False,
            noise_gain_list=[(2.0, common.PolarisationType.HV)],
            ground_projection_flag=False,
        )

        assert translate_common_annotation_l1.translate_processing_parameters(model) == translated
        assert translate_common_annotation_l1.translate_processing_parameters_to_model(translated) == model

    def test_translate_dc_estimate_type(self):
        model = v450.DcEstimateType(
            azimuth_time="2020-01-01T00:00:00.000000",
            t0=common_v450.DoubleWithUnit(value=0.5, units=common_v450.UomType.S),
            geometry_dcpolynomial=common_v450.FloatArray(value="8.0", count=1),
            combined_dcpolynomial=common_v450.FloatArray(value="", count=0),
            combined_dcvalues=common_v450.DoubleArrayWithUnits(value="1.0 5.0", count=2, units=common_v450.UomType.HZ),
            combined_dcslant_range_times=common_v450.DoubleArrayWithUnits(
                value="0.1 0.5", count=2, units=common_v450.UomType.S
            ),
            combined_dcrmserror=common_v450.DoubleWithUnit(value=10, units=common_v450.UomType.HZ),
            combined_dcrmserror_above_threshold="false",
        )

        translated = common_annotation_l1.DcEstimateType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2020),
            t0=0.5,
            geometry_dcpolynomial=[8],
            combined_dcpolynomial=[],
            combined_dcvalues=[1, 5],
            combined_dcslant_range_times=[0.1, 0.5],
            combined_dcrmserror=10,
            combined_dcrmserror_above_threshold=False,
        )

        assert translate_common_annotation_l1.translate_dc_estimate_type(model) == translated
        assert translate_common_annotation_l1.translate_dc_estimate_type_to_model(translated) == model

    def test_translate_dc_estimate_list_type(self):
        model_1 = v450.DcEstimateType(
            azimuth_time="2020-01-01T00:00:00.000000",
            t0=common_v450.DoubleWithUnit(value=0.5, units=common_v450.UomType.S),
            geometry_dcpolynomial=common_v450.FloatArray(value="8.0", count=1),
            combined_dcpolynomial=common_v450.FloatArray(value="", count=0),
            combined_dcvalues=common_v450.DoubleArrayWithUnits(value="1.0 5.0", count=2, units=common_v450.UomType.HZ),
            combined_dcslant_range_times=common_v450.DoubleArrayWithUnits(
                value="0.1 0.5", count=2, units=common_v450.UomType.S
            ),
            combined_dcrmserror=common_v450.DoubleWithUnit(value=10, units=common_v450.UomType.HZ),
            combined_dcrmserror_above_threshold="false",
        )

        model_2 = copy.copy(model_1)
        model_2.combined_dcrmserror_above_threshold = "true"

        translated_1 = common_annotation_l1.DcEstimateType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2020),
            t0=0.5,
            geometry_dcpolynomial=[8],
            combined_dcpolynomial=[],
            combined_dcvalues=[1, 5],
            combined_dcslant_range_times=[0.1, 0.5],
            combined_dcrmserror=10,
            combined_dcrmserror_above_threshold=False,
        )
        translated_2 = copy.copy(translated_1)
        translated_2.combined_dcrmserror_above_threshold = True

        model = v450.DcEstimateListType(dc_estimate=[model_1, model_2], count=2)
        translated = [translated_1, translated_2]

        self.assertEqual(translate_common_annotation_l1.translate_dc_estimate_list_type(model), translated)
        assert translate_common_annotation_l1.translate_dc_estimate_list_type_to_model(translated) == model

        assert (
            translate_common_annotation_l1.translate_dc_estimate_list_type(
                v450.DcEstimateListType(dc_estimate=[], count=0)
            )
            == []
        )
        assert translate_common_annotation_l1.translate_dc_estimate_list_type_to_model([]) == v450.DcEstimateListType(
            dc_estimate=[], count=0
        )

    def test_translate_rfi_isolated_fmreport_type(self):
        """Translate rfi isolated fm report"""
        model = v450.RfiIsolatedFmreportType(
            percentage_affected_lines=0.2,
            max_percentage_affected_bw=0.8,
            avg_percentage_affected_bw=0.0,
            polarisation=common_v450.PolarisationType.HH,
        )
        translated = common_annotation_l1.RfiIsolatedFmreportType(
            percentage_affected_lines=0.2,
            max_percentage_affected_bw=0.8,
            avg_percentage_affected_bw=0.0,
            polarisation=common.PolarisationType.HH,
        )
        assert translate_common_annotation_l1.translate_rfi_isolated_fmreport_type(model) == translated
        assert translate_common_annotation_l1.translate_rfi_isolated_fmreport_type_to_model(translated) == model

    def test_translate_rfi_isolated_fmreport_type_list(self):
        """Translate rfi isolated fm report"""
        model_1 = v450.RfiIsolatedFmreportType(
            percentage_affected_lines=0.2,
            max_percentage_affected_bw=0.8,
            avg_percentage_affected_bw=0.0,
            polarisation=common_v450.PolarisationType.HH,
        )
        translated_1 = common_annotation_l1.RfiIsolatedFmreportType(
            percentage_affected_lines=0.2,
            max_percentage_affected_bw=0.8,
            avg_percentage_affected_bw=0.0,
            polarisation=common.PolarisationType.HH,
        )
        model_2 = v450.RfiIsolatedFmreportType(
            percentage_affected_lines=0.1,
            max_percentage_affected_bw=0.1,
            avg_percentage_affected_bw=0.0,
            polarisation=common_v450.PolarisationType.HV,
        )
        translated_2 = common_annotation_l1.RfiIsolatedFmreportType(
            percentage_affected_lines=0.1,
            max_percentage_affected_bw=0.1,
            avg_percentage_affected_bw=0.0,
            polarisation=common.PolarisationType.HV,
        )
        model = v450.RfiIsolatedFmreportListType(rfi_isolated_fmreport=[model_1, model_2], count=2)
        translated = [translated_1, translated_2]

        assert translate_common_annotation_l1.translate_rfi_isolated_fmreport_list_type(model) == translated
        assert translate_common_annotation_l1.translate_rfi_isolated_fmreport_list_type_to_model(translated) == model

        assert (
            translate_common_annotation_l1.translate_rfi_isolated_fmreport_list_type(
                v450.RfiIsolatedFmreportListType(rfi_isolated_fmreport=[], count=0)
            )
            == []
        )
        assert translate_common_annotation_l1.translate_rfi_isolated_fmreport_list_type_to_model(
            []
        ) == v450.RfiIsolatedFmreportListType(rfi_isolated_fmreport=[], count=0)

    def test_translate_coordinate_conversions(self):
        translated = common_annotation_l1.CoordinateConversionType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(year=2020),
            t0=0.3,
            sr0=200.3,
            slant_to_ground_coefficients=[1.2, 3.5],
            ground_to_slant_coefficients=[],
            gr0=400.4,
        )
        model = v450.CoordinateConversionType(
            azimuth_time="2020-01-01T00:00:00.000000",
            t0=common_v450.DoubleWithUnit(value=0.3, units=common_v450.UomType.S),
            sr0=common_v450.DoubleWithUnit(value=200.3, units=common_v450.UomType.M),
            slant_to_ground_coefficients=common_v450.DoubleArray(value="1.2 3.5", count=2),
            ground_to_slant_coefficients=common_v450.DoubleArray(value="", count=0),
            gr0=common_v450.DoubleWithUnit(value=400.4, units=common_v450.UomType.M),
        )
        self.assertEqual(translate_common_annotation_l1.translate_coordinate_conversion_type(model), translated)
        self.assertEqual(
            translate_common_annotation_l1.translate_coordinate_conversion_type_to_model(translated), model
        )

    def test_translate_polarisation_list(self):
        translated = [common.PolarisationType.HH, common.PolarisationType.HV]
        model = v450.PolarisationListType(
            polarisation=[
                common_v450.PolarisationType.HH,
                common_v450.PolarisationType.HV,
            ],
            count=2,
        )

        self.assertEqual(translate_common_annotation_l1.translate_polarisation_list(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_polarisation_list_to_model(translated), model)

        translated = []
        model = v450.PolarisationListType(
            polarisation=[],
            count=0,
        )

        self.assertEqual(translate_common_annotation_l1.translate_polarisation_list(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_polarisation_list_to_model(translated), model)

    def test_translate_acquisition_information(self):
        translated = common_annotation_l1.AcquisitionInformationType(
            mission=common.MissionType.BIOMASS,
            swath=common.SwathType.S1,
            product_type=common.ProductType.DGM,
            polarisation_list=[common.PolarisationType.HH],
            start_time=PreciseDateTime.from_numeric_datetime(2021),
            stop_time=PreciseDateTime.from_numeric_datetime(2022),
            mission_phase_id=common.MissionPhaseIdtype.COM,
            drift_phase_flag=True,
            sensor_mode=common.SensorModeType.MEASUREMENT,
            global_coverage_id=2,
            major_cycle_id=3,
            repeat_cycle_id=4,
            absolute_orbit_number=99,
            relative_orbit_number=102,
            orbit_pass=common.OrbitPassType.DESCENDING,
            platform_heading=0.2,
            data_take_id=888,
            frame=5,
            product_composition=common.ProductCompositionType.INCOMPLETE,
        )
        model = v450.AcquisitionInformationType(
            mission=common_v450.MissionType.BIOMASS,
            swath=common_v450.SwathType.S1,
            product_type=common_v450.ProductType.DGM,
            polarisation_list=v450.PolarisationListType(polarisation=[common_v450.PolarisationType.HH], count=1),
            start_time="2021-01-01T00:00:00.000000",
            stop_time="2022-01-01T00:00:00.000000",
            mission_phase_id=common_v450.MissionPhaseIdtype.COM,
            drift_phase_flag="true",
            sensor_mode=common_v450.SensorModeType.MEASUREMENT,
            global_coverage_id=2,
            major_cycle_id=3,
            repeat_cycle_id=4,
            absolute_orbit_number=99,
            relative_orbit_number=102,
            orbit_pass=common_v450.OrbitPassType.DESCENDING,
            platform_heading=common_v450.DoubleWithUnit(value=0.2, units=common_v450.UomType.DEG),
            data_take_id=888,
            frame=5,
            product_composition=common_v450.ProductCompositionType.INCOMPLETE,
        )
        self.assertEqual(translate_common_annotation_l1.translate_acquisition_information(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_acquisition_information_to_model(translated), model)

    def test_translate_sar_image(self):
        translated = common_annotation_l1.SarImageType(
            first_sample_slant_range_time=0.5,
            last_sample_slant_range_time=0.6,
            first_line_azimuth_time=PreciseDateTime.from_numeric_datetime(2020),
            last_line_azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
            range_time_interval=1.0,
            azimuth_time_interval=2.5,
            range_pixel_spacing=8.1,
            azimuth_pixel_spacing=9.2,
            number_of_samples=158,
            number_of_lines=898,
            projection=common.ProjectionType.GROUND_RANGE,
            range_coordinate_conversion=[
                common_annotation_l1.CoordinateConversionType(
                    azimuth_time=PreciseDateTime.from_numeric_datetime(year=2020),
                    t0=0.3,
                    sr0=200.3,
                    slant_to_ground_coefficients=[1.2, 3.5],
                    ground_to_slant_coefficients=[],
                    gr0=400.4,
                ),
                common_annotation_l1.CoordinateConversionType(
                    azimuth_time=PreciseDateTime.from_numeric_datetime(year=2021),
                    t0=2.3,
                    sr0=800.3,
                    slant_to_ground_coefficients=[],
                    ground_to_slant_coefficients=[1.2, 3.5],
                    gr0=21.4,
                ),
            ],
            datum=common.DatumType(
                coordinate_reference_system="EXAMPLE",
                geodetic_reference_frame=common.GeodeticReferenceFrameType.WGS84,
            ),
            footprint=[40.5, 23.1, 56.7, 32.2],
            pixel_representation=common.PixelRepresentationType.ABS_PHASE,
            pixel_type=common.PixelTypeType.VALUE_32_BIT_FLOAT,
            pixel_quantity=common.PixelQuantityType.SIGMA_NOUGHT,
            no_data_value=-89,
        )
        model = v450.SarImageType(
            first_sample_slant_range_time=common_v450.DoubleWithUnit(value=0.5, units=common_v450.UomType.S),
            last_sample_slant_range_time=common_v450.DoubleWithUnit(value=0.6, units=common_v450.UomType.S),
            first_line_azimuth_time="2020-01-01T00:00:00.000000",
            last_line_azimuth_time="2021-01-01T00:00:00.000000",
            range_time_interval=common_v450.DoubleWithUnit(value=1.0, units=common_v450.UomType.S),
            azimuth_time_interval=common_v450.DoubleWithUnit(value=2.5, units=common_v450.UomType.S),
            range_pixel_spacing=common_v450.FloatWithUnit(value=8.1, units=common_v450.UomType.M),
            azimuth_pixel_spacing=common_v450.FloatWithUnit(value=9.2, units=common_v450.UomType.M),
            number_of_samples=158,
            number_of_lines=898,
            projection=common_v450.ProjectionType.GROUND_RANGE,
            range_coordinate_conversion=v450.CoordinateConversionListType(
                coordinate_conversion=[
                    v450.CoordinateConversionType(
                        azimuth_time="2020-01-01T00:00:00.000000",
                        t0=common_v450.DoubleWithUnit(value=0.3, units=common_v450.UomType.S),
                        sr0=common_v450.DoubleWithUnit(value=200.3, units=common_v450.UomType.M),
                        slant_to_ground_coefficients=common_v450.DoubleArray(value="1.2 3.5", count=2),
                        ground_to_slant_coefficients=common_v450.DoubleArray(value="", count=0),
                        gr0=common_v450.DoubleWithUnit(value=400.4, units=common_v450.UomType.M),
                    ),
                    v450.CoordinateConversionType(
                        azimuth_time="2021-01-01T00:00:00.000000",
                        t0=common_v450.DoubleWithUnit(value=2.3, units=common_v450.UomType.S),
                        sr0=common_v450.DoubleWithUnit(value=800.3, units=common_v450.UomType.M),
                        slant_to_ground_coefficients=common_v450.DoubleArray(value="", count=0),
                        ground_to_slant_coefficients=common_v450.DoubleArray(value="1.2 3.5", count=2),
                        gr0=common_v450.DoubleWithUnit(value=21.4, units=common_v450.UomType.M),
                    ),
                ],
                count=2,
            ),
            datum=common_v450.DatumType(
                coordinate_reference_system="EXAMPLE",
                geodetic_reference_frame=common_v450.GeodeticReferenceFrameType.WGS84,
            ),
            footprint=common_v450.FloatArrayWithUnits(
                value="40.5 23.1 56.7 32.2", count=4, units=common_v450.UomType.DEG
            ),
            pixel_representation=common_v450.PixelRepresentationType.ABS_PHASE,
            pixel_type=common_v450.PixelTypeType.VALUE_32_BIT_FLOAT,
            pixel_quantity=common_v450.PixelQuantityType.SIGMA_NOUGHT,
            no_data_value=-89,
        )

        self.assertEqual(translate_common_annotation_l1.translate_sar_image(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_sar_image_to_model(translated), model)

    def test_translate_tx_pulse(self):
        model = v450.TxPulseType(
            azimuth_time="2021-01-01T00:00:00.000000",
            tx_pulse_length=common_v450.DoubleWithUnit(value=1.2, units=common_v450.UomType.S),
            tx_pulse_start_frequency=common_v450.DoubleWithUnit(value=2.3, units=common_v450.UomType.HZ),
            tx_pulse_start_phase=common_v450.DoubleWithUnit(value=4.5, units=common_v450.UomType.RAD),
            tx_pulse_ramp_rate=common_v450.DoubleWithUnit(value=5.6, units=common_v450.UomType.HZ_S),
        )
        translated = common_annotation_l1.TxPulseType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
            tx_pulse_length=1.2,
            tx_pulse_start_frequency=2.3,
            tx_pulse_start_phase=4.5,
            tx_pulse_ramp_rate=5.6,
        )
        self.assertEqual(translate_common_annotation_l1.translate_tx_pulse(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_tx_pulse_to_model(translated), model)

    def test_translate_data_format(self):
        translated = common_annotation_l1.DataFormatType(
            echo_format=common.DataFormatModeType.BAQ_4_BIT,
            calibration_format=common.DataFormatModeType.BAQ_5_BIT,
            noise_format=common.DataFormatModeType.BAQ_6_BIT,
            mean_bit_rate=10.4,
        )

        model = v450.DataFormatType(
            echo_format=common_v450.DataFormatModeType.BAQ_4_BIT,
            calibration_format=common_v450.DataFormatModeType.BAQ_5_BIT,
            noise_format=common_v450.DataFormatModeType.BAQ_6_BIT,
            mean_bit_rate=common_v450.DoubleWithUnit(value=10.4, units=common_v450.UomType.MBPS),
        )
        self.assertEqual(translate_common_annotation_l1.translate_data_format_type(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_data_format_type_to_model(translated), model)

    def test_translate_instrument_parameters(self):
        translated = common_annotation_l1.InstrumentParametersType(
            first_line_sensing_time_list={
                common.PolarisationType.HV: PreciseDateTime.from_numeric_datetime(2021),
                common.PolarisationType.VV: PreciseDateTime.from_numeric_datetime(2024),
            },
            last_line_sensing_time_list={common.PolarisationType.VH: PreciseDateTime.from_numeric_datetime(2022)},
            number_of_input_samples=52,
            number_of_input_lines=89,
            swp_list=[
                (PreciseDateTime.from_numeric_datetime(2021), 4.5),
                (PreciseDateTime.from_numeric_datetime(2022), 5.4),
            ],
            swl_list=[
                (PreciseDateTime.from_numeric_datetime(2020), 6.5),
            ],
            prf_list=[],
            rank=8,
            tx_pulse_list=[
                common_annotation_l1.TxPulseType(PreciseDateTime.from_numeric_datetime(2020), 2.3, 3.5, 5.4, 9.2),
                common_annotation_l1.TxPulseType(PreciseDateTime.from_numeric_datetime(2021), 2.4, 3.6, 5.5, 9.3),
            ],
            instrument_configuration_id=9,
            radar_carrier_frequency=500.5,
            rx_gain_list={common.PolarisationType.H: 1.0, common.PolarisationType.V: 2.0},
            preamble_flag=True,
            postamble_flag=False,
            interleaved_calibration_flag=False,
            data_format=common_annotation_l1.DataFormatType(
                echo_format=common.DataFormatModeType.BAQ_4_BIT,
                calibration_format=common.DataFormatModeType.BAQ_5_BIT,
                noise_format=common.DataFormatModeType.BAQ_6_BIT,
                mean_bit_rate=10.4,
            ),
        )

        model = v450.InstrumentParametersType(
            first_line_sensing_time_list=v450.FirstLineSensingTimeListType(
                first_line_sensing_time=[
                    common_v450.TimeTypeWithPolarisation(
                        value="2021-01-01T00:00:00.000000",
                        polarisation=common_v450.PolarisationType.HV,
                    ),
                    common_v450.TimeTypeWithPolarisation(
                        value="2024-01-01T00:00:00.000000",
                        polarisation=common_v450.PolarisationType.VV,
                    ),
                ],
                count=2,
            ),
            last_line_sensing_time_list=v450.LastLineSensingTimeListType(
                last_line_sensing_time=[
                    common_v450.TimeTypeWithPolarisation(
                        value="2022-01-01T00:00:00.000000",
                        polarisation=common_v450.PolarisationType.VH,
                    ),
                ],
                count=1,
            ),
            number_of_input_samples=52,
            number_of_input_lines=89,
            swp_list=v450.SwpListType(
                swp=[
                    common_v450.StateType(
                        azimuth_time="2021-01-01T00:00:00.000000",
                        value=common_v450.FloatWithUnit(value=4.5, units=common_v450.UomType.S),
                    ),
                    common_v450.StateType(
                        azimuth_time="2022-01-01T00:00:00.000000",
                        value=common_v450.FloatWithUnit(value=5.4, units=common_v450.UomType.S),
                    ),
                ],
                count=2,
            ),
            swl_list=v450.SwlListType(
                swl=[
                    common_v450.StateType(
                        azimuth_time="2020-01-01T00:00:00.000000",
                        value=common_v450.FloatWithUnit(value=6.5, units=common_v450.UomType.S),
                    ),
                ],
                count=1,
            ),
            prf_list=v450.PrfListType(prf=[], count=0),
            rank=8,
            tx_pulse_list=v450.TxPulseListType(
                tx_pulse=[
                    v450.TxPulseType(
                        azimuth_time="2020-01-01T00:00:00.000000",
                        tx_pulse_length=common_v450.DoubleWithUnit(value=2.3, units=common_v450.UomType.S),
                        tx_pulse_start_frequency=common_v450.DoubleWithUnit(value=3.5, units=common_v450.UomType.HZ),
                        tx_pulse_start_phase=common_v450.DoubleWithUnit(value=5.4, units=common_v450.UomType.RAD),
                        tx_pulse_ramp_rate=common_v450.DoubleWithUnit(value=9.2, units=common_v450.UomType.HZ_S),
                    ),
                    v450.TxPulseType(
                        azimuth_time="2021-01-01T00:00:00.000000",
                        tx_pulse_length=common_v450.DoubleWithUnit(value=2.4, units=common_v450.UomType.S),
                        tx_pulse_start_frequency=common_v450.DoubleWithUnit(value=3.6, units=common_v450.UomType.HZ),
                        tx_pulse_start_phase=common_v450.DoubleWithUnit(value=5.5, units=common_v450.UomType.RAD),
                        tx_pulse_ramp_rate=common_v450.DoubleWithUnit(value=9.3, units=common_v450.UomType.HZ_S),
                    ),
                ],
                count=2,
            ),
            instrument_configuration_id=9,
            radar_carrier_frequency=common_v450.DoubleWithUnit(value=500.5, units=common_v450.UomType.HZ),
            rx_gain_list=v450.RxGainListType(
                rx_gain=[
                    common_v450.FloatWithPolarisation(value=1.0, polarisation=common_v450.PolarisationType.H),
                    common_v450.FloatWithPolarisation(value=2.0, polarisation=common_v450.PolarisationType.V),
                ],
                count=2,
            ),
            preamble_flag="true",
            postamble_flag="false",
            interleaved_calibration_flag="false",
            data_format=v450.DataFormatType(
                echo_format=common_v450.DataFormatModeType.BAQ_4_BIT,
                calibration_format=common_v450.DataFormatModeType.BAQ_5_BIT,
                noise_format=common_v450.DataFormatModeType.BAQ_6_BIT,
                mean_bit_rate=common_v450.DoubleWithUnit(value=10.4, units=common_v450.UomType.MBPS),
            ),
        )

        self.assertEqual(translate_common_annotation_l1.translate_instrument_parameters(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_instrument_parameters_to_model(translated), model)

    def test_translate_ionosphere_correction_to_model(self):
        translated = common_annotation_l1.IonosphereCorrection(
            ionosphere_height_used=350_000.0,
            ionosphere_height_estimated=1.0,
            ionosphere_height_estimation_method_selected=common.IonosphereHeightEstimationMethodType.AUTOMATIC,
            ionosphere_height_estimation_latitude_value=34.0,
            ionosphere_height_estimation_flag=False,
            ionosphere_height_estimation_method_used=common.IonosphereHeightEstimationMethodType.FEATURE_TRACKING,
            gaussian_filter_computation_flag=True,
            faraday_rotation_correction_applied=False,
            autofocus_shifts_applied=True,
        )
        model = v450.IonosphereCorrectionType(
            ionosphere_height_used=common_v450.FloatWithUnit(value=350_000.0, units=common_v450.UomType.M),
            ionosphere_height_estimated=common_v450.FloatWithUnit(value=1.0, units=common_v450.UomType.M),
            ionosphere_height_estimation_method_selected=common_v450.IonosphereHeightEstimationMethodType.AUTOMATIC,
            ionosphere_height_estimation_latitude_value=common_v450.FloatWithUnit(
                value=34.0, units=common_v450.UomType.DEG
            ),
            ionosphere_height_estimation_flag="false",
            ionosphere_height_estimation_method_used=common_v450.IonosphereHeightEstimationMethodType.FEATURE_TRACKING,
            gaussian_filter_computation_flag="true",
            faraday_rotation_correction_applied="false",
            autofocus_shifts_applied="true",
        )

        self.assertEqual(translate_common_annotation_l1.translate_ionosphere_correction(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_ionosphere_correction_to_model(translated), model)

    def test_translate_fm_rate_estimate_list(self):
        translated = [
            common.SlantRangePolynomialType(
                azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
                t0=4.5,
                polynomial=[1.3, 4.5],
            ),
            common.SlantRangePolynomialType(
                azimuth_time=PreciseDateTime.from_numeric_datetime(2022),
                t0=9,
                polynomial=[2.3, 2.5],
            ),
        ]
        model = v450.FmRateEstimatesListType(
            fm_rate_estimate=[
                common_v450.SlantRangePolynomialType(
                    azimuth_time="2021-01-01T00:00:00.000000",
                    t0=common_v450.DoubleWithUnit(value=4.5, units=common_v450.UomType.S),
                    polynomial=common_v450.DoubleArray(value="1.3 4.5", count=2),
                ),
                common_v450.SlantRangePolynomialType(
                    azimuth_time="2022-01-01T00:00:00.000000",
                    t0=common_v450.DoubleWithUnit(value=9, units=common_v450.UomType.S),
                    polynomial=common_v450.DoubleArray(value="2.3 2.5", count=2),
                ),
            ],
            count=2,
        )
        self.assertEqual(translate_common_annotation_l1.translate_fm_rate_estimate_list(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_fm_rate_estimate_list_to_model(translated), model)

    def test_translate_doppler_parameters(self):
        translated = common_annotation_l1.DopplerParametersType(
            [
                common_annotation_l1.DcEstimateType(
                    azimuth_time=PreciseDateTime.from_numeric_datetime(2020),
                    t0=0.5,
                    geometry_dcpolynomial=[8],
                    combined_dcpolynomial=[],
                    combined_dcvalues=[1, 5],
                    combined_dcslant_range_times=[0.1, 0.5],
                    combined_dcrmserror=10,
                    combined_dcrmserror_above_threshold=False,
                )
            ],
            [
                common.SlantRangePolynomialType(
                    azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
                    t0=4.5,
                    polynomial=[1.3, 4.5],
                )
            ],
        )

        model = v450.DopplerParametersType(
            dc_estimate_list=v450.DcEstimateListType(
                dc_estimate=[
                    v450.DcEstimateType(
                        azimuth_time="2020-01-01T00:00:00.000000",
                        t0=common_v450.DoubleWithUnit(value=0.5, units=common_v450.UomType.S),
                        geometry_dcpolynomial=common_v450.FloatArray(value="8.0", count=1),
                        combined_dcpolynomial=common_v450.FloatArray(value="", count=0),
                        combined_dcvalues=common_v450.DoubleArrayWithUnits(
                            value="1.0 5.0", count=2, units=common_v450.UomType.HZ
                        ),
                        combined_dcslant_range_times=common_v450.DoubleArrayWithUnits(
                            value="0.1 0.5", count=2, units=common_v450.UomType.S
                        ),
                        combined_dcrmserror=common_v450.DoubleWithUnit(value=10, units=common_v450.UomType.HZ),
                        combined_dcrmserror_above_threshold="false",
                    )
                ],
                count=1,
            ),
            fm_rate_estimate_list=v450.FmRateEstimatesListType(
                fm_rate_estimate=[
                    common_v450.SlantRangePolynomialType(
                        azimuth_time="2021-01-01T00:00:00.000000",
                        t0=common_v450.DoubleWithUnit(value=4.5, units=common_v450.UomType.S),
                        polynomial=common_v450.DoubleArray(value="1.3 4.5", count=2),
                    )
                ],
                count=1,
            ),
        )

        self.assertEqual(translate_common_annotation_l1.translate_doppler_parameters(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_doppler_parameters_to_model(translated), model)

    def test_translate_error_counters(self):
        translated = common_annotation_l1.ErrorCountersType(1, 2)
        model = v450.ErrorCountersType(num_isp_header_errors=1, num_isp_missing=2)
        self.assertEqual(translate_common_annotation_l1.translate_error_counters(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_error_counters_to_model(translated), model)

    def test_translate_raw_data_statistics(self):
        translated = common_annotation_l1.RawDataStatisticsType(
            i_bias=2.2,
            q_bias=1.1,
            iq_quadrature_departure=3.3,
            iq_gain_imbalance=4.4,
            polarisation=common.PolarisationType.HV,
        )
        model = v450.RawDataStatisticsType(
            i_bias=2.2,
            q_bias=1.1,
            iq_quadrature_departure=3.3,
            iq_gain_imbalance=4.4,
            polarisation=common_v450.PolarisationType.HV,
        )
        self.assertEqual(translate_common_annotation_l1.translate_raw_data_statistics(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_raw_data_statistics_to_model(translated), model)

    def test_translate_raw_data_statistics_list(self):
        translated = [
            common_annotation_l1.RawDataStatisticsType(
                i_bias=2.2,
                q_bias=1.1,
                iq_quadrature_departure=3.3,
                iq_gain_imbalance=4.4,
                polarisation=common.PolarisationType.HV,
            ),
            common_annotation_l1.RawDataStatisticsType(
                i_bias=12.2,
                q_bias=11.1,
                iq_quadrature_departure=13.3,
                iq_gain_imbalance=14.4,
                polarisation=common.PolarisationType.VV,
            ),
        ]
        model = v450.RawDataStatisticsListType(
            raw_data_statistics=[
                v450.RawDataStatisticsType(
                    i_bias=2.2,
                    q_bias=1.1,
                    iq_quadrature_departure=3.3,
                    iq_gain_imbalance=4.4,
                    polarisation=common_v450.PolarisationType.HV,
                ),
                v450.RawDataStatisticsType(
                    i_bias=12.2,
                    q_bias=11.1,
                    iq_quadrature_departure=13.3,
                    iq_gain_imbalance=14.4,
                    polarisation=common_v450.PolarisationType.VV,
                ),
            ],
            count=2,
        )
        self.assertEqual(translate_common_annotation_l1.translate_raw_data_statistics_list(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_raw_data_statistics_list_to_model(translated), model)

    def test_translate_raw_data_analysis(self):
        translated = common_annotation_l1.RawDataAnalysisType(
            common_annotation_l1.ErrorCountersType(1, 2),
            [
                common_annotation_l1.RawDataStatisticsType(
                    i_bias=2.2,
                    q_bias=1.1,
                    iq_quadrature_departure=3.3,
                    iq_gain_imbalance=4.4,
                    polarisation=common.PolarisationType.HV,
                ),
                common_annotation_l1.RawDataStatisticsType(
                    i_bias=12.2,
                    q_bias=11.1,
                    iq_quadrature_departure=13.3,
                    iq_gain_imbalance=14.4,
                    polarisation=common.PolarisationType.VV,
                ),
            ],
            {
                common.PolarisationType.HH: [
                    common_annotation_l1.PowerRatioType(
                        azimuth_time=PreciseDateTime.from_numeric_datetime(2021), value=1.0
                    )
                ]
            },
        )

        model = v450.RawDataAnalysisType(
            error_counters=v450.ErrorCountersType(
                num_isp_header_errors=1,
                num_isp_missing=2,
            ),
            raw_data_statistics_list=v450.RawDataStatisticsListType(
                raw_data_statistics=[
                    v450.RawDataStatisticsType(
                        i_bias=2.2,
                        q_bias=1.1,
                        iq_quadrature_departure=3.3,
                        iq_gain_imbalance=4.4,
                        polarisation=common_v450.PolarisationType.HV,
                    ),
                    v450.RawDataStatisticsType(
                        i_bias=12.2,
                        q_bias=11.1,
                        iq_quadrature_departure=13.3,
                        iq_gain_imbalance=14.4,
                        polarisation=common_v450.PolarisationType.VV,
                    ),
                ],
                count=2,
            ),
            in_out_band_power_ratio_list=v450.InOutBandPowerRatioListType(
                power_ratio_list=[
                    v450.PowerRatioListType(
                        power_ratio=[v450.PowerRatioType(azimuth_time="2021-01-01T00:00:00.000000", value=1.0)],
                        polarisation=common_v450.PolarisationType.HH,
                        count=1,
                    )
                ],
                count=1,
            ),
        )
        self.assertEqual(translate_common_annotation_l1.translate_raw_data_analysis(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_raw_data_analysis_to_model(translated), model)

    def test_translate_internal_calibration_sequence(self):
        translated = common_annotation_l1.InternalCalibrationSequenceType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
            drift_amplitude=1.1,
            drift_phase=2.2,
            model_drift_amplitude=3.3,
            model_drift_phase=4.4,
            relative_drift_valid_flag=True,
            absolute_drift_valid_flag=False,
            cross_correlation_bandwidth=5.5,
            cross_correlation_pslr=6.6,
            cross_correlation_islr=7.7,
            cross_correlation_peak_location=8.8,
            reconstructed_replica_valid_flag=True,
            internal_time_delay=9.9,
            internal_tx_channel_imbalance_amplitude=13.4,
            internal_tx_channel_imbalance_phase=10.1,
            internal_rx_channel_imbalance_amplitude=13.4,
            internal_rx_channel_imbalance_phase=10.1,
            transmit_power_tracking_d1_amplitude=11.2,
            transmit_power_tracking_d1_phase=12.3,
            receive_power_tracking_d1_amplitude=11.2,
            receive_power_tracking_d1_phase=12.3,
            transmit_power_tracking_d2_amplitude=11.2,
            transmit_power_tracking_d2_phase=12.3,
            receive_power_tracking_d2_amplitude=11.2,
            receive_power_tracking_d2_phase=12.3,
        )
        model = v450.InternalCalibrationSequenceType(
            azimuth_time="2021-01-01T00:00:00.000000",
            drift_amplitude=1.1,
            drift_phase=common_v450.FloatWithUnit(value=2.2, units=common_v450.UomType.RAD),
            model_drift_amplitude=3.3,
            model_drift_phase=common_v450.FloatWithUnit(value=4.4, units=common_v450.UomType.RAD),
            relative_drift_valid_flag="true",
            absolute_drift_valid_flag="false",
            cross_correlation_bandwidth=common_v450.FloatWithUnit(value=5.5, units=common_v450.UomType.HZ),
            cross_correlation_pslr=common_v450.FloatWithUnit(value=6.6, units=common_v450.UomType.D_B),
            cross_correlation_islr=common_v450.FloatWithUnit(value=7.7, units=common_v450.UomType.D_B),
            cross_correlation_peak_location=common_v450.FloatWithUnit(value=8.8, units=common_v450.UomType.SAMPLES),
            reconstructed_replica_valid_flag="true",
            internal_time_delay=common_v450.FloatWithUnit(value=9.9, units=common_v450.UomType.S),
            internal_tx_channel_imbalance_amplitude=13.4,
            internal_tx_channel_imbalance_phase=common_v450.FloatWithUnit(value=10.1, units=common_v450.UomType.RAD),
            internal_rx_channel_imbalance_amplitude=13.4,
            internal_rx_channel_imbalance_phase=common_v450.FloatWithUnit(value=10.1, units=common_v450.UomType.RAD),
            transmit_power_tracking_d1_amplitude=11.2,
            transmit_power_tracking_d1_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
            receive_power_tracking_d1_amplitude=11.2,
            receive_power_tracking_d1_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
            transmit_power_tracking_d2_amplitude=11.2,
            transmit_power_tracking_d2_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
            receive_power_tracking_d2_amplitude=11.2,
            receive_power_tracking_d2_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
        )
        self.assertEqual(translate_common_annotation_l1.translate_internal_calibration_sequence(model), translated)
        self.assertEqual(
            translate_common_annotation_l1.translate_internal_calibration_sequence_to_model(translated), model
        )

    def test_translate_internal_calibration_sequence_list(self):
        translated_s_1 = common_annotation_l1.InternalCalibrationSequenceType(
            azimuth_time=PreciseDateTime.from_numeric_datetime(2021),
            drift_amplitude=1.1,
            drift_phase=2.2,
            model_drift_amplitude=3.3,
            model_drift_phase=4.4,
            relative_drift_valid_flag=True,
            absolute_drift_valid_flag=False,
            cross_correlation_bandwidth=5.5,
            cross_correlation_pslr=6.6,
            cross_correlation_islr=7.7,
            cross_correlation_peak_location=8.8,
            reconstructed_replica_valid_flag=True,
            internal_time_delay=9.9,
            internal_tx_channel_imbalance_amplitude=13.4,
            internal_tx_channel_imbalance_phase=10.1,
            internal_rx_channel_imbalance_amplitude=13.4,
            internal_rx_channel_imbalance_phase=10.1,
            transmit_power_tracking_d1_amplitude=11.2,
            transmit_power_tracking_d1_phase=12.3,
            receive_power_tracking_d1_amplitude=11.2,
            receive_power_tracking_d1_phase=12.3,
            transmit_power_tracking_d2_amplitude=11.2,
            transmit_power_tracking_d2_phase=12.3,
            receive_power_tracking_d2_amplitude=11.2,
            receive_power_tracking_d2_phase=12.3,
        )
        model_s_1 = v450.InternalCalibrationSequenceType(
            azimuth_time="2021-01-01T00:00:00.000000",
            drift_amplitude=1.1,
            drift_phase=common_v450.FloatWithUnit(value=2.2, units=common_v450.UomType.RAD),
            model_drift_amplitude=3.3,
            model_drift_phase=common_v450.FloatWithUnit(value=4.4, units=common_v450.UomType.RAD),
            relative_drift_valid_flag="true",
            absolute_drift_valid_flag="false",
            cross_correlation_bandwidth=common_v450.FloatWithUnit(value=5.5, units=common_v450.UomType.HZ),
            cross_correlation_pslr=common_v450.FloatWithUnit(value=6.6, units=common_v450.UomType.D_B),
            cross_correlation_islr=common_v450.FloatWithUnit(value=7.7, units=common_v450.UomType.D_B),
            cross_correlation_peak_location=common_v450.FloatWithUnit(value=8.8, units=common_v450.UomType.SAMPLES),
            reconstructed_replica_valid_flag="true",
            internal_time_delay=common_v450.FloatWithUnit(value=9.9, units=common_v450.UomType.S),
            internal_tx_channel_imbalance_amplitude=13.4,
            internal_tx_channel_imbalance_phase=common_v450.FloatWithUnit(value=10.1, units=common_v450.UomType.RAD),
            internal_rx_channel_imbalance_amplitude=13.4,
            internal_rx_channel_imbalance_phase=common_v450.FloatWithUnit(value=10.1, units=common_v450.UomType.RAD),
            transmit_power_tracking_d1_amplitude=11.2,
            transmit_power_tracking_d1_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
            receive_power_tracking_d1_amplitude=11.2,
            receive_power_tracking_d1_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
            transmit_power_tracking_d2_amplitude=11.2,
            transmit_power_tracking_d2_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
            receive_power_tracking_d2_amplitude=11.2,
            receive_power_tracking_d2_phase=common_v450.FloatWithUnit(value=12.3, units=common_v450.UomType.RAD),
        )
        translated_s_2 = copy.deepcopy(translated_s_1)
        model_s_2 = copy.deepcopy(model_s_1)
        translated_s_2.drift_amplitude = 5.6
        model_s_2.drift_amplitude = 5.6

        translated = [translated_s_1, translated_s_2]
        model = v450.InternalCalibrationSequenceListType(
            internal_calibration_sequence=[model_s_1, model_s_2],
            count=2,
            polarisation=common_v450.PolarisationType.HV,
        )
        self.assertEqual(
            translate_common_annotation_l1.translate_internal_calibration_sequence_list(model),
            (common.PolarisationType.HV, translated),
        )
        self.assertEqual(
            translate_common_annotation_l1.translate_internal_calibration_sequence_list_to_model(
                common.PolarisationType.HV, translated
            ),
            model,
        )

    def test_translate_noise_sequence(self):
        translated = common_annotation_l1.NoiseSequenceType(PreciseDateTime.from_numeric_datetime(2021), 5.4, 1250)
        model = v450.NoiseSequenceType(
            azimuth_time="2021-01-01T00:00:00.000000",
            noise_power_correction_factor=5.4,
            number_of_noise_lines=1250,
        )
        self.assertEqual(translate_common_annotation_l1.translate_noise_sequence(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_noise_sequence_to_model(translated), model)

    def test_translate_noise_sequence_list(self):
        translated = [
            common_annotation_l1.NoiseSequenceType(PreciseDateTime.from_numeric_datetime(2021), 5.4, 1250),
            common_annotation_l1.NoiseSequenceType(PreciseDateTime.from_numeric_datetime(2022), 5.6, 320),
        ]
        model = v450.NoiseSequenceListType(
            noise_sequence=[
                v450.NoiseSequenceType(
                    azimuth_time="2021-01-01T00:00:00.000000",
                    noise_power_correction_factor=5.4,
                    number_of_noise_lines=1250,
                ),
                v450.NoiseSequenceType(
                    azimuth_time="2022-01-01T00:00:00.000000",
                    noise_power_correction_factor=5.6,
                    number_of_noise_lines=320,
                ),
            ],
            count=2,
            polarisation=common_v450.PolarisationType.VH,
        )
        self.assertEqual(
            translate_common_annotation_l1.translate_noise_sequence_list(model),
            (common.PolarisationType.VH, translated),
        )
        self.assertEqual(
            translate_common_annotation_l1.translate_noise_sequence_list_to_model(
                common.PolarisationType.VH, translated
            ),
            model,
        )

    def test_translate_radiometric_calibration(self):
        translated = common_annotation_l1.RadiometricCalibrationType(
            absolute_calibration_constant_list={
                common.PolarisationType.HH: 34.5,
                common.PolarisationType.HV: 4.5,
            }
        )
        model = v450.RadiometricCalibrationType(
            absolute_calibration_constant_list=v450.CalibrationConstantListType(
                absolute_calibration_constant=[
                    common_v450.FloatWithPolarisation(value=34.5, polarisation=common_v450.PolarisationType.HH),
                    common_v450.FloatWithPolarisation(value=4.5, polarisation=common_v450.PolarisationType.HV),
                ],
                count=2,
            )
        )

        self.assertEqual(translate_common_annotation_l1.translate_radiometric_calibration(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_radiometric_calibration_to_model(translated), model)

    def test_translate_geometry(self):
        translated = common_annotation_l1.GeometryType(
            height_model=common.HeightModelType(common.HeightModelBaseType.COPERNICUS_DEM, version="version"),
            height_model_used_flag=True,
            roll_bias=0.2,
        )
        model = v450.GeometryType(
            height_model=common_v450.HeightModelType(
                value=common_v450.HeightModelBaseType.COPERNICUS_DEM, version="version"
            ),
            height_model_used_flag="true",
            roll_bias=common_v450.FloatWithUnit(value=0.2, units=common_v450.UomType.DEG),
        )

        self.assertEqual(translate_common_annotation_l1.translate_geometry(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_geometry_to_model(translated), model)

    def test_translate_quality_params(self):
        translated = common_annotation_l1.QualityParametersType(
            missing_ispfraction=0.1,
            max_ispgap=2,
            max_ispgap_threshold=3,
            invalid_raw_data_samples=5,
            raw_mean_expected=0.6,
            raw_mean_threshold=0.7,
            raw_std_expected=0.8,
            raw_std_threshold=0.9,
            rfi_tmfraction=1.0,
            max_rfitmpercentage=0.1,
            rfi_fmfraction=0.2,
            max_rfifmpercentage=0.3,
            invalid_drift_fraction=0.4,
            max_invalid_drift_fraction=0.5,
            invalid_replica_fraction=0.6,
            invalid_dcestimates_fraction=0.7,
            dc_rmserror_threshold=52.2,
            residual_ionospheric_phase_screen_std=2.1,
            invalid_blocks_percentage=0.5,
            invalid_blocks_percentage_threshold=0.6,
            polarisation=common.PolarisationType.HV,
        )
        model = v450.QualityParametersType(
            missing_ispfraction=0.1,
            max_ispgap=2,
            max_ispgap_threshold=3,
            invalid_raw_data_samples=5,
            raw_mean_expected=0.6,
            raw_mean_threshold=0.7,
            raw_std_expected=0.8,
            raw_std_threshold=0.9,
            rfi_tmfraction=1.0,
            max_rfitmpercentage=0.1,
            rfi_fmfraction=0.2,
            max_rfifmpercentage=0.3,
            invalid_drift_fraction=0.4,
            max_invalid_drift_fraction=0.5,
            invalid_replica_fraction=0.6,
            invalid_dcestimates_fraction=0.7,
            dc_rmserror_threshold=common_v450.FloatWithUnit(value=52.2, units=common_v450.UomType.HZ),
            residual_ionospheric_phase_screen_std=common_v450.FloatWithUnit(value=2.1, units=common_v450.UomType.RAD),
            invalid_blocks_percentage=0.5,
            invalid_blocks_percentage_threshold=0.6,
            polarisation=common_v450.PolarisationType.HV,
        )

        self.assertEqual(translate_common_annotation_l1.translate_quality_parameters(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_quality_parameters_to_model(translated), model)

    def test_translate_quality(self):
        translated = common_annotation_l1.QualityType(
            overall_product_quality_index=20,
            quality_parameters_list=[
                common_annotation_l1.QualityParametersType(
                    missing_ispfraction=0.1,
                    max_ispgap=2,
                    max_ispgap_threshold=3,
                    invalid_raw_data_samples=5,
                    raw_mean_expected=0.6,
                    raw_mean_threshold=0.7,
                    raw_std_expected=0.8,
                    raw_std_threshold=0.9,
                    rfi_tmfraction=1.0,
                    max_rfitmpercentage=0.1,
                    rfi_fmfraction=0.2,
                    max_rfifmpercentage=0.3,
                    invalid_drift_fraction=0.4,
                    max_invalid_drift_fraction=0.5,
                    invalid_replica_fraction=0.6,
                    invalid_dcestimates_fraction=0.7,
                    dc_rmserror_threshold=52.2,
                    residual_ionospheric_phase_screen_std=2.1,
                    invalid_blocks_percentage=0.5,
                    invalid_blocks_percentage_threshold=0.6,
                    polarisation=common.PolarisationType.HV,
                ),
                common_annotation_l1.QualityParametersType(
                    missing_ispfraction=0.1,
                    max_ispgap=2,
                    max_ispgap_threshold=3,
                    invalid_raw_data_samples=5,
                    raw_mean_expected=0.6,
                    raw_mean_threshold=0.7,
                    raw_std_expected=0.8,
                    raw_std_threshold=0.4,
                    rfi_tmfraction=1.0,
                    max_rfitmpercentage=0.1,
                    rfi_fmfraction=0.2,
                    max_rfifmpercentage=0.3,
                    invalid_drift_fraction=0.4,
                    max_invalid_drift_fraction=0.5,
                    invalid_replica_fraction=0.6,
                    invalid_dcestimates_fraction=0.7,
                    dc_rmserror_threshold=52.2,
                    residual_ionospheric_phase_screen_std=2.1,
                    invalid_blocks_percentage=0.5,
                    invalid_blocks_percentage_threshold=0.6,
                    polarisation=common.PolarisationType.VH,
                ),
            ],
        )
        model = v450.QualityType(
            overall_product_quality_index=20,
            quality_parameters_list=v450.QualityParametersListType(
                quality_parameters=[
                    v450.QualityParametersType(
                        missing_ispfraction=0.1,
                        max_ispgap=2,
                        max_ispgap_threshold=3,
                        invalid_raw_data_samples=5,
                        raw_mean_expected=0.6,
                        raw_mean_threshold=0.7,
                        raw_std_expected=0.8,
                        raw_std_threshold=0.9,
                        rfi_tmfraction=1.0,
                        max_rfitmpercentage=0.1,
                        rfi_fmfraction=0.2,
                        max_rfifmpercentage=0.3,
                        invalid_drift_fraction=0.4,
                        max_invalid_drift_fraction=0.5,
                        invalid_replica_fraction=0.6,
                        invalid_dcestimates_fraction=0.7,
                        dc_rmserror_threshold=common_v450.FloatWithUnit(value=52.2, units=common_v450.UomType.HZ),
                        residual_ionospheric_phase_screen_std=common_v450.FloatWithUnit(
                            value=2.1, units=common_v450.UomType.RAD
                        ),
                        invalid_blocks_percentage=0.5,
                        invalid_blocks_percentage_threshold=0.6,
                        polarisation=common_v450.PolarisationType.HV,
                    ),
                    v450.QualityParametersType(
                        missing_ispfraction=0.1,
                        max_ispgap=2,
                        max_ispgap_threshold=3,
                        invalid_raw_data_samples=5,
                        raw_mean_expected=0.6,
                        raw_mean_threshold=0.7,
                        raw_std_expected=0.8,
                        raw_std_threshold=0.4,
                        rfi_tmfraction=1.0,
                        max_rfitmpercentage=0.1,
                        rfi_fmfraction=0.2,
                        max_rfifmpercentage=0.3,
                        invalid_drift_fraction=0.4,
                        max_invalid_drift_fraction=0.5,
                        invalid_replica_fraction=0.6,
                        invalid_dcestimates_fraction=0.7,
                        dc_rmserror_threshold=common_v450.FloatWithUnit(value=52.2, units=common_v450.UomType.HZ),
                        residual_ionospheric_phase_screen_std=common_v450.FloatWithUnit(
                            value=2.1, units=common_v450.UomType.RAD
                        ),
                        invalid_blocks_percentage=0.5,
                        invalid_blocks_percentage_threshold=0.6,
                        polarisation=common_v450.PolarisationType.VH,
                    ),
                ],
                count=2,
            ),
        )

        self.assertEqual(translate_common_annotation_l1.translate_quality(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_quality_to_model(translated), model)

    def test_translate_polarimetric_distortion(self):
        translated = common_annotation_l1.PolarimetricDistortionType(
            cross_talk=common.CrossTalkList(
                hv_rx=complex(0, 1.2),
                hv_tx=complex(1, 2.2),
                vh_rx=complex(2.2, -1.2),
                vh_tx=complex(3, 1.2),
            ),
            channel_imbalance=common.ChannelImbalanceList(complex(3.2, 3.4), complex(4.2, 1.4)),
        )

        model = v450.PolarimetricDistortionType(
            cross_talk_list=common_v450.CrossTalkList(
                cross_talk_hvrx=common_v450.Complex(re=0, im=1.2),
                cross_talk_hvtx=common_v450.Complex(re=1, im=2.2),
                cross_talk_vhrx=common_v450.Complex(re=2.2, im=-1.2),
                cross_talk_vhtx=common_v450.Complex(re=3, im=1.2),
            ),
            channel_imbalance_list=common_v450.ChannelImbalanceList(
                channel_imbal_hvrx=common_v450.Complex(re=3.2, im=3.4),
                channel_imbal_hvtx=common_v450.Complex(re=4.2, im=1.4),
            ),
        )
        self.assertEqual(translate_common_annotation_l1.translate_polarimetric_distortion(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_polarimetric_distortion_to_model(translated), model)

    def test_translate_rfi_mitigation(self):
        translated = common_annotation_l1.RfiMitigationType(
            rfi_tmreport_list=[],
            rfi_isolated_fmreport_list=[],
            rfi_persistent_fmreport_list=[],
        )
        model = v450.RfiMitigationType(
            rfi_tmreport_list=None,
            rfi_isolated_fmreport_list=None,
            rfi_persistent_fmreport_list=None,
        )

        self.assertEqual(translate_common_annotation_l1.translate_rfi_mitigation(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_rfi_mitigation_to_model(translated), model)

        translated = common_annotation_l1.RfiMitigationType(
            rfi_tmreport_list=[
                common_annotation_l1.RfiTmreportType(
                    percentage_affected_lines=0.2,
                    avg_percentage_affected_samples=0.5,
                    max_percentage_affected_samples=0.6,
                    polarisation=common.PolarisationType.HV,
                )
            ],
            rfi_isolated_fmreport_list=[
                common_annotation_l1.RfiIsolatedFmreportType(
                    percentage_affected_lines=0.8,
                    max_percentage_affected_bw=0.1,
                    avg_percentage_affected_bw=0.05,
                    polarisation=common.PolarisationType.VH,
                )
            ],
            rfi_persistent_fmreport_list=[
                common_annotation_l1.RfiPersistentFmreportType(
                    max_percentage_affected_bw=0.2,
                    avg_percentage_affected_bw=0.15,
                    percentage_affected_lines=0.2,
                    polarisation=common.PolarisationType.VV,
                )
            ],
        )
        model = v450.RfiMitigationType(
            rfi_tmreport_list=v450.RfiTmreportListType(
                rfi_tmreport=[
                    v450.RfiTmreportType(
                        percentage_affected_lines=0.2,
                        avg_percentage_affected_samples=0.5,
                        max_percentage_affected_samples=0.6,
                        polarisation=common_v450.PolarisationType.HV,
                    )
                ],
                count=1,
            ),
            rfi_isolated_fmreport_list=v450.RfiIsolatedFmreportListType(
                rfi_isolated_fmreport=[
                    v450.RfiIsolatedFmreportType(
                        percentage_affected_lines=0.8,
                        max_percentage_affected_bw=0.1,
                        avg_percentage_affected_bw=0.05,
                        polarisation=common_v450.PolarisationType.VH,
                    )
                ],
                count=1,
            ),
            rfi_persistent_fmreport_list=v450.RfiPersistentFmreportListType(
                rfi_persistent_fmreport=[
                    v450.RfiPersistentFmreportType(
                        max_percentage_affected_bw=0.2,
                        avg_percentage_affected_bw=0.15,
                        percentage_affected_lines=0.2,
                        polarisation=common_v450.PolarisationType.VV,
                    )
                ],
                count=1,
            ),
        )

        self.assertEqual(translate_common_annotation_l1.translate_rfi_mitigation(model), translated)
        self.assertEqual(translate_common_annotation_l1.translate_rfi_mitigation_to_model(translated), model)


if __name__ == "__main__":
    unittest.main()
