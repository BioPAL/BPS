import unittest

from bps.common.io import common
from bps.common.io.aresys_configuration_models.models import (
    CalibrationConstantsConfType,
    ComplexNumberType,
    PolarimetricProcessorConfType,
    RangeConfType,
)
from bps.l1_core_processor.processing_parameters import (
    AzimuthConf,
    BorderPolicy,
    CalibrationConstantsConf,
    PolarimetricProcessorConf,
    Quantity,
    RangeFocuserConf,
    RFIMitigationConf,
    WindowConf,
)
from bps.l1_core_processor.translate import (
    translate_calibration_constants_conf_to_model,
    translate_complex_to_model,
    translate_output_prf_value_to_model,
    translate_polarimetric_processor_conf_to_model,
)
from bps.l1_processor.core.processing_parameters_utils import (
    fill_azimuth_conf,
    fill_calibration_constants_conf,
    fill_polarimetric_processor_conf,
    fill_range_focuser_conf,
    fill_rfi_mitigation_conf,
)
from bps.l1_processor.processor_interface import aux_pp1


class FillRFIMitigatorConfTest(unittest.TestCase):
    def setUp(self) -> None:
        self.aux_time_domain_processing_parameters = aux_pp1.RFIMitigationConf.TimeDomainParams(
            block_lines=256,
            median_filter_length=5,
            box_samples=19,
            box_lines=19,
            percentile_threshold=0.99,
            morphological_open_operator_samples=1,
            morphological_open_operator_lines=3,
            morphological_close_operator_samples=1,
            morphological_close_operator_lines=9,
            max_rfi_percentage=0.5,
        )

        self.aux_freq_domain_processing_parameters = aux_pp1.RFIMitigationConf.FreqDomainParams(
            block_lines=512,
            block_overlap=0,
            persistent_rfi_threshold=2.0,
            isolated_rfi_threshold=4.0,
            isolated_rfi_psd_std_threshold=7,
            max_rfi_percentage=0.5,
            periodgram_size=128,
            enable_power_loss_compensation=False,
            power_loss_threshold=0.3,
            chirp_source=aux_pp1.ChirpSource.NOMINAL,
            mitigation_method="NOTCH_FILTER",
        )

        self.rfi_mitigator_conf = RFIMitigationConf(
            rfi_mitigation_method=RFIMitigationConf.Method.FREQUENCY_AND_TIME,
            rfi_mask_composition_method=RFIMitigationConf.MaskCompositionMethod.AND,
            time_domain_conf=RFIMitigationConf.TimeDomainConf(
                correction_mode=RFIMitigationConf.TimeDomainConf.CorrectionMode.NEAREST,
                percentile_threshold=0.99,
                median_filter_block_lines=5,
                lines_in_estimate_block=256,
                box_filter_azimuth_dimension=19,
                box_filter_range_dimension=19,
                morph_open_line_length=3,
                morph_close_line_length=9,
                morph_close_before_open=None,
                morph_open_close_iterations=None,
                swath=None,
            ),
            frequency_domain_conf=RFIMitigationConf.FrequencyDomainConf(
                block_size=512,
                periodgram_size=128,
                persistent_rfi_threshold=2.0,
                isolated_rfi_threshold=4.0,
                threshold_std=7,
                percentile_low=0.25,
                percentile_high=0.25,
                power_loss_threshold=None,
                remove_interferences=True,
                swath=None,
            ),
            swath="S1",
        )

        self.aux_rfi_mitigator = aux_pp1.RFIMitigationConf(
            detection_flag=False,
            activation_mode="Disabled",
            activation_mask_threshold=0.25,
            mitigation_method=aux_pp1.RFIMitigationConf.MitigationMethod.FREQUENCY_AND_TIME,
            mask=aux_pp1.RFIMitigationConf.MaskType.SINGLE,
            mask_generation_method=aux_pp1.RFIMitigationConf.MaskGenerationMethod.AND,
            time_domain_processing_parameters=self.aux_time_domain_processing_parameters,
            freq_domain_processing_parameters=self.aux_freq_domain_processing_parameters,
        )

    def test_mask_single(self):
        self.aux_rfi_mitigator.mask = aux_pp1.RFIMitigationConf.MaskType.SINGLE

        self.aux_rfi_mitigator.mask_generation_method = aux_pp1.RFIMitigationConf.MaskGenerationMethod.AND
        self.rfi_mitigator_conf.rfi_mask_composition_method = RFIMitigationConf.MaskCompositionMethod.AND

        self.assertEqual(
            fill_rfi_mitigation_conf(self.aux_rfi_mitigator, "S1"),
            self.rfi_mitigator_conf,
        )

        self.aux_rfi_mitigator.mask_generation_method = aux_pp1.RFIMitigationConf.MaskGenerationMethod.OR
        self.rfi_mitigator_conf.rfi_mask_composition_method = RFIMitigationConf.MaskCompositionMethod.OR

        self.assertEqual(
            fill_rfi_mitigation_conf(self.aux_rfi_mitigator, "S1"),
            self.rfi_mitigator_conf,
        )

    def test_mask_multiple(self):
        self.aux_rfi_mitigator.mask = aux_pp1.RFIMitigationConf.MaskType.MULTIPLE
        self.aux_rfi_mitigator.mask_generation_method = aux_pp1.RFIMitigationConf.MaskGenerationMethod.AND

        self.rfi_mitigator_conf.rfi_mask_composition_method = RFIMitigationConf.MaskCompositionMethod.NONE

        self.assertEqual(
            fill_rfi_mitigation_conf(self.aux_rfi_mitigator, "S1"),
            self.rfi_mitigator_conf,
        )

        self.aux_rfi_mitigator.mask = aux_pp1.RFIMitigationConf.MaskType.MULTIPLE
        self.aux_rfi_mitigator.mask_generation_method = aux_pp1.RFIMitigationConf.MaskGenerationMethod.OR

        self.rfi_mitigator_conf.rfi_mask_composition_method = RFIMitigationConf.MaskCompositionMethod.NONE

        self.assertEqual(
            fill_rfi_mitigation_conf(self.aux_rfi_mitigator, "S1"),
            self.rfi_mitigator_conf,
        )


class AzimuthConfTest(unittest.TestCase):
    def setUp(self) -> None:
        aux_swath = aux_pp1.Swath.S1

        aux_azimuth_conf_parameters = aux_pp1.AzimuthCompressionConf.Parameters(
            time_bias=0.0,
            window_type=aux_pp1.WindowType.HAMMING,
            window_coefficient=0.7735,
            processing_bandwidth=849.0,
        )

        self.aux_azimuth_compression_conf = aux_pp1.AzimuthCompressionConf(
            block_samples=256,
            block_lines=25000,
            block_overlap_samples=46,
            block_overlap_lines=10888,
            parameters={aux_swath: aux_azimuth_conf_parameters},
            bistatic_delay_correction=True,
            bistatic_delay_correction_method=aux_pp1.AzimuthCompressionConf.Method.FULL,
            azimuth_resampling=True,
            azimuth_resampling_frequency=1018.8,
            azimuth_coregistration_flag=False,
            azimuth_focusing_margins_removal_flag=True,
            filter_type="GLS",
            filter_bandwidth=849,
            filter_length=11,
            number_of_filters=512,
        )

        processing_bandwidth = Quantity(
            849.0,
            unit=Quantity.Unit.HZ,
        )

        window = WindowConf(
            window_type=WindowConf.Type.HAMMING,
            window_parameter=0.7735,
            window_look_bandwidth=processing_bandwidth,
            window_transition_bandwidth=processing_bandwidth,
        )

        self.azimuth_conf = AzimuthConf(
            lines_in_block=25000,
            samples_in_block=256,
            azimuth_overlap=10888,
            range_overlap=46,
            perform_interpolation=0,
            stolt_padding=0.4,
            range_modulation=False,
            apply_azimuth_spectral_weighting_window=True,
            azimuth_spectral_weighting_window=window,
            apply_rg_shift=True,
            apply_az_shift=True,
            whitening_flag=False,
            antenna_length=0.0,
            pad_result=0,
            lines_to_skip_dc_fr=None,
            samples_to_skip_dc_fr=None,
            focusing_method=AzimuthConf.Method.WK,
            az_proc_bandwidth=processing_bandwidth,
            bistatic_delay_correction_mode=AzimuthConf.BistaticDelayCorrectionMode.RANGE_DEPENDENT,
            azimuth_time_bias=0.0,
            antenna_shift_compensation_mode=None,
            apply_pol_channels_coregistration=False,
            nominal_block_memory_size_cpu=None,
            nominal_block_memory_size_gpu=None,
            swath="S1",
        )

    def test_fill_azimuth_conf(self):
        self.assertEqual(
            fill_azimuth_conf(self.aux_azimuth_compression_conf, "S1", ionospheric_correction_enabled=False),
            self.azimuth_conf,
        )

    def test_apply_pol_channels_coregistration(self):
        aziumuth_conf = fill_azimuth_conf(self.aux_azimuth_compression_conf, "S1", ionospheric_correction_enabled=False)

        self.assertEqual(aziumuth_conf.apply_pol_channels_coregistration, False)


class RangeFocuserConfTest(unittest.TestCase):
    def setUp(self) -> None:
        aux_swath = aux_pp1.Swath.S1

        aux_range_compression_parameters = aux_pp1.RangeCompressionConf.Parameters(
            time_bias=0.0,
            window_type=aux_pp1.WindowType.HAMMING,
            window_coefficient=0.9055,
            processing_bandwidth=6000000.0,
        )

        self.aux_range_compression_conf = aux_pp1.RangeCompressionConf(
            range_compression_method=aux_pp1.RangeCompressionConf.Method.INVERSE_FILTER,
            extended_swath_processing=False,
            range_reference_function_source=aux_pp1.ChirpSource.NOMINAL,
            parameters={aux_swath: aux_range_compression_parameters},
        )

        processing_bandwidth = Quantity(
            6000000.0,
            unit=Quantity.Unit.HZ,
        )

        window = WindowConf(
            window_type=WindowConf.Type.HAMMING,
            window_parameter=0.9055,
            window_look_bandwidth=processing_bandwidth,
            window_transition_bandwidth=processing_bandwidth,
        )

        self.range_focuser_conf = RangeFocuserConf(
            flag_ortog=False,
            apply_range_spectral_weighting_window=True,
            range_spectral_weighting_window=window,
            swst_bias=1.0,
            range_decimation_factor=None,
            apply_rx_gain_correction=None,
            focusing_method=RangeFocuserConf.Method.INVERSE_FILTER,
            output_prf_value=None,
            output_range_border_policy=BorderPolicy.CUT,
            swath="S1",
        )

    def test_fill_range_focuser_conf(self):
        self.assertEqual(
            fill_range_focuser_conf(
                conf=self.aux_range_compression_conf, output_prf_value=None, instrument_swst_bias=1.0, swath="S1"
            ),
            self.range_focuser_conf,
        )

    def test_translate_output_prf_value_to_model(self):
        value = 1600.0

        output_prf = RangeConfType.PostProcessing.AzimuthResampling.OutputPrf(value=value)
        azimuth_resampling = RangeConfType.PostProcessing.AzimuthResampling(output_prf=output_prf)
        model_range_focuser_post_processing = RangeConfType.PostProcessing(azimuth_resampling=azimuth_resampling)

        self.assertEqual(
            translate_output_prf_value_to_model(value=1600.0),
            model_range_focuser_post_processing,
        )


class PolarimetricCalibrationConfTest(unittest.TestCase):
    def setUp(self) -> None:
        self.aux_polarimetric_processing_parameters = aux_pp1.PolarimetricCalibrationConf(
            polarimetric_correction_flag=True,
            tx_distortion_matrix_correction_flag=True,
            rx_distortion_matrix_correction_flag=True,
            cross_talk_correction_flag=True,
            cross_talk=common.CrossTalkList(
                hv_rx=complex(0.0, 0.0),
                vh_rx=complex(0.0, 0.0),
                vh_tx=complex(0.0, 0.0),
                hv_tx=complex(0.0, 0.0),
            ),
            channel_imbalance_correction_flag=True,
            channel_imbalance=common.ChannelImbalanceList(
                hv_rx=complex(1.0, 0.0),
                hv_tx=complex(1.0, 0.0),
            ),
        )

        self.polarimetric_processor_conf = PolarimetricProcessorConf(
            enable_channel_imbalance_compensation=True,
            enable_cross_talk_compensation=True,
            swath="S1",
        )

        self.polarimetric_calibration_constants_conf = CalibrationConstantsConf(
            channel_imbalance_tx=complex(1.0, 0.0),
            channel_imbalance_rx=complex(1.0, 0.0),
            cross_talk_hv_rx=complex(0.0, 0.0),
            cross_talk_vh_rx=complex(0.0, 0.0),
            cross_talk_vh_tx=complex(0.0, 0.0),
            cross_talk_hv_tx=complex(0.0, 0.0),
            internal_delay_hh=0.0,
            internal_delay_hv=0.0,
            internal_delay_vh=0.0,
            internal_delay_vv=0.0,
        )

    def test_fill_polarimetric_processor_conf(self):
        self.assertEqual(
            fill_polarimetric_processor_conf(conf=self.aux_polarimetric_processing_parameters, swath="S1"),
            self.polarimetric_processor_conf,
        )

    def test_fill_calibration_constants_conf(self):
        self.assertEqual(
            fill_calibration_constants_conf(
                conf=self.aux_polarimetric_processing_parameters,
                channel_imbalance=None,
            ),
            self.polarimetric_calibration_constants_conf,
        )

    def test_translate_polarimetric_processor_conf_to_model(self):
        model_polarimetric_processor = PolarimetricProcessorConfType(
            beam="S1",
            enable_channel_imbalance_compensation=True,
            enable_cross_talk_compensation=True,
        )

        self.assertEqual(
            translate_polarimetric_processor_conf_to_model(self.polarimetric_processor_conf),
            model_polarimetric_processor,
        )

    def test_translate_complex_to_model(self):
        complex_number_translated_a = translate_complex_to_model(complex(5.0, -2.0))
        complex_number_model_a = ComplexNumberType(amplitude=5.385164807134504, phase=-0.3805063771123649)

        assert complex_number_translated_a.amplitude is not None
        assert complex_number_model_a.amplitude is not None

        self.assertAlmostEqual(
            complex_number_translated_a.amplitude,
            complex_number_model_a.amplitude,
            places=7,
        )

        assert complex_number_translated_a.phase is not None
        assert complex_number_model_a.phase is not None

        self.assertAlmostEqual(
            complex_number_translated_a.phase,
            complex_number_model_a.phase,
            places=7,
        )

    def test_translate_calibration_constants_conf_to_model(self):
        channel_imbalance_values = CalibrationConstantsConfType.Radiometric.ChannelImbalanceValues(
            rx=translate_complex_to_model(complex_number=complex(1.0, 0.0)),
            tx=translate_complex_to_model(complex_number=complex(1.0, 0.0)),
        )
        cross_talk_correction = CalibrationConstantsConfType.Radiometric.CrossTalkCorrection(
            xtalk1=translate_complex_to_model(complex(0.0)),
            xtalk2=translate_complex_to_model(complex(0.0)),
            xtalk3=translate_complex_to_model(complex(0.0)),
            xtalk4=translate_complex_to_model(complex(0.0)),
        )

        radiometric = CalibrationConstantsConfType.Radiometric(
            channel_imbalance_values=channel_imbalance_values,
            cross_talk_correction=cross_talk_correction,
        )
        geometric = CalibrationConstantsConfType.Geometric(
            internal_delay_hh=0.0,
            internal_delay_hv=0.0,
            internal_delay_vh=0.0,
            internal_delay_vv=0.0,
        )

        model_calibration_constants = CalibrationConstantsConfType(radiometric=radiometric, geometric=geometric)

        self.assertEqual(
            translate_calibration_constants_conf_to_model(self.polarimetric_calibration_constants_conf),
            model_calibration_constants,
        )


if __name__ == "__main__":
    unittest.main()
