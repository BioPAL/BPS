# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common annotations l1
---------------------
"""

from dataclasses import dataclass
from typing import Literal

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common

ProcessingGainList = list[common.FloatWithPolarisation]
NoiseGainList = list[common.FloatWithPolarisation]


@dataclass
class SpectrumProcessingParametersType:
    """Spectrum parameters"""

    window_type: common.WeightingWindowType
    window_coefficient: float
    total_bandwidth: float
    processing_bandwidth: float
    look_bandwidth: float
    number_of_looks: int
    look_overlap: float


@dataclass
class ProcessingParameters:
    """Processing parameters"""

    processor_version: str
    product_generation_time: PreciseDateTime
    processing_mode: common.ProcessingModeType
    orbit_source: common.OrbitAttitudeSourceType
    attitude_source: common.OrbitAttitudeSourceType
    raw_data_correction_flag: bool
    rfi_detection_flag: bool
    rfi_correction_flag: bool
    rfi_mitigation_method: common.RfiMitigationMethodType
    rfi_mask: common.RfiMaskType
    rfi_mask_generation_method: common.RfiMaskGenerationMethodType
    rfi_fm_mitigation_method: Literal["NOTCH_FILTER", "NEAREST_NEIGHBOUR_INTERPOLATION"]
    rfi_fm_chirp_source: common.RangeReferenceFunctionType
    internal_calibration_estimation_flag: bool
    internal_calibration_correction_flag: bool
    range_reference_function_source: common.RangeReferenceFunctionType
    range_compression_method: common.RangeCompressionMethodType
    extended_swath_processing_flag: bool
    dc_method: common.DcMethodType
    dc_value: float
    antenna_pattern_correction1_flag: bool
    antenna_pattern_correction2_flag: bool
    antenna_cross_talk_correction_flag: bool
    range_processing_parameters: SpectrumProcessingParametersType
    azimuth_processing_parameters: SpectrumProcessingParametersType
    bistatic_delay_correction_flag: bool
    bistatic_delay_correction_method: common.BistaticDelayCorrectionMethodType
    range_spreading_loss_compensation_flag: bool
    reference_range: float
    processing_gain_list: ProcessingGainList
    polarimetric_correction_flag: bool
    ionosphere_height_defocusing_flag: bool
    ionosphere_height_estimation_method: common.IonosphereHeightEstimationMethodType
    faraday_rotation_correction_flag: bool
    ionospheric_phase_screen_correction_flag: bool
    group_delay_correction_flag: bool
    autofocus_flag: bool
    autofocus_method: common.AutofocusMethodType
    detection_flag: bool
    thermal_denoising_flag: bool
    noise_gain_list: NoiseGainList
    ground_projection_flag: bool


@dataclass
class DcEstimateType:
    """DC estimation section"""

    azimuth_time: PreciseDateTime
    t0: float
    geometry_dcpolynomial: list[float]
    combined_dcpolynomial: list[float]
    combined_dcvalues: list[float]
    combined_dcslant_range_times: list[float]
    combined_dcrmserror: float
    combined_dcrmserror_above_threshold: bool


@dataclass
class CoordinateConversionType:
    """Coordinate conversions"""

    azimuth_time: PreciseDateTime
    t0: float
    sr0: float
    slant_to_ground_coefficients: list[float]
    gr0: float
    ground_to_slant_coefficients: list[float]


@dataclass
class AcquisitionInformationType:
    """Acquisition information"""

    mission: common.MissionType
    swath: common.SwathType
    product_type: common.ProductType
    polarisation_list: list[common.PolarisationType]
    start_time: PreciseDateTime
    stop_time: PreciseDateTime
    mission_phase_id: common.MissionPhaseIdtype
    drift_phase_flag: bool
    sensor_mode: common.SensorModeType
    global_coverage_id: int
    major_cycle_id: int
    repeat_cycle_id: int
    absolute_orbit_number: int
    relative_orbit_number: int
    orbit_pass: common.OrbitPassType
    platform_heading: float
    data_take_id: int
    frame: int
    product_composition: common.ProductCompositionType


@dataclass
class TxPulseType:
    """TX Pulse"""

    azimuth_time: PreciseDateTime
    tx_pulse_length: float
    tx_pulse_start_frequency: float
    tx_pulse_start_phase: float
    tx_pulse_ramp_rate: float


@dataclass
class DataFormatType:
    """Data format"""

    echo_format: common.DataFormatModeType
    calibration_format: common.DataFormatModeType
    noise_format: common.DataFormatModeType
    mean_bit_rate: float


@dataclass
class InstrumentParametersType:
    """Instrument parameters"""

    first_line_sensing_time_list: dict[common.PolarisationType, PreciseDateTime]
    last_line_sensing_time_list: dict[common.PolarisationType, PreciseDateTime]
    number_of_input_samples: int
    number_of_input_lines: int
    swp_list: list[tuple[PreciseDateTime, float]]
    swl_list: list[tuple[PreciseDateTime, float]]
    prf_list: list[tuple[PreciseDateTime, float]]
    rank: int
    tx_pulse_list: list[TxPulseType]
    instrument_configuration_id: int
    radar_carrier_frequency: float
    rx_gain_list: dict[common.PolarisationType, float]
    preamble_flag: bool
    postamble_flag: bool
    interleaved_calibration_flag: bool
    data_format: DataFormatType


@dataclass
class SarImageType:
    """SAR image"""

    first_sample_slant_range_time: float
    last_sample_slant_range_time: float
    first_line_azimuth_time: PreciseDateTime
    last_line_azimuth_time: PreciseDateTime
    range_time_interval: float
    azimuth_time_interval: float
    range_pixel_spacing: float
    azimuth_pixel_spacing: float
    number_of_samples: int
    number_of_lines: int
    projection: common.ProjectionType
    range_coordinate_conversion: list[CoordinateConversionType]
    datum: common.DatumType
    footprint: list[float]
    pixel_representation: common.PixelRepresentationType
    pixel_type: common.PixelTypeType
    pixel_quantity: common.PixelQuantityType
    no_data_value: float


@dataclass
class IonosphereCorrection:
    "Ionosphere correction"

    ionosphere_height_used: float
    ionosphere_height_estimated: float
    ionosphere_height_estimation_method_selected: common.IonosphereHeightEstimationMethodType
    ionosphere_height_estimation_latitude_value: float
    ionosphere_height_estimation_flag: bool
    ionosphere_height_estimation_method_used: common.IonosphereHeightEstimationMethodType
    gaussian_filter_computation_flag: bool
    faraday_rotation_correction_applied: bool
    autofocus_shifts_applied: bool


@dataclass
class DopplerParametersType:
    """Doppler parameters"""

    dc_estimate_list: list[DcEstimateType]
    fm_rate_estimate_list: list[common.SlantRangePolynomialType]


@dataclass
class ErrorCountersType:
    """Error counters"""

    num_isp_header_errors: int
    num_isp_missing: int


@dataclass
class RawDataStatisticsType:
    """raw data statistics"""

    i_bias: float
    q_bias: float
    iq_quadrature_departure: float
    iq_gain_imbalance: float
    polarisation: common.PolarisationType


@dataclass
class RawDataAnalysisType:
    """Raw data analysis"""

    error_counters: ErrorCountersType
    raw_data_statistics_list: list[RawDataStatisticsType]


@dataclass
class InternalCalibrationSequenceType:
    """Internal calibration sequence"""

    azimuth_time: PreciseDateTime
    drift_amplitude: float
    drift_phase: float
    model_drift_amplitude: float
    model_drift_phase: float
    relative_drift_valid_flag: bool
    absolute_drift_valid_flag: bool
    cross_correlation_bandwidth: float
    cross_correlation_pslr: float
    cross_correlation_islr: float
    cross_correlation_peak_location: float
    reconstructed_replica_valid_flag: bool
    internal_time_delay: float
    internal_tx_channel_imbalance_amplitude: float
    internal_tx_channel_imbalance_phase: float
    internal_rx_channel_imbalance_amplitude: float
    internal_rx_channel_imbalance_phase: float
    transmit_power_tracking_d1_amplitude: float
    transmit_power_tracking_d1_phase: float
    receive_power_tracking_d1_amplitude: float
    receive_power_tracking_d1_phase: float
    transmit_power_tracking_d2_amplitude: float
    transmit_power_tracking_d2_phase: float
    receive_power_tracking_d2_amplitude: float
    receive_power_tracking_d2_phase: float


@dataclass
class NoiseSequenceType:
    """Noise sequence"""

    azimuth_time: PreciseDateTime
    noise_power_correction_factor: float
    number_of_noise_lines: int


@dataclass
class InternalCalibrationType:
    """Internal calibration information"""

    internal_calibration_parameters_used: common.InternalCalibrationSourceType
    range_reference_function_used: common.RangeReferenceFunctionType
    noise_parameters_used: common.InternalCalibrationSourceType

    internal_calibration_parameters_list: dict[common.PolarisationType, list[InternalCalibrationSequenceType]]
    noise_list: dict[common.PolarisationType, list[NoiseSequenceType]]


@dataclass
class RfiTmreportType:
    """RFI time domain report"""

    percentage_affected_lines: float
    avg_percentage_affected_samples: float
    max_percentage_affected_samples: float
    polarisation: common.PolarisationType


@dataclass
class RfiIsolatedFmreportType:
    """Isolated FM report"""

    percentage_affected_lines: float
    max_percentage_affected_bw: float
    avg_percentage_affected_bw: float
    polarisation: common.PolarisationType


@dataclass
class RfiPersistentFmreportType:
    """RFI persistent FM report"""

    percentage_affected_lines: float
    max_percentage_affected_bw: float
    avg_percentage_affected_bw: float
    polarisation: common.PolarisationType


@dataclass
class RfiMitigationType:
    """RFI mitigation annotation"""

    rfi_tmreport_list: list[RfiTmreportType]
    rfi_isolated_fmreport_list: list[RfiIsolatedFmreportType]
    rfi_persistent_fmreport_list: list[RfiPersistentFmreportType]


@dataclass
class RadiometricCalibrationType:
    """Radiometric calibration section"""

    absolute_calibration_constant_list: dict[common.PolarisationType, float]


@dataclass
class GeometryType:
    """Geometry annotation"""

    height_model: common.HeightModelType

    height_model_used_flag: bool
    roll_bias: float


@dataclass
class QualityParametersType:
    """Quality parameters"""

    missing_ispfraction: float
    max_ispgap: int
    max_ispgap_threshold: int
    invalid_raw_data_samples: float
    raw_mean_expected: float
    raw_mean_threshold: float
    raw_std_expected: float
    raw_std_threshold: float
    rfi_tmfraction: float
    max_rfitmpercentage: float
    rfi_fmfraction: float
    max_rfifmpercentage: float
    invalid_drift_fraction: float
    max_invalid_drift_fraction: float
    invalid_replica_fraction: float
    invalid_dcestimates_fraction: float
    dc_rmserror_threshold: float
    residual_ionospheric_phase_screen_std: float
    invalid_blocks_percentage: float
    invalid_blocks_percentage_threshold: float
    polarisation: common.PolarisationType


@dataclass
class QualityType:
    """Quality"""

    overall_product_quality_index: int
    quality_parameters_list: list[QualityParametersType]


@dataclass
class PolarimetricDistortionType:
    """Polarimetric distortion"""

    cross_talk: common.CrossTalkList
    channel_imbalance: common.ChannelImbalanceList
