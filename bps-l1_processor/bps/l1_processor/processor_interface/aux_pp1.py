# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP1
-------
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from bps.common import Polarization, Swath, bps_logger
from bps.common.io import common


class RGBChannel(Enum):
    """RGB channels"""

    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"


@dataclass
class GeneralConf:
    """High level configuration"""

    class EarthModel(Enum):
        """Earth model"""

        ELLIPSOID = "ELLIPSOID"
        SRTM = "SRTM"
        COPERNICUS_DEM = "COPERNICUS_DEM"

    class OutputProductFormat(Enum):
        """Output product format"""

        OFFICIAL = "OFFICIAL"
        INTERNAL = "INTERNAL"

    requested_height_model: EarthModel
    requested_height_model_version: str
    height_model: EarthModel
    height_model_version: str
    height_model_margin: float
    parc_roi_samples: int
    parc_roi_lines: int
    dual_polarisation_processing_flag: bool


@dataclass
class L0ProductImportConf:
    """L0 importer configuration"""

    block_size: int
    max_isp_gap: int
    raw_mean_expected: float
    raw_mean_threshold: float
    raw_std_expected: float
    raw_std_threshold: float
    internal_calibration_estimation_flag: bool


@dataclass
class RawDataCorrectionConf:
    """Raw data correction configuration"""

    bias_correction_flag: bool
    gain_imbalance_correction_flag: bool
    non_orthogonality_correction_flag: bool
    raw_data_correction_flag: bool


class ChirpSource(Enum):
    """Chirp source options"""

    NOMINAL = "NOMINAL"
    REPLICA = "REPLICA"
    INTERNAL = "INTERNAL"


RFIActivationMode = Literal["Enabled", "Disabled", "MaskBased"]


@dataclass
class RFIMitigationConf:
    """RFI mitigation configuration"""

    class MitigationMethod(Enum):
        """Mitigation algorithm"""

        TIME = "TIME"
        FREQUENCY = "FREQUENCY"
        TIME_AND_FREQUENCY = "TIME_AND_FREQUENCY"
        FREQUENCY_AND_TIME = "FREQUENCY_AND_TIME"

    class MaskType(Enum):
        """Type of mask"""

        SINGLE = "SINGLE"
        MULTIPLE = "MULTIPLE"

    class MaskGenerationMethod(Enum):
        """Mask generation strategy"""

        AND = "AND"
        OR = "OR"

    @dataclass
    class TimeDomainParams:
        """Time domain method parameters"""

        block_lines: int
        median_filter_length: int
        box_samples: int
        box_lines: int
        percentile_threshold: float
        morphological_open_operator_samples: int
        morphological_open_operator_lines: int
        morphological_close_operator_samples: int
        morphological_close_operator_lines: int
        max_rfi_percentage: float

    @dataclass
    class FreqDomainParams:
        """Frequenct domain method parameters"""

        block_lines: int
        block_overlap: int
        persistent_rfi_threshold: float
        isolated_rfi_threshold: float
        isolated_rfi_psd_std_threshold: float
        max_rfi_percentage: float
        periodgram_size: int
        enable_power_loss_compensation: bool
        power_loss_threshold: float
        chirp_source: ChirpSource
        mitigation_method: Literal["NOTCH_FILTER", "NEAREST_NEIGHBOUR_INTERPOLATION"]

    detection_flag: bool
    activation_mode: RFIActivationMode
    activation_mask_threshold: float
    mitigation_method: MitigationMethod
    mask: MaskType
    mask_generation_method: MaskGenerationMethod
    time_domain_processing_parameters: TimeDomainParams
    freq_domain_processing_parameters: FreqDomainParams


@dataclass
class InternalCalibrationConf:
    """Internal calibration configuration"""

    class Source(Enum):
        """Internal calibration source"""

        EXTRACTED = "EXTRACTED"
        MODEL = "MODEL"

    internal_calibration_correction_flag: bool
    drift_correction_flag: bool
    delay_correction_flag: bool
    channel_imbalance_correction_flag: bool
    internal_calibration_source: Source
    max_drift_amplitude_std_fraction: float
    max_drift_phase_std_fraction: float
    max_drift_amplitude_error: float
    max_drift_phase_error: float
    max_invalid_drift_fraction: float


class WindowType(Enum):
    """Type of windows"""

    HAMMING = "HAMMING"
    KAISER = "KAISER"
    NONE = "NONE"


@dataclass
class RangeCompressionConf:
    """Range focuser configuration"""

    class Method(Enum):
        """Range focusing method"""

        MATCHED_FILTER = "MATCHED_FILTER"
        INVERSE_FILTER = "INVERSE_FILTER"

    @dataclass
    class Parameters:
        """Swath specific parameters"""

        time_bias: float
        window_type: WindowType
        window_coefficient: float
        processing_bandwidth: float

    range_reference_function_source: ChirpSource
    range_compression_method: Method
    extended_swath_processing: bool
    parameters: dict[Swath, Parameters]


@dataclass
class DopplerEstimationConf:
    """Doppler estimators configuration"""

    class Method(Enum):
        """Doppler estimation strategy"""

        GEOMETRY = "GEOMETRY"
        COMBINED = "COMBINED"
        FIXED = "FIXED"

    method: Method
    value: float
    block_samples: int
    block_lines: int
    polynomial_update_rate: float
    rms_error_threshold: float


@dataclass
class AntennaPatternCorrectionConf:
    """Antenna pattern compensation configuration"""

    antenna_pattern_correction1_flag: bool
    antenna_pattern_correction2_flag: bool
    antenna_cross_talk_correction_flag: bool
    elevation_mispointing_bias: float
    azimuth_mispointing_bias: float


@dataclass
class AzimuthCompressionConf:
    """Azimuth focuser configuration"""

    @dataclass
    class Parameters:
        """Swath specific parameters"""

        time_bias: float
        window_type: WindowType
        window_coefficient: float
        processing_bandwidth: float

    class Method(Enum):
        """Bistatic delay correction method"""

        BULK = "BULK"
        FULL = "FULL"

    block_samples: int
    block_lines: int
    block_overlap_samples: int
    block_overlap_lines: int
    parameters: dict[Swath, Parameters]
    bistatic_delay_correction: bool
    bistatic_delay_correction_method: Method
    azimuth_resampling: bool
    azimuth_resampling_frequency: float
    azimuth_focusing_margins_removal_flag: bool
    azimuth_coregistration_flag: bool
    filter_type: str
    filter_bandwidth: float
    filter_length: int
    number_of_filters: int


@dataclass
class RadiometricCalibrationConf:
    """Radiometric calibration configuration"""

    absolute_calibration_constant: dict[Polarization, float]
    processing_gain: dict[Polarization, float]
    reference_range: float
    range_spreading_loss_compensation_enabled: bool


@dataclass
class PolarimetricCalibrationConf:
    """Polarimetric calibration configuration"""

    polarimetric_correction_flag: bool
    tx_distortion_matrix_correction_flag: bool
    rx_distortion_matrix_correction_flag: bool
    cross_talk_correction_flag: bool
    cross_talk: common.CrossTalkList
    channel_imbalance_correction_flag: bool
    channel_imbalance: common.ChannelImbalanceList


@dataclass
class IonosphereCalibrationConf:
    """Ionosphere calibration configuration"""

    class Method(Enum):
        """Ionosphere calibration method"""

        AUTOMATIC = "AUTOMATIC"
        FEATURE_TRACKING = "FEATURE_TRACKING"
        SQUINT_SENSITIVITY = "SQUINT_SENSITIVITY"
        MODEL = "MODEL"
        FIXED = "FIXED"

    block_lines: int
    block_overlap_lines: int
    ionosphere_height_defocusing_flag: bool
    ionosphere_height_estimation_method: Method
    ionosphere_height_value: float
    ionosphere_height_estimation_method_latitude_threshold: float
    ionosphere_height_minimum_value: float
    ionosphere_height_maximum_value: float
    squint_sensitivity_number_of_looks: int
    squint_sensitivity_number_of_ticks: int
    squint_sensitivity_fitting_error: list[float]
    gaussian_filter_maximum_major_axis_length: int
    gaussian_filter_maximum_minor_axis_length: int
    gaussian_filter_major_axis_length: int
    gaussian_filter_minor_axis_length: int
    gaussian_filter_slope: float
    faraday_rotation_correction_flag: bool
    ionospheric_phase_screen_correction_flag: bool
    group_delay_correction_flag: bool


@dataclass
class AutofocusConf:
    """Autofocusing configuration"""

    class Method(Enum):
        """Autofocus method"""

        MAP_DRIFT = "MAP_DRIFT"
        PGA = "PGA"

    autofocus_flag: bool
    autofocus_method: Method
    map_drift_azimuth_sub_bands: int
    map_drift_correlation_window_width: int
    map_drift_correlation_window_height: int
    map_drift_range_correlation_windows: int
    map_drift_azimuth_correlation_windows: int
    max_valid_shift: float
    valid_blocks_percentage: float


@dataclass
class MultilookConf:
    """Multilooker configuration"""

    @dataclass
    class Parameters:
        """Swath and direction specific parameters"""

        window_type: WindowType
        window_coefficient: float
        look_bandwidth: float
        number_of_looks: int
        look_central_frequencies: list[float]
        upsampling_factor: int
        downsampling_factor: int

    range_parameters: dict[Swath, Parameters]
    azimuth_parameters: dict[Swath, Parameters]
    apply_detection: bool


@dataclass
class ThermalDenoisingConf:
    """Thermal denoising configuration"""

    class Source(Enum):
        """Noise parameters source"""

        EXTRACTED = "EXTRACTED"
        MODEL = "MODEL"

    thermal_denoising_flag: bool
    noise_parameters_source: Source
    noise_equivalent_echoes_flag: bool
    noise_gain_list: dict[Polarization, float]


@dataclass
class GroundProjectionConf:
    """Ground projection configuration"""

    class FilterType(Enum):
        """Filter type"""

        SINC = "SINC"
        GLS = "GLS"

    ground_projection_flag: bool
    range_pixel_spacing: float
    filter_type: FilterType
    filter_bandwidth: float
    filter_length: int
    number_of_filters: int


@dataclass
class L1ProductExportConf:
    """L1 product exporter configuration"""

    class PixelType(Enum):
        """Pixel type"""

        FLOAT32 = "FLOAT32"
        SINT16 = "SINT16"
        UINT16 = "UINT16"

    class CompressionMethodType(Enum):
        """TIFF compression methods"""

        NONE = "NONE"
        DEFLATE = "DEFLATE"
        ZSTD = "ZSTD"
        LERC = "LERC"
        LERC_DEFLATE = "LERC_DEFLATE"
        LERC_ZSTD = "LERC_ZSTD"

    @dataclass
    class LutDecimationFactors:
        """LUT factors by group"""

        dem_based_quantity: int
        rfi_based_quantity: int
        image_based_quantity: int

    l1a_product_doi: str
    l1b_product_doi: str
    pixel_representation: common.PixelRepresentationType
    pixel_quantity: common.PixelQuantityType
    abs_compression_method: CompressionMethodType
    abs_max_zerror: float
    abs_max_zerror_percentile: float
    phase_compression_method: CompressionMethodType
    phase_max_zerror: float
    phase_max_zerror_percentile: float
    no_pixel_value: float
    block_size: int
    lut_range_decimation_factor: LutDecimationFactors
    lut_azimuth_decimation_factor: LutDecimationFactors
    lut_block_size: int
    lut_layers_completeness_flag: bool
    ql_range_decimation_factor: int
    ql_range_averaging_factor: int
    ql_azimuth_decimation_factor: int
    ql_azimuth_averaging_factor: int
    ql_absolute_scaling_factor_list: dict[RGBChannel, float]


@dataclass
class AuxProcessingParametersL1:
    """BPS L1 Processing parameters"""

    product_id: str
    general: GeneralConf
    l0_product_import: L0ProductImportConf
    raw_data_correction: RawDataCorrectionConf
    rfi_mitigation: RFIMitigationConf
    internal_calibration_correction: InternalCalibrationConf
    range_compression: RangeCompressionConf
    doppler_estimation: DopplerEstimationConf
    antenna_pattern_correction: AntennaPatternCorrectionConf
    azimuth_compression: AzimuthCompressionConf
    radiometric_calibration: RadiometricCalibrationConf
    polarimetric_calibration: PolarimetricCalibrationConf
    ionosphere_calibration: IonosphereCalibrationConf
    autofocus: AutofocusConf
    multilook: MultilookConf
    thermal_denoising: ThermalDenoisingConf
    ground_projection: GroundProjectionConf
    l1_product_export: L1ProductExportConf

    def is_ionospheric_calibration_enabled(self) -> bool:
        """Wether ionospheric_calibration is enabled"""
        return self.polarimetric_calibration.polarimetric_correction_flag and (
            self.ionosphere_calibration.faraday_rotation_correction_flag
            or self.ionosphere_calibration.group_delay_correction_flag
            or self.ionosphere_calibration.ionospheric_phase_screen_correction_flag
        )

    def raise_if_inconsistent(self) -> None:
        """raise if the aux pp1 is not valid"""
        valid = True

        # check rfi configuration
        rfi_may_be_enabled = self.rfi_mitigation.activation_mode != "Disabled" or self.rfi_mitigation.detection_flag
        if rfi_may_be_enabled:
            frequency_domain_enabled = self.rfi_mitigation.mitigation_method in (
                RFIMitigationConf.MitigationMethod.FREQUENCY,
                RFIMitigationConf.MitigationMethod.FREQUENCY_AND_TIME,
                RFIMitigationConf.MitigationMethod.TIME_AND_FREQUENCY,
            )
            if frequency_domain_enabled:
                rfi_chirp_source = self.rfi_mitigation.freq_domain_processing_parameters.chirp_source
                range_comp_chirp_source = self.range_compression.range_reference_function_source

                if rfi_chirp_source != range_comp_chirp_source and rfi_chirp_source != ChirpSource.NOMINAL:
                    valid = False
                    rgc_chirp_tag = "rangeCompression/rangeReferenceFunctionSource"
                    rfi_chirp_tag = "rfiFMProcessingParameters/chirpSource"
                    condition = f" with range compression chirp ({rgc_chirp_tag}={range_comp_chirp_source.value})"
                    invalid_field = f"RFI frequence domain chirp ({rfi_chirp_tag}={rfi_chirp_source.value})"
                    bps_logger.error(invalid_field + condition)

        # check int cal configuration
        intcal_estimations_enabled = self.l0_product_import.internal_calibration_estimation_flag
        intcal_corrections_enabled = self.internal_calibration_correction.internal_calibration_correction_flag
        intcal_estimations_enabled_tag = "'l0ProductImport/internalCalibrationEstimationFlag'"
        intcal_corrections_enabled_tag = "'internalCalibrationCorrection/internalCalibrationCorrectionFlag'"

        if not intcal_estimations_enabled and intcal_corrections_enabled:
            valid = False
            condition = f" with disabled estimations ({intcal_estimations_enabled_tag}=false)"
            invalid_field = f"Internal calibration corrections enabled ({intcal_corrections_enabled_tag}=true)"
            bps_logger.error(invalid_field + condition)

        if not intcal_corrections_enabled:
            condition = f" with disabled corrections ({intcal_corrections_enabled_tag}=false)"
            if self.internal_calibration_correction.drift_correction_flag:
                valid = False
                drift_correction_tag = "'internalCalibrationCorrection/driftCorrectionFlag'"
                invalid_field = f"Drift correction enabled ({drift_correction_tag}=true)"
                bps_logger.error(invalid_field + condition)

            if self.internal_calibration_correction.delay_correction_flag:
                valid = False
                drift_correction_tag = "'internalCalibrationCorrection/delayCorrectionFlag'"
                invalid_field = f"Delay correction enabled ({drift_correction_tag}=true)"
                bps_logger.error(invalid_field + condition)

            if self.internal_calibration_correction.channel_imbalance_correction_flag:
                valid = False
                drift_correction_tag = "'internalCalibrationCorrection/channelImbalanceCorrectionFlag'"
                invalid_field = f"Channel imbalance correction enabled ({drift_correction_tag}=true)"
                bps_logger.error(invalid_field + condition)

            if self.range_compression.range_reference_function_source == ChirpSource.REPLICA:
                valid = False
                drift_correction_tag = "'rangeCompression/rangeReferenceFunctionSource'"
                invalid_field = f"Range reference function set to REPLICA ({drift_correction_tag}=REPLICA)"
                bps_logger.error(invalid_field + condition)

        # check ionospheric calibration and autofocus configuration
        autofocus_enabled = self.autofocus.autofocus_flag
        autofocus_enabled_tag = "'autofocus/autofocusFlag'"
        ionospheric_calibration_enabled = self.is_ionospheric_calibration_enabled()
        ionospheric_calibration_enabled_tag = "'ionosphereCalibration/*CorrectionFlag'"
        ionospheric_phase_screen_correction_enabled = (
            self.ionosphere_calibration.ionospheric_phase_screen_correction_flag
        )
        ionospheric_phase_screen_correction_enabled_tag = (
            "'ionosphereCalibration/*ionosphericPhaseScreenCorrectionFlag'"
        )
        if autofocus_enabled and not ionospheric_calibration_enabled:
            valid = False
            condition = f" with disabled ionospheric calibration ({ionospheric_calibration_enabled_tag}=false)"
            invalid_field = f"Autofocus enabled ({autofocus_enabled_tag}=true)"
            bps_logger.error(invalid_field + condition)
        if autofocus_enabled and ionospheric_phase_screen_correction_enabled:
            valid = False
            condition = f" with enabled ionospheric phase screen correction ({ionospheric_phase_screen_correction_enabled_tag}=true)"
            invalid_field = f"Autofocus enabled ({autofocus_enabled_tag}=true)"
            bps_logger.error(invalid_field + condition)

        if not valid:
            raise RuntimeError("AuxPP1 is not consistent")

    def switch_off_steps_requiring_quad_pol_data(self):
        """Switch off steps that requires quad pol data"""
        if self.polarimetric_calibration.polarimetric_correction_flag:
            bps_logger.warning("Input product is not quad pol: polarimetric calibration is switched off")
            self.polarimetric_calibration.polarimetric_correction_flag = False
        if self.is_ionospheric_calibration_enabled():
            bps_logger.warning("Input product is not quad pol: ionospheric calibration is switched off")
            self.ionosphere_calibration.faraday_rotation_correction_flag = False
            self.ionosphere_calibration.group_delay_correction_flag = False
            self.ionosphere_calibration.ionospheric_phase_screen_correction_flag = False
