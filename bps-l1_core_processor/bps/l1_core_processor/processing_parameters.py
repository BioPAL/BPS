# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 processing parameters
------------------------
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


@dataclass
class Quantity:
    """Quantity"""

    class Unit(Enum):
        """Unit of measurement"""

        NORMALIZED = "Normalized"
        HZ = "Hz"
        S = "s"

    value: float
    unit: Unit


@dataclass
class WindowConf:
    """Window configuration"""

    class Type(Enum):
        """Window type"""

        HAMMING = "HAMMING"
        KAISER = "KAISER"

    window_type: Type
    window_parameter: float
    window_look_bandwidth: Quantity
    window_transition_bandwidth: Quantity


class BorderPolicy(Enum):
    """Border Policy"""

    CUT = "CUT"
    DATA = "DATA"
    PAD = "PAD"


@dataclass
class RFIMitigationConf:
    """RFI mitigation step configuration"""

    class Method(Enum):
        """RFI mitigation algorithms"""

        FREQUENCY = "FREQUENCY"
        TIME = "TIME"
        TIME_AND_FREQUENCY = "TIME_AND_FREQUENCY"
        FREQUENCY_AND_TIME = "FREQUENCY_AND_TIME"

    class MaskCompositionMethod(Enum):
        """RFI mask composition method"""

        AND = "AND"
        OR = "OR"
        NONE = "NONE"

    @dataclass
    class TimeDomainConf:
        """Time domain method configuration"""

        class CorrectionMode(Enum):
            """Correction modes"""

            NEAREST = "NEAREST"
            ZERO = "ZERO"
            GAUSSNOISE = "GAUSSNOISE"

        correction_mode: CorrectionMode
        """Correction mode"""

        percentile_threshold: float
        median_filter_block_lines: int
        lines_in_estimate_block: int
        box_filter_azimuth_dimension: int
        box_filter_range_dimension: int
        morph_open_line_length: int
        morph_close_line_length: int
        morph_close_before_open: bool | None = None
        morph_open_close_iterations: int | None = None

        swath: str | None = None
        """Swath name"""

    @dataclass
    class FrequencyDomainConf:
        """Frequency domain method configuration"""

        block_size: int
        periodgram_size: int
        persistent_rfi_threshold: float
        isolated_rfi_threshold: float
        threshold_std: float
        percentile_low: float
        percentile_high: float
        power_loss_threshold: float | None
        remove_interferences: bool | None
        filtering_mode: Literal["NOTCH", "NOISE", "NEAREST_NEIGHBOUR"] = "NOTCH"

        swath: str | None = None
        """Swath name"""

    rfi_mitigation_method: Method
    """RFI mitigation method"""
    rfi_mask_composition_method: MaskCompositionMethod
    """RFI mask composition method"""
    time_domain_conf: TimeDomainConf | None
    """Time domain method configuration"""
    frequency_domain_conf: FrequencyDomainConf | None
    """Frequency domain method configuration"""
    swath: str | None = None
    """Swath name"""


@dataclass
class RangeFocuserConf:
    """Range focuser step configuration"""

    class Method(Enum):
        """Range focusing method"""

        MATCHED_FILTER = "MATCHED_FILTER"
        INVERSE_FILTER = "INVERSE_FILTER"
        INVERSE_FFT = "INVERSE_FFT"

    flag_ortog: bool
    apply_range_spectral_weighting_window: bool
    range_spectral_weighting_window: WindowConf
    swst_bias: float | None = None
    range_decimation_factor: int | None = None
    apply_rx_gain_correction: bool | None = None
    focusing_method: Method | None = None
    output_prf_value: float | None = None
    output_range_border_policy: BorderPolicy | None = None
    swath: str | None = None
    """Swath name"""


@dataclass
class DopplerEstimatorStripmapConf:
    """Doppler estimation step configuration"""

    class Method(Enum):
        """Estimation method"""

        GEOMETRICAL = "GEOMETRICAL"
        DATA = "DATA"
        COMBINED = "COMBINED"

    class AttitudeFitting(Enum):
        """Attitude fitting strategy"""

        LINEAR = "LINEAR"
        AVERAGE = "AVERAGE"
        DISABLED = "DISABLED"

    class PolyEstimationConstraint(Enum):
        "Constraint for polynomial estimation"

        FULL = "FULL"
        UNCONSTRAINED = "UNCONSTRAINED"

    @dataclass
    class DcCoreAlgorithm:
        """Core algorithm parameters"""

        ...

    blocks: int
    blockl: int
    undersampling_snrd_cazimuth_ratio: int
    undersampling_snrd_crange_ratio: int
    az_max_frequency_search_bin_number: int
    rg_max_frequency_search_bin_number: int
    az_max_frequency_search_norm_band: float
    rg_max_frequency_search_norm_band: float
    nummlbf: int
    nbestblocks: int
    rg_band: float
    an_len: float
    lookbf: float
    lookbt: float
    lookrp: float
    lookrs: float
    decfac: int
    flength: int
    dftstep: float
    peakwid: float
    minamb: float
    maxamb: float
    sthr: float
    varth: float
    pol_weights: list[int]
    dc_estimation_method: Method
    attitude_fitting: AttitudeFitting
    poly_changing_freq: float | None = None
    poly_estimation_constraint: PolyEstimationConstraint | None = None
    dc_core_algorithm: DcCoreAlgorithm | None = None
    joint_estimation: bool | None = True
    swath: str | None = None
    """Swath name"""


@dataclass
class AzimuthConf:
    """Azimuth compression step configuration"""

    class Method(Enum):
        """Azimuth focusing method"""

        WK = "WK"
        CZT = "CZT"
        BP = "BP"

    class BistaticDelayCorrectionMode(Enum):
        """Bistatic delay options"""

        BIAS_ONLY = "BIAS_ONLY"
        NEAR_RANGE = "NEAR_RANGE"
        MIDDLE_RANGE = "MIDDLE_RANGE"
        SCENE_CENTER = "SCENE_CENTER"
        RANGE_DEPENDENT = "RANGE_DEPENDENT"

    class AntennaShiftCompensationMode(Enum):
        """Antenna shift compensation options"""

        DISABLED = "DISABLED"
        FORCED = "FORCED"

    lines_in_block: int
    samples_in_block: int
    azimuth_overlap: int
    range_overlap: int
    perform_interpolation: int
    stolt_padding: float
    range_modulation: bool
    apply_azimuth_spectral_weighting_window: bool
    azimuth_spectral_weighting_window: WindowConf
    apply_rg_shift: bool
    apply_az_shift: bool
    whitening_flag: bool
    antenna_length: float
    pad_result: int
    lines_to_skip_dc_fr: int | None = None
    samples_to_skip_dc_fr: int | None = None
    focusing_method: Method | None = None
    az_proc_bandwidth: Quantity | None = None
    bistatic_delay_correction_mode: BistaticDelayCorrectionMode | None = None
    azimuth_time_bias: float | None = None
    apply_pol_channels_coregistration: bool | None = None
    antenna_shift_compensation_mode: AntennaShiftCompensationMode | None = None
    nominal_block_memory_size_cpu: int | None = None
    """MByte"""
    nominal_block_memory_size_gpu: int | None = None
    """MByte"""
    swath: str | None = None
    """Swath name"""


@dataclass
class RadiometricCalibrationConf:
    """Radiometric calibration step configuration"""

    class OutputQuantity(Enum):
        """Output data content"""

        BETA = "BETA"
        SIGMA = "SIGMA"
        GAMMA = "GAMMA"

    rsl_reference_distance: float
    perform_rsl_compensation: bool
    perform_pattern_compensation: bool
    external_calibration_factor: complex
    apply_external_calibration_factor: bool | None = None
    output_quantity: OutputQuantity | None = None
    perform_line_correction: bool | None = None
    fast_mode: bool | None = None
    processing_gain: complex | None = None
    swath: str | None = None
    """Swath name"""


@dataclass
class PolarimetricProcessorConf:
    """Polarimetric step configuration"""

    enable_cross_talk_compensation: bool
    enable_channel_imbalance_compensation: bool
    swath: str | None = None
    """Swath name"""


class IonosphericHeightEstimationMethod(Enum):
    """Estimation method for ionospheric height estimation"""

    NONE = "None"
    FEATURE_TRACKING = "FeatureTracking"
    SQUINT_SENSITIVITY = "SquintSensitivity"
    MODEL = "Model"
    AUTO = "Auto"


@dataclass
class IonosphericSquintSensitivity:
    """Ionospheric squint sensitivity params"""

    number_of_looks: int
    height_step: float
    faraday_rotation_bias: float


@dataclass
class IonosphericFeatureTracking:
    """Ionospheric feature tacking params"""

    max_offset: int
    profile_step: int
    normalized_min_value_threshold: float


@dataclass
class IonosphericCalibrationConf:
    """Ionospheric calibration step configuration"""

    perform_defocusing_on_ionospheric_height: bool
    perform_faraday_rotation_correction: bool
    perform_phase_screen_correction: bool
    perform_group_delay_correction: bool
    ionospheric_height_estimation_method: IonosphericHeightEstimationMethod
    squint_sensitivity: IonosphericSquintSensitivity | None
    feature_tracking: IonosphericFeatureTracking | None
    z_threshold: float
    gaussian_filter_max_size_azimuth: int
    gaussian_filter_max_size_range: int
    gaussian_filter_default_size_azimuth: int
    gaussian_filter_default_size_range: int
    default_ionospheric_height: float
    max_ionospheric_height: float
    min_ionospheric_height: float
    azimuth_block_size: int
    azimuth_block_overlap: int

    swath: str | None = None
    """Swath name"""


@dataclass
class CalibrationConstantsConf:
    """Polarimetric step configuration"""

    channel_imbalance_tx: complex
    channel_imbalance_rx: complex
    cross_talk_hv_rx: complex
    cross_talk_vh_rx: complex
    cross_talk_vh_tx: complex
    cross_talk_hv_tx: complex
    internal_delay_hh: float
    internal_delay_hv: float
    internal_delay_vh: float
    internal_delay_vv: float


@dataclass
class MultilookerConf:
    """Multilooker step configuration"""

    @dataclass
    class WindowInfo:
        """Windowing info"""

        apply: bool
        window: WindowConf

    @dataclass
    class MultilookerDoubleDirectionConf:
        """Multilooker configuration for both direction"""

        @dataclass
        class MultilookerSingleDirectionConf:
            """Multilooker configuration for one direction"""

            p_factor: int
            q_factor: int
            weighting_window: WindowConf
            central_frequency: list[Quantity]

            def __setattr__(self, prop, value):
                if prop == "central_frequency":
                    if not self._is_central_frequency_unit_valid(value):
                        raise ValueError("Invalid central frequency unit")

                super().__setattr__(prop, value)

            @staticmethod
            def _is_central_frequency_unit_valid(
                central_frequency: list[Quantity],
            ) -> bool:
                valid_units = [Quantity.Unit.HZ, Quantity.Unit.NORMALIZED]

                return all(frequency.unit in valid_units for frequency in central_frequency)

        slow_multilook: MultilookerSingleDirectionConf | None
        fast_multilook: MultilookerSingleDirectionConf | None

    @dataclass
    class PresumConf:
        """Presum configuration for both direction"""

        fast_factor: int
        slow_factor: int

    multilook_conf_name: str
    azimuth_time_weighting_window_info: WindowInfo | None
    normalization_factor: float | None
    multilook_conf: MultilookerDoubleDirectionConf | PresumConf
    invalid_value: complex | None
    swath: str | None = None
    """Swath name"""


@dataclass
class NoiseMapConf:
    """Noise map generator configuration"""

    noise_normalization_constant: float
    swath: str | None = None
    """Swath name"""


@dataclass
class SlantToGroundConf:
    """Slant to ground configuration"""

    ground_step: float
    invalid_value: complex | None = None
    swath: str | None = None
    """Swath name"""


@dataclass
class SarfocProcessingParameters:
    """Sarfoc processing steps configurations"""

    rfi_mitigation_conf: list[RFIMitigationConf]
    range_focuser_conf: list[RangeFocuserConf]
    doppler_estimator_conf: list[DopplerEstimatorStripmapConf]
    azimuth_conf: list[AzimuthConf]
    radiometric_calibration_conf: list[RadiometricCalibrationConf]
    polarimetric_processor_conf: list[PolarimetricProcessorConf]
    ionospheric_calibration_conf: list[IonosphericCalibrationConf]
    calibration_constants_conf: CalibrationConstantsConf
    multilooker_conf: list[MultilookerConf]
    noise_map_conf: list[NoiseMapConf]
    slant_to_ground_conf: list[SlantToGroundConf]
