# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2A
----------
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from bps.l2a_processor.io.aux_pp2_2a_models import CalibrationScreenType
from bps.transcoder.io import common_annotation_models_l2


@dataclass
class GeneralConf:
    """High level configuration"""

    class SubsettingRules(Enum):
        """TOM acquisitons subsetting rules"""

        GEOMETRY = "geometry"
        MAINTAIN_ALL = "maintain all"

    apply_calibration_screen: CalibrationScreenType
    forest_coverage_threshold: float
    forest_mask_interpolation_threshold: float
    subsetting_rule: SubsettingRules


class OperationalModeType(Enum):
    """Possible operational modes for ground cancellation algorithm"""

    MULTI_REFERENCE = "multi reference"
    SINGLE_REFERENCE = "single reference"
    INSAR_PAIR = "insar pair"


@dataclass
class GroundCancellationConfFD:
    """Ground Cancellation configurations for FD"""

    emphasized_forest_height: float
    operational_mode: OperationalModeType
    images_pair_selection: common_annotation_models_l2.AcquisitionListType
    disable_ground_cancellation_flag: bool


@dataclass
class GroundCancellationConfAGB(GroundCancellationConfFD):
    """Ground Cancellation configurations for GN (AGB)"""

    compute_gn_power_flag: bool
    radiometric_calibration_flag: bool


@dataclass
class MinMaxType:
    """Min Max values"""

    min: float
    max: float


@dataclass
class MinMaxNumType:
    """Min Max Number values"""

    min: float
    max: float
    num: int


@dataclass
class verticalRangeWithUnitsType:
    """Min Max sampling values"""

    min: float
    max: float
    sampling: int


@dataclass
class IntCompressionType:
    """Compression options for MDS and ADS integer images"""

    compression_factor: int


@dataclass
class MDSfloatCompressionType:
    """Compression options for MDS float images"""

    compression_factor: int
    max_z_error: float


@dataclass
class ADSfloatCompressionType:
    """Compression options for ADS float images"""

    compression_factor: int
    least_significant_digit: int


@dataclass
class L2aAGBConf:
    """AGB L2a configuration"""

    @dataclass
    class CompressionConf:
        """Compression configuration for all AGB L2a products: MDS and ADS"""

        @dataclass
        class MDS:
            """Compression configuration for AGB L2a MDS"""

            gn: MDSfloatCompressionType

        @dataclass
        class ADS:
            """Compression configuration for AGB L2a ADS"""

            fnf: IntCompressionType
            incidence_angle: ADSfloatCompressionType

        mds: MDS
        ads: ADS
        mds_block_size: int
        ads_block_size: int

    l2aAGBProductDOI: str
    product_id: str
    enable_product_flag: bool
    ground_cancellaton: GroundCancellationConfAGB
    product_resolution: float
    upsampling_factor: int
    compression_options: CompressionConf


@dataclass
class L2aFDConf:
    """FD L2a configuration"""

    @dataclass
    class CompressionConf:
        """Compression configuration for all FD L2a products: MDS and ADS"""

        @dataclass
        class MDS:
            """Compression configuration for FD L2a MDS"""

            fd: IntCompressionType
            probability_of_change: MDSfloatCompressionType
            cfm: IntCompressionType

        @dataclass
        class ADS:
            """Compression configuration for FD L2a ADS"""

            fnf: IntCompressionType
            acm: ADSfloatCompressionType
            number_of_averages: IntCompressionType

        mds: MDS
        ads: ADS
        mds_block_size: int
        ads_block_size: int

    l2aFDProductDOI: str
    product_id: str
    enable_product_flag: bool
    ground_cancellaton: GroundCancellationConfFD
    significance_level: float
    product_resolution: float
    numerical_determinant_limit: float
    upsampling_factor: int
    compression_options: CompressionConf


@dataclass
class L2aFHConf:
    """FH L2a configuration"""

    class verticalReflectivityOptions(Enum):
        """Vertical profile to be used"""

        VERTICAL_PROFILE = "default profile"

    class ModelInversionOptions(Enum):
        SINGLE = "single"
        DUAL = "dual"

    @dataclass
    class CompressionConf:
        """Compression configuration for all FH L2a products: MDS and ADS"""

        @dataclass
        class MDS:
            """Compression configuration for FH L2a MDS"""

            fh: MDSfloatCompressionType
            quality: MDSfloatCompressionType

        @dataclass
        class ADS:
            """Compression configuration for FH L2a ADS"""

            fnf: IntCompressionType

        mds: MDS
        ads: ADS
        mds_block_size: int
        ads_block_size: int

    l2aFHProductDOI: str
    product_id: str
    enable_product_flag: bool
    vertical_reflectivity_option: verticalReflectivityOptions
    vertical_reflectivity_default_profile: np.ndarray
    model_inversion: ModelInversionOptions
    spectral_decorrelation_compensation_flag: bool
    snr_decorrelation_compensation_flag: bool
    correct_terrain_slopes_flag: bool
    normalised_height_estimation_range: MinMaxType
    normalised_wavenumber_estimation_range: MinMaxNumType
    ground_to_volume_ratio_range: MinMaxNumType
    temporal_decorrelation_estimation_range: MinMaxNumType
    temporal_decorrelation_ground_to_volume_ratio: float
    residual_decorrelation: float
    product_resolution: float
    uncertainty_valid_values_limits: MinMaxType
    vertical_wavenumber_valid_values_limits: MinMaxType
    lower_height_limit: float
    upsampling_factor: int
    compression_options: CompressionConf


@dataclass
class L2aTFHConf:
    """TOMO FH L2a configuration"""

    @dataclass
    class CompressionConf:
        """Compression configuration for all TOMO FH L2a products: MDS and ADS"""

        @dataclass
        class MDS:
            """Compression configuration for TOMO FH L2a MDS"""

            tfh: MDSfloatCompressionType
            quality: MDSfloatCompressionType

        @dataclass
        class ADS:
            """Compression configuration for TOMO FH L2a ADS"""

            fnf: IntCompressionType

        mds: MDS
        ads: ADS
        mds_block_size: int
        ads_block_size: int

    l2aTOMOFHProductDOI: str
    product_id: str
    enable_product_flag: bool
    enable_super_resolution: bool
    product_resolution: float
    regularization_noise_factor: float
    power_threshold: float
    median_factor: int
    estimation_valid_values_limits: MinMaxType
    vertical_range: verticalRangeWithUnitsType
    compression_options: CompressionConf


@dataclass
class AuxProcessingParametersL2A:
    """BPS L2a Processing parameters"""

    general: GeneralConf
    agb: L2aAGBConf
    fd: L2aFDConf
    fh: L2aFHConf
    tfh: L2aTFHConf
