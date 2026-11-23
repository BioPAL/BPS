# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""In-memory representation of the BIOMASS L2A main annotation."""

from dataclasses import dataclass

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.transcoder.io import common_annotation_l2l3


@dataclass
class CompressionFactorType:
    compression_factor: int


@dataclass
class CompressionFactorMaxZErrorType:
    compression_factor: int
    max_z_error: float


@dataclass
class CompressionFactorLeastSignificantDigitType:
    compression_factor: int
    least_significant_digit: int


@dataclass
class CompressionOptionsL2AFH:
    @dataclass
    class Mds:
        fh: CompressionFactorMaxZErrorType
        quality: CompressionFactorMaxZErrorType

    @dataclass
    class Ads:
        fnf: CompressionFactorType
        geometric_luts: CompressionFactorLeastSignificantDigitType

    mds: Mds
    ads: Ads
    mds_block_size: int
    ads_block_size: int


@dataclass
class BIOMASSL2aMainADSProcessingParametersFH:
    processor_version: str
    product_generation_time: PreciseDateTime
    general_configuration: common_annotation_l2l3.GeneralConfigurationParametersType
    compression_options: CompressionOptionsL2AFH

    vertical_reflectivity_option: str
    model_inversion: str
    spectral_decorrelation_compensation_flag: bool
    snr_decorrelation_compensation_flag: bool
    correct_terrain_slopes_flag: bool
    normalised_height_estimation_range: common.MinMaxType
    normalised_wavenumber_estimation_range: common_annotation_l2l3.MinMaxNumType
    ground_to_volume_ratio_range: common_annotation_l2l3.MinMaxNumType
    temporal_decorrelation_estimation_range: common_annotation_l2l3.MinMaxNumType
    temporal_decorrelation_ground_to_volume_ratio: float
    residual_decorrelation: float
    product_resolution: float
    uncertainty_valid_values_limits: common.MinMaxType
    vertical_wavenumber_valid_values_limits: common.MinMaxType
    lower_height_limit: float
    upsampling_factor: int
    vertical_reflectivity_default_profile: list[float]


@dataclass
class MainAnnotationL2aFH:
    """FH L2A main annotation data."""

    product: common_annotation_l2l3.BIOMASSL2aMainADSproduct
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage
    input_information: common_annotation_l2l3.BIOMASSL2aMainADSInputInformation
    processing_parameters: BIOMASSL2aMainADSProcessingParametersFH
    annotation_lut: list[common.LayerType]
