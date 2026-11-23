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
class CompressionOptionsL2AFD:
    @dataclass
    class Mds:
        fd: CompressionFactorType
        probability_ofchange: CompressionFactorMaxZErrorType
        cfm: CompressionFactorType

    @dataclass
    class Ads:
        fnf: CompressionFactorType
        acm: CompressionFactorLeastSignificantDigitType
        number_of_averages: CompressionFactorType
        geometric_luts: CompressionFactorLeastSignificantDigitType

    mds: Mds
    ads: Ads
    mds_block_size: int
    ads_block_size: int


@dataclass
class BIOMASSL2aMainADSProcessingParametersFD:
    processor_version: str
    product_generation_time: PreciseDateTime
    general_configuration: common_annotation_l2l3.GeneralConfigurationParametersType
    compression_options: CompressionOptionsL2AFD

    emphasized_forest_height: float
    operational_mode: str
    significance_level: float
    product_resolution: float
    numerical_determinant_limit: float
    upsampling_factor: int
    images_pair_selection: common_annotation_l2l3.AcquisitionListType | None = None
    disable_ground_cancellation_flag: bool | None = False


@dataclass
class MainAnnotationL2aFD:
    """FD L2A main annotation data."""

    product: common_annotation_l2l3.BIOMASSL2aMainADSproduct
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage
    input_information: common_annotation_l2l3.BIOMASSL2aMainADSInputInformation
    processing_parameters: BIOMASSL2aMainADSProcessingParametersFD
    annotation_lut: list[common.LayerType]
