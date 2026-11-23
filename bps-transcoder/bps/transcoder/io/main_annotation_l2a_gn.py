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
class CompressionOptionsL2AGN:
    @dataclass
    class Mds:
        gn: CompressionFactorMaxZErrorType

    @dataclass
    class Ads:
        fnf: CompressionFactorType
        geometric_luts: CompressionFactorLeastSignificantDigitType

    mds: Mds
    ads: Ads
    mds_block_size: int
    ads_block_size: int


@dataclass
class BIOMASSL2aMainADSProcessingParametersGN:
    processor_version: str
    product_generation_time: PreciseDateTime
    general_configuration: common_annotation_l2l3.GeneralConfigurationParametersType
    compression_options: CompressionOptionsL2AGN

    emphasized_forest_height: float
    operational_mode: str
    radiometric_calibration_flag: bool
    product_resolution: float
    upsampling_factor: int
    images_pair_selection: common_annotation_l2l3.AcquisitionListType | None = None
    disable_ground_cancellation_flag: bool | None = False


@dataclass
class MainAnnotationL2aGN:
    """GN L2A main annotation data."""

    product: common_annotation_l2l3.BIOMASSL2aMainADSproduct
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage
    input_information: common_annotation_l2l3.BIOMASSL2aMainADSInputInformation
    processing_parameters: BIOMASSL2aMainADSProcessingParametersGN
    annotation_lut: list[common.LayerType]
