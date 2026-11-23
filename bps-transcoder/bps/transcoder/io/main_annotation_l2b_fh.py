# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""In-memory representation of the BIOMASS L2B FH main annotation."""

from dataclasses import dataclass

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.io import common_annotation_l2l3


@dataclass
class CompressionFactorType:
    compression_factor: int


@dataclass
class CompressionFactorMaxZErrorType:
    compression_factor: int
    max_z_error: float


@dataclass
class CompressionOptionsL2B:
    @dataclass
    class Mds:
        fh: CompressionFactorMaxZErrorType
        quality: CompressionFactorMaxZErrorType
        bps_fnf: CompressionFactorType
        heat_map: CompressionFactorMaxZErrorType
        acquisition_id_image: CompressionFactorType

    mds: Mds
    mds_block_size: int


@dataclass
class ProcessingParametersAnnotation:
    processor_version: str
    product_generation_time: PreciseDateTime
    forest_masking_flag: bool
    bps_fnf: common_annotation_l2l3.BpsFnfType
    compression_options: CompressionOptionsL2B
    minimum_l2a_coverage: float
    roll_off_factor_azimuth: float
    roll_off_factor_range: float


@dataclass
class MainAnnotationL2bFH:
    product: common_annotation_l2l3.ProductL2bL3Type
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage
    input_information: common_annotation_l2l3.InputInformationL2BL3ListType
    processing_parameters: ProcessingParametersAnnotation
