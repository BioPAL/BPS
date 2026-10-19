# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""In-memory representation of the BIOMASS L2B FD main annotation."""

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
        fd: CompressionFactorType
        probability_of_change: CompressionFactorMaxZErrorType
        cfm: CompressionFactorType
        heat_map: CompressionFactorType
        acquisition_id_image: CompressionFactorType

    mds: Mds
    mds_block_size: int


@dataclass
class ProcessingParametersAnnotation:
    processor_version: str
    product_generation_time: PreciseDateTime
    compression_options: CompressionOptionsL2B
    minimum_l2a_coverage: float


@dataclass
class MainAnnotationL2bFD:
    product: common_annotation_l2l3.ProductL2bL3Type
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage
    input_information: common_annotation_l2l3.InputInformationL2BL3ListType
    processing_parameters: ProcessingParametersAnnotation
