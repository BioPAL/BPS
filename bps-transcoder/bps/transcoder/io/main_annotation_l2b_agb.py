# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""In-memory representation of the BIOMASS L2B AGB main annotation."""

from dataclasses import dataclass
from enum import Enum

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.transcoder.io import common_annotation_l2l3


class ReferenceSelectionType(Enum):
    REF_ONLY = "refOnly"
    FIRST_ITERATION_ONLY = "firstIterationOnly"
    WEIGHTED_MEAN = "weightedMean"


class AgbIndexingType(Enum):
    NONE = "none"
    P = "p"
    J = "j"
    K = "k"
    PJ = "pj"
    PK = "pk"
    JK = "jk"
    PJK = "pjk"


class EstimatedParametersPolarisationType(Enum):
    HH_VH_VV = "HH VH VV"
    HH = "HH"
    VH = "VH"
    VV = "VV"


@dataclass
class PercentEstimate:
    mean: float
    std: float


@dataclass
class EstimatedParametersDate:
    n: PercentEstimate
    a: PercentEstimate
    l: PercentEstimate  # noqa: E741
    dates: str


@dataclass
class EstimatedParametersLcm:
    date: list[EstimatedParametersDate]
    classes: str


@dataclass
class EstimatedParametersPolarisation:
    lcm: list[EstimatedParametersLcm]
    polarisations: EstimatedParametersPolarisationType


@dataclass
class EstimatedParametersL2BAgb:
    rho: float
    polarisation: list[EstimatedParametersPolarisation]


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
        agb: CompressionFactorMaxZErrorType
        agbstandard_deviation: CompressionFactorMaxZErrorType
        bps_fnf: CompressionFactorType
        heat_map: CompressionFactorType
        acquisition_id_image: CompressionFactorType

    mds: Mds
    mds_block_size: int


@dataclass
class ProcessingParametersAnnotation:
    processor_version: str
    product_generation_time: PreciseDateTime
    forest_masking_flag: bool
    minimum_l2a_coverage: float
    rejected_landcover_classes: list[int]
    backscatter_limits: dict[common.PolarisationType, common.MinMaxType]
    angle_limits: common.MinMaxTypeWithUnit
    mean_agblimits: common.MinMaxTypeWithUnit
    std_agblimits: common.MinMaxTypeWithUnit
    relative_agblimits: common.MinMaxType
    reference_selection: ReferenceSelectionType
    indexing_l: AgbIndexingType
    indexing_a: AgbIndexingType
    indexing_n: AgbIndexingType
    use_constant_n: bool
    values_constant_n: list[float]
    regression_solver: str
    regression_matrix_subsampling_factor: int
    minimum_percentage_of_fillable_voids: float
    estimated_parameters: EstimatedParametersL2BAgb
    compression_options: CompressionOptionsL2B


@dataclass
class MainAnnotationL2bAGB:
    product: common_annotation_l2l3.ProductL2bL3Type
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage
    input_information: common_annotation_l2l3.InputInformationL2BL3ListType
    processing_parameters: ProcessingParametersAnnotation
