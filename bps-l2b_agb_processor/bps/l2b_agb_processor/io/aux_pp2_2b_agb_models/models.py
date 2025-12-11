# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PP2 2B AGB models
---------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bps.common.io.common_types import (
    AzimuthPolynomialType,
    ChannelImbalanceList,
    ChannelType,
    Complex,
    ComplexArray,
    CrossTalkList,
    DatumType,
    DoubleArray,
    DoubleArrayWithUnits,
    DoubleWithUnit,
    FloatArray,
    FloatArrayWithUnits,
    FloatWithChannel,
    FloatWithPolarisation,
    FloatWithUnit,
    GeodeticReferenceFrameType,
    GroupType,
    HeightModelBaseType,
    HeightModelType,
    IntArray,
    InterferometricPairListType,
    InterferometricPairType,
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    PolarisationType,
    SlantRangePolynomialType,
    StateType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
)


class AgbIndexingType(Enum):
    """
    Enumeration of AGB algorithm indexings.
    """

    NONE = "none"
    P = "p"
    J = "j"
    K = "k"
    PJ = "pj"
    PK = "pk"
    JK = "jk"
    PJK = "pjk"


@dataclass
class CompressionOptionsL2BAb:
    """
    Parameters
    ----------
    mds
    mds_block_size
        Blocking size of all MDS.
    """

    class Meta:
        name = "compressionOptionsL2bAB"

    mds: Optional["CompressionOptionsL2BAb.Mds"] = field(
        default=None,
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    mds_block_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "MDS_blockSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class Mds:
        agb: Optional["CompressionOptionsL2BAb.Mds.Agb"] = field(
            default=None,
            metadata={
                "name": "AGB",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        agbstandard_deviation: Optional["CompressionOptionsL2BAb.Mds.AgbstandardDeviation"] = field(
            default=None,
            metadata={
                "name": "AGBstandardDeviation",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        bps_fnf: Optional["CompressionOptionsL2BAb.Mds.BpsFnf"] = field(
            default=None,
            metadata={
                "name": "BPS_FNF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        heat_map: Optional["CompressionOptionsL2BAb.Mds.HeatMap"] = field(
            default=None,
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        acquisition_id_image: Optional["CompressionOptionsL2BAb.Mds.AcquisitionIdImage"] = field(
            default=None,
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
        class Agb:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor. From 1 to 9.
            max_z_error
                define exactly how lossy the LERC compression algorithm is allowed to be, specifying the absolute
                maximum error admitted. Zero means loss-less compression.
            """

            compression_factor: Optional[int] = field(
                default=None,
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            max_z_error: Optional[float] = field(
                default=None,
                metadata={
                    "name": "MAX_Z_ERROR",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

        @dataclass
        class AgbstandardDeviation:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor. From 1 to 9.
            max_z_error
                define exactly how lossy the LERC compression algorithm is allowed to be, specifying the absolute
                maximum error admitted. Zero means loss-less compression.
            """

            compression_factor: Optional[int] = field(
                default=None,
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            max_z_error: Optional[float] = field(
                default=None,
                metadata={
                    "name": "MAX_Z_ERROR",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

        @dataclass
        class BpsFnf:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the BPS FNF image MDS. From 1 to 9.
            """

            compression_factor: Optional[int] = field(
                default=None,
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

        @dataclass
        class HeatMap:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the Heat Map. From 1 to 9.
            max_z_error
                define exactly how lossy the LERC compression algorithm is allowed to be, specifying the absolute
                maximum error admitted. Zero means loss-less compression.
            """

            compression_factor: Optional[int] = field(
                default=None,
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            max_z_error: Optional[float] = field(
                default=None,
                metadata={
                    "name": "MAX_Z_ERROR",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

        @dataclass
        class AcquisitionIdImage:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor. From 1 to 9.
            """

            compression_factor: Optional[int] = field(
                default=None,
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )


class ReferenceSelectionType(Enum):
    REF_ONLY = "refOnly"
    FIRST_ITERATION_ONLY = "firstIterationOnly"
    WEIGHTED_MEAN = "weightedMean"


@dataclass
class BackscatterLimitsType:
    class Meta:
        name = "backscatterLimitsType"

    hh: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vh: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vv: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryL2BAbprocessingParametersType:
    """
    Parameters
    ----------
    l2b_agbproduct_doi
        Digital Object Identifier (DOI) to be written in output products.
    minimum_l2a_coverage
        Minimum configurable percentage of the output tile to be processed, which the inputs provided to L2b
        processor shall coverage.
    forest_masking_flag
        True if forest/non-forest map masking should be performed during L2a products merging, false otherwise.
    rejected_landcover_classes
        rejectedLandcoverClasses Set of landcover class indices to ignore
    backscatter_limits
        Lower and upper limits on backscatter in linear units.
    angle_limits
        Lower and upper limits on the local incidence angle in radians.
    mean_agblimits
        Lower and upper limits on the AGB mean in t/ha.
    std_agblimits
        Lower and upper limits on the AGB std in t/ha.
    relative_agblimits
        Lower and upper limits on AGB standard deviation relative AGB mean (coefficient of variability), in linear
        units.
    reference_selection
        Selection of reference data (second iteration only). String with following possible values: refOnly,
        firstIterationOnly,weightedMean
    indexing_l
        L parameter variability with polarization ’p’, date ’j’, forest class ’k’
    indexing_a
        a parameter variability with polarization ’p’, date ’j’, forest class ’k’
    indexing_n
        n parameter variability with polarization ’p’, date ’j’, forest class ’k’
    use_constant_n
        If True, indexingN=p is forced
    values_constant_n
        Values to use if useConstantN=True
    regression_solver
        Select the computational complexity for the regression:'float', 'double'
    minimum_percentage_of_fillable_voids
        The minimum [%] of invalid pixels that triggers a new iteration.Default value is 5.0
    compression_options
        Configurable compression options for all the ABG L2B MDS variables.
    """

    class Meta:
        name = "auxiliaryL2bABProcessingParametersType"

    l2b_agbproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2bAGBProductDOI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    minimum_l2a_coverage: Optional[float] = field(
        default=None,
        metadata={
            "name": "minimumL2aCoverage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    forest_masking_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "forestMaskingFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    rejected_landcover_classes: Optional[IntArray] = field(
        default=None,
        metadata={
            "name": "rejectedLandcoverClasses",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    backscatter_limits: Optional[BackscatterLimitsType] = field(
        default=None,
        metadata={
            "name": "backscatterLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    angle_limits: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "angleLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    mean_agblimits: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "meanAGBLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    std_agblimits: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "stdAGBLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    relative_agblimits: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "name": "relativeAGBLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    reference_selection: Optional[ReferenceSelectionType] = field(
        default=None,
        metadata={
            "name": "referenceSelection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    indexing_l: Optional[AgbIndexingType] = field(
        default=None,
        metadata={
            "name": "indexingL",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    indexing_a: Optional[AgbIndexingType] = field(
        default=None,
        metadata={
            "name": "indexingA",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    indexing_n: Optional[AgbIndexingType] = field(
        default=None,
        metadata={
            "name": "indexingN",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    use_constant_n: Optional[str] = field(
        default=None,
        metadata={
            "name": "useConstantN",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    values_constant_n: Optional[FloatArray] = field(
        default=None,
        metadata={
            "name": "valuesConstantN",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    regression_solver: Optional[str] = field(
        default=None,
        metadata={
            "name": "regressionSolver",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    minimum_percentage_of_fillable_voids: Optional[float] = field(
        default=None,
        metadata={
            "name": "minimumPercentageOfFillableVoids",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    compression_options: Optional[CompressionOptionsL2BAb] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryL2BAbprocessingParameters(AuxiliaryL2BAbprocessingParametersType):
    """
    BIOMASS configuration parameters for the L2b AGB Processor.
    """

    class Meta:
        name = "auxiliaryL2bABProcessingParameters"
