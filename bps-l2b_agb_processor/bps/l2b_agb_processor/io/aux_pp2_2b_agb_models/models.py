# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PP2 2B AGB models
---------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


@dataclass(kw_only=True)
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

    mds: CompressionOptionsL2BAb.Mds = field(
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
        },
    )
    mds_block_size: int = field(
        metadata={
            "name": "MDS_blockSize",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class Mds:
        agb: CompressionOptionsL2BAb.Mds.Agb = field(
            metadata={
                "name": "AGB",
                "type": "Element",
                "namespace": "",
            },
        )
        agbstandard_deviation: CompressionOptionsL2BAb.Mds.AgbstandardDeviation = field(
            metadata={
                "name": "AGBstandardDeviation",
                "type": "Element",
                "namespace": "",
            },
        )
        bps_fnf: CompressionOptionsL2BAb.Mds.BpsFnf = field(
            metadata={
                "name": "BPS_FNF",
                "type": "Element",
                "namespace": "",
            },
        )
        heat_map: CompressionOptionsL2BAb.Mds.HeatMap = field(
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
            },
        )
        acquisition_id_image: CompressionOptionsL2BAb.Mds.AcquisitionIdImage = field(
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
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

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )
            max_z_error: float = field(
                metadata={
                    "name": "MAX_Z_ERROR",
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
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

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )
            max_z_error: float = field(
                metadata={
                    "name": "MAX_Z_ERROR",
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
        class BpsFnf:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the BPS FNF image MDS. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
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

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )
            max_z_error: float = field(
                metadata={
                    "name": "MAX_Z_ERROR",
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
        class AcquisitionIdImage:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )


class ReferenceSelectionType(Enum):
    REF_ONLY = "refOnly"
    FIRST_ITERATION_ONLY = "firstIterationOnly"
    WEIGHTED_MEAN = "weightedMean"


@dataclass(kw_only=True)
class BackscatterLimitsType:
    class Meta:
        name = "backscatterLimitsType"

    hh: MinMaxType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    vh: MinMaxType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    vv: MinMaxType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    l2b_agbproduct_doi: str = field(
        metadata={
            "name": "l2bAGBProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    minimum_l2a_coverage: float = field(
        metadata={
            "name": "minimumL2aCoverage",
            "type": "Element",
            "namespace": "",
        },
    )
    forest_masking_flag: str = field(
        metadata={"name": "forestMaskingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    rejected_landcover_classes: IntArray = field(
        metadata={
            "name": "rejectedLandcoverClasses",
            "type": "Element",
            "namespace": "",
        },
    )
    backscatter_limits: BackscatterLimitsType = field(
        metadata={
            "name": "backscatterLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    angle_limits: MinMaxTypeWithUnit = field(
        metadata={
            "name": "angleLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    mean_agblimits: MinMaxTypeWithUnit = field(
        metadata={
            "name": "meanAGBLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    std_agblimits: MinMaxTypeWithUnit = field(
        metadata={
            "name": "stdAGBLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    relative_agblimits: MinMaxType = field(
        metadata={
            "name": "relativeAGBLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    reference_selection: ReferenceSelectionType = field(
        metadata={
            "name": "referenceSelection",
            "type": "Element",
            "namespace": "",
        },
    )
    indexing_l: AgbIndexingType = field(
        metadata={
            "name": "indexingL",
            "type": "Element",
            "namespace": "",
        },
    )
    indexing_a: AgbIndexingType = field(
        metadata={
            "name": "indexingA",
            "type": "Element",
            "namespace": "",
        },
    )
    indexing_n: AgbIndexingType = field(
        metadata={
            "name": "indexingN",
            "type": "Element",
            "namespace": "",
        },
    )
    use_constant_n: str = field(
        metadata={"name": "useConstantN", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    values_constant_n: FloatArray = field(
        metadata={
            "name": "valuesConstantN",
            "type": "Element",
            "namespace": "",
        },
    )
    regression_solver: str = field(
        metadata={
            "name": "regressionSolver",
            "type": "Element",
            "namespace": "",
        },
    )
    minimum_percentage_of_fillable_voids: float = field(
        metadata={
            "name": "minimumPercentageOfFillableVoids",
            "type": "Element",
            "namespace": "",
        },
    )
    compression_options: CompressionOptionsL2BAb = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryL2BAbprocessingParameters(AuxiliaryL2BAbprocessingParametersType):
    """
    BIOMASS configuration parameters for the L2b AGB Processor.
    """

    class Meta:
        name = "auxiliaryL2bABProcessingParameters"
