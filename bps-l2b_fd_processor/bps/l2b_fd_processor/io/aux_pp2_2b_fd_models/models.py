# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PP2 2B FD models
--------------------
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


@dataclass
class CompressionOptionsL2BFd:
    """
    Parameters
    ----------
    mds
    mds_block_size
        Blocking size of all MDS.
    """

    class Meta:
        name = "compressionOptionsL2bFD"

    mds: Optional["CompressionOptionsL2BFd.Mds"] = field(
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
        fd: Optional["CompressionOptionsL2BFd.Mds.Fd"] = field(
            default=None,
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        probability_of_change: Optional["CompressionOptionsL2BFd.Mds.ProbabilityOfChange"] = field(
            default=None,
            metadata={
                "name": "probabilityOfChange",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        cfm: Optional["CompressionOptionsL2BFd.Mds.Cfm"] = field(
            default=None,
            metadata={
                "name": "CFM",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        heat_map: Optional["CompressionOptionsL2BFd.Mds.HeatMap"] = field(
            default=None,
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        acquisition_id_image: Optional["CompressionOptionsL2BFd.Mds.AcquisitionIdImage"] = field(
            default=None,
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
        class Fd:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the FD image MDS. From 1 to 9.
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
        class ProbabilityOfChange:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the probability of change image MDS. From 1 to 9.
            max_z_error
                For the probability of change image MDS, define exactly how lossy the LERC compression algorithm is
                allowed to be, specifying the absolute maximum error admitted. Zero means loss-less compression.
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
        class Cfm:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the CFM image MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the Heat Map image MDS. From 1 to 9.
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
        class AcquisitionIdImage:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the acquisitionIdImage image MDS. From 1 to 9.
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
class AuxiliaryL2BFdprocessingParametersType:
    """
    Parameters
    ----------
    l2b_fdproduct_doi
        Digital Object Identifier (DOI) to be written in output products.
    minimum_l2a_coverage
        Minimum coverage in percentage of the output tile to enable L2b processing.
    compression_options
        Configurable compression options for all the FD L2B MDS variables.
    """

    class Meta:
        name = "auxiliaryL2bFDProcessingParametersType"

    l2b_fdproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2bFDProductDOI",
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
    compression_options: Optional[CompressionOptionsL2BFd] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryL2BFdprocessingParameters(AuxiliaryL2BFdprocessingParametersType):
    """
    BIOMASS configuration parameters for the L2b FD Processor.
    """

    class Meta:
        name = "auxiliaryL2bFDProcessingParameters"
