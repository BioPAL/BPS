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


@dataclass(kw_only=True)
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

    mds: CompressionOptionsL2BFd.Mds = field(
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
        fd: CompressionOptionsL2BFd.Mds.Fd = field(
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
            },
        )
        probability_of_change: CompressionOptionsL2BFd.Mds.ProbabilityOfChange = field(
            metadata={
                "name": "probabilityOfChange",
                "type": "Element",
                "namespace": "",
            },
        )
        cfm: CompressionOptionsL2BFd.Mds.Cfm = field(
            metadata={
                "name": "CFM",
                "type": "Element",
                "namespace": "",
            },
        )
        heat_map: CompressionOptionsL2BFd.Mds.HeatMap = field(
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
            },
        )
        acquisition_id_image: CompressionOptionsL2BFd.Mds.AcquisitionIdImage = field(
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
        class Fd:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the FD image MDS. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
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
        class Cfm:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the CFM image MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the Heat Map image MDS. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
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
                ZSTD algorithm compression factor for the acquisitionIdImage image MDS. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )


@dataclass(kw_only=True)
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

    l2b_fdproduct_doi: str = field(
        metadata={
            "name": "l2bFDProductDOI",
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
    compression_options: CompressionOptionsL2BFd = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryL2BFdprocessingParameters(AuxiliaryL2BFdprocessingParametersType):
    """
    BIOMASS configuration parameters for the L2b FD Processor.
    """

    class Meta:
        name = "auxiliaryL2bFDProcessingParameters"
