# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PP2 2B FH models
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
class CompressionOptionsL2BFh:
    """
    Parameters
    ----------
    mds
    mds_block_size
        Blocking size of all MDS.
    """

    class Meta:
        name = "compressionOptionsL2bFH"

    mds: CompressionOptionsL2BFh.Mds = field(
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
        fh: CompressionOptionsL2BFh.Mds.Fh = field(
            metadata={
                "name": "FH",
                "type": "Element",
                "namespace": "",
            },
        )
        quality: CompressionOptionsL2BFh.Mds.Quality = field(
            metadata={
                "name": "Quality",
                "type": "Element",
                "namespace": "",
            },
        )
        bps_fnf: CompressionOptionsL2BFh.Mds.BpsFnf = field(
            metadata={
                "name": "BPS_FNF",
                "type": "Element",
                "namespace": "",
            },
        )
        heat_map: CompressionOptionsL2BFh.Mds.HeatMap = field(
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
            },
        )
        acquisition_id_image: CompressionOptionsL2BFh.Mds.AcquisitionIdImage = field(
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
        class Fh:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            max_z_error
                For both FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
        class Quality:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            max_z_error
                For both FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
                ZSTD algorithm compression factor for the Heat Map image MDS. From 1 to 9.
            max_z_error
                For both FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
class AuxiliaryL2BFhprocessingParametersType:
    """
    Parameters
    ----------
    l2b_fhproduct_doi
        Digital Object Identifier (DOI) to be written in output products.
    minimum_l2a_coverage
        Minimum configurable percentage of the output tile to be processed, which the inputs provided to L2b
        processor shall coverage.
    forest_masking_flag
        True if forest/non-forest map masking should be performed during L2a products merging, false otherwise.
    roll_off_factor_azimuth
        Feathering roll-off factor. Values: 0.0 - 1.0. Default: TBD.
    roll_off_factor_range
        Feathering roll-off factor. Values: 0.0 - 1.0. Default: TBD.
    compression_options
        Configurable compression options for all the FH L2B MDS variables.
    """

    class Meta:
        name = "auxiliaryL2bFHProcessingParametersType"

    l2b_fhproduct_doi: str = field(
        metadata={
            "name": "l2bFHProductDOI",
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
    roll_off_factor_azimuth: float = field(
        metadata={
            "name": "rollOffFactorAzimuth",
            "type": "Element",
            "namespace": "",
        },
    )
    roll_off_factor_range: float = field(
        metadata={
            "name": "rollOffFactorRange",
            "type": "Element",
            "namespace": "",
        },
    )
    compression_options: CompressionOptionsL2BFh = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryL2BFhprocessingParameters(AuxiliaryL2BFhprocessingParametersType):
    """
    BIOMASS configuration parameters for the L2b FH Processor.
    """

    class Meta:
        name = "auxiliaryL2bFHProcessingParameters"
