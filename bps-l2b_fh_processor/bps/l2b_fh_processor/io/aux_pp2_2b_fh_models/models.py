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

    mds: Optional["CompressionOptionsL2BFh.Mds"] = field(
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
        fh: Optional["CompressionOptionsL2BFh.Mds.Fh"] = field(
            default=None,
            metadata={
                "name": "FH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        quality: Optional["CompressionOptionsL2BFh.Mds.Quality"] = field(
            default=None,
            metadata={
                "name": "Quality",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        bps_fnf: Optional["CompressionOptionsL2BFh.Mds.BpsFnf"] = field(
            default=None,
            metadata={
                "name": "BPS_FNF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        heat_map: Optional["CompressionOptionsL2BFh.Mds.HeatMap"] = field(
            default=None,
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        acquisition_id_image: Optional["CompressionOptionsL2BFh.Mds.AcquisitionIdImage"] = field(
            default=None,
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
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
                ZSTD algorithm compression factor for the Heat Map image MDS. From 1 to 9.
            max_z_error
                For both FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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

    l2b_fhproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2bFHProductDOI",
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
    roll_off_factor_azimuth: Optional[float] = field(
        default=None,
        metadata={
            "name": "rollOffFactorAzimuth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    roll_off_factor_range: Optional[float] = field(
        default=None,
        metadata={
            "name": "rollOffFactorRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    compression_options: Optional[CompressionOptionsL2BFh] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryL2BFhprocessingParameters(AuxiliaryL2BFhprocessingParametersType):
    """
    BIOMASS configuration parameters for the L2b FH Processor.
    """

    class Meta:
        name = "auxiliaryL2bFHProcessingParameters"
