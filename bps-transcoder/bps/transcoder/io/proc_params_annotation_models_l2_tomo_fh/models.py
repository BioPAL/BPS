# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD processing parameters annotation models L2 TOMO FH
------------------------------------------------------
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
    MissionPhaseIdtype,
    MissionType,
    OrbitPassType,
    PixelRepresentationType,
    PixelTypeType,
    PolarisationType,
    ProductType,
    ProjectionType,
    SensorModeType,
    SlantRangePolynomialType,
    StateType,
    SwathType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
)
from bps.transcoder.io.common_annotation_models_l2 import (
    AcquisitionListType,
    AcquisitionType,
    BackscatterLimitsType,
    BpsFnfType,
    CalAbcoverageTilesListType,
    CalAbcoverageType,
    CalAbfilteredCoverageType,
    CalibrationScreenType,
    GeneralConfigurationParametersType,
    GncoverageListType,
    GncoverageTilesListType,
    GncoverageType,
    GnfilteredCoverageType,
    InputInformationL2AType,
    InputInformationL2BL3ListType,
    IntegerListType,
    MinMaxNumType,
    NoDataValueChoiceType,
    PercentPixelsType,
    PixelRepresentationChoiceType,
    PixelTypeChoiceType,
    PolarisationListType,
    ProductL2AType,
    ProductL2BL3Type,
    RasterImageType,
    SelectedReferenceImageType,
    StaQualityParametersListType,
    StaQualityParametersType,
    StaQualityType,
    StringListType,
    SubsettingRuleType,
)


@dataclass
class CompressionOptionsL2A:
    """
    Parameters
    ----------
    mds
    ads
    mds_block_size
        MDS COG blocking algorithm size.
    ads_block_size
        NetCDF ADS chunking algorithm size.
    """

    class Meta:
        name = "compressionOptionsL2a"

    mds: Optional["CompressionOptionsL2A.Mds"] = field(
        default=None,
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ads: Optional["CompressionOptionsL2A.Ads"] = field(
        default=None,
        metadata={
            "name": "ADS",
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
    ads_block_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "ADS_blockSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class Mds:
        tfh: Optional["CompressionOptionsL2A.Mds.Tfh"] = field(
            default=None,
            metadata={
                "name": "TFH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        quality: Optional["CompressionOptionsL2A.Mds.Quality"] = field(
            default=None,
            metadata={
                "name": "Quality",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
        class Tfh:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            max_z_error
                For both TOMO FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
                For both TOMO FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
    class Ads:
        fnf: Optional["CompressionOptionsL2A.Ads.Fnf"] = field(
            default=None,
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
        class Fnf:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
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
class CompressionOptionsL2B:
    """
    Parameters
    ----------
    mds
    mds_block_size
        MDS COG blocking algorithm size.
    """

    class Meta:
        name = "compressionOptionsL2b"

    mds: Optional["CompressionOptionsL2B.Mds"] = field(
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
        tfh: Optional["CompressionOptionsL2B.Mds.Tfh"] = field(
            default=None,
            metadata={
                "name": "TFH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        quality: Optional["CompressionOptionsL2B.Mds.Quality"] = field(
            default=None,
            metadata={
                "name": "Quality",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        bps_fnf: Optional["CompressionOptionsL2B.Mds.BpsFnf"] = field(
            default=None,
            metadata={
                "name": "BPS_FNF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        heat_map: Optional["CompressionOptionsL2B.Mds.HeatMap"] = field(
            default=None,
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        acquisition_id_image: Optional["CompressionOptionsL2B.Mds.AcquisitionIdImage"] = field(
            default=None,
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
        class Tfh:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            max_z_error
                For both TOMO FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
                For both TOMO FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
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
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            max_z_error
                For heat map MDS, define exactly how lossy the LERC compression algorithm is allowed to be,
                specifying the absolute maximum error admitted. Zero means loss-less compression.
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
class VerticalReflectivityProfileType:
    """
    Parameters
    ----------
    val
    count
        Number of values
    """

    class Meta:
        name = "verticalReflectivityProfileType"

    val: list[float] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ProcessingParametersL2BType:
    """
    Parameters
    ----------
    processor_version
        Version of the processor used to generate the product.
    product_generation_time
        Product generation time [UTC].
    minimum_l2a_coverage
        Minimum coverage in percentage of the output tile to enable L2b processing
    forest_masking_flag
        True if forest/non-forest map masking has been performed during L2a products merging, false otherwise. Also,
        if True, the used forest/non-forest mask is specified in the maskFromForestDisturbanceFlag
    bps_fnf
        Type of Forest Mask in MDS among CFM or FNF: CFM if provided in input to L2b processor, global FNF
        otherwise.
    roll_off_factor_azimuth
        Feathering roll-off factor used.
    roll_off_factor_range
        Feathering roll-off factor used.
    compression_options
        Configurable compression options for all the L2a MDS COG and ADS NetCDF LUT variables.
    """

    class Meta:
        name = "processingParametersL2bType"

    processor_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_generation_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
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
    bps_fnf: Optional[BpsFnfType] = field(
        default=None,
        metadata={
            "name": "BPS_FNF",
            "type": "Element",
            "namespace": "",
            "required": True,
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
    compression_options: Optional[CompressionOptionsL2B] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class ProcessingParametersL2AType:
    """
    Parameters
    ----------
    processor_version
        Version of the processor used to generate the product.
    product_generation_time
        Product generation time [UTC].
    general_configuration_parameters
        General processing flags (not specific to the Forest Height processing).
    enable_super_resolution
        True to enable the TOMO FH super resolution algorithm.
    product_resolution
        Value in [m] to be used as the resolution on ground map and also to perform the covariance averaging in
        radar coordinates.
    regularization_noise_factor
        regularization Noise Factor
    power_threshold
        power threshold
    median_factor
        median Factor
    estimation_valid_values_limits
        Estimation valid values limits [m], values of estimations out of this limits are discarded and set to no
        data value
    compression_options
        Configurable compression options for all the L2a MDS and ADS NetCDF LUT variables.
    """

    class Meta:
        name = "processingParametersL2aType"

    processor_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_generation_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    general_configuration_parameters: Optional[GeneralConfigurationParametersType] = field(
        default=None,
        metadata={
            "name": "generalConfigurationParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_super_resolution: Optional[str] = field(
        default=None,
        metadata={
            "name": "enableSuperResolution",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    product_resolution: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "productResolution",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    regularization_noise_factor: Optional[float] = field(
        default=None,
        metadata={
            "name": "regularizationNoiseFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    power_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "powerThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    median_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "medianFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    estimation_valid_values_limits: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "estimationValidValuesLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    compression_options: Optional[CompressionOptionsL2A] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
