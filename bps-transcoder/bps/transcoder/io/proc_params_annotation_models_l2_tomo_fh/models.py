# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD processing parameters annotation models L2 TOMO FH
------------------------------------------------------
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
    MissionPhaseIdtype,
    MissionType,
    OrbitPassType,
    PixelRepresentationType,
    PixelTypeType,
    PolarisationCombinationMethodType,
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


@dataclass(kw_only=True)
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

    mds: CompressionOptionsL2A.Mds = field(
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
        },
    )
    ads: CompressionOptionsL2A.Ads = field(
        metadata={
            "name": "ADS",
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
    ads_block_size: int = field(
        metadata={
            "name": "ADS_blockSize",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class Mds:
        tfh: CompressionOptionsL2A.Mds.Tfh = field(
            metadata={
                "name": "TFH",
                "type": "Element",
                "namespace": "",
            },
        )
        quality: CompressionOptionsL2A.Mds.Quality = field(
            metadata={
                "name": "Quality",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
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
                For both TOMO FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
    class Ads:
        fnf: CompressionOptionsL2A.Ads.Fnf = field(
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
        class Fnf:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )


@dataclass(kw_only=True)
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

    mds: CompressionOptionsL2B.Mds = field(
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
        tfh: CompressionOptionsL2B.Mds.Tfh = field(
            metadata={
                "name": "TFH",
                "type": "Element",
                "namespace": "",
            },
        )
        quality: CompressionOptionsL2B.Mds.Quality = field(
            metadata={
                "name": "Quality",
                "type": "Element",
                "namespace": "",
            },
        )
        bps_fnf: CompressionOptionsL2B.Mds.BpsFnf = field(
            metadata={
                "name": "BPS_FNF",
                "type": "Element",
                "namespace": "",
            },
        )
        heat_map: CompressionOptionsL2B.Mds.HeatMap = field(
            metadata={
                "name": "HeatMap",
                "type": "Element",
                "namespace": "",
            },
        )
        acquisition_id_image: CompressionOptionsL2B.Mds.AcquisitionIdImage = field(
            metadata={
                "name": "acquisitionIdImage",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
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
                For both TOMO FH and quality images MDS, define exactly how lossy the LERC compression algorithm is
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
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
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
                ZLIB algorithm compression factor for the FNF ADS. From 1 to 9.
            max_z_error
                For heat map MDS, define exactly how lossy the LERC compression algorithm is allowed to be,
                specifying the absolute maximum error admitted. Zero means loss-less compression.
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    processor_version: str = field(
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
        },
    )
    product_generation_time: str = field(
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
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
    bps_fnf: BpsFnfType = field(
        metadata={
            "name": "BPS_FNF",
            "type": "Element",
            "namespace": "",
        },
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
    compression_options: CompressionOptionsL2B = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    processor_version: str = field(
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
        },
    )
    product_generation_time: str = field(
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    general_configuration_parameters: GeneralConfigurationParametersType = field(
        metadata={
            "name": "generalConfigurationParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_super_resolution: str = field(
        metadata={"name": "enableSuperResolution", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    product_resolution: FloatWithUnit = field(
        metadata={
            "name": "productResolution",
            "type": "Element",
            "namespace": "",
        },
    )
    regularization_noise_factor: float = field(
        metadata={
            "name": "regularizationNoiseFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    power_threshold: float = field(
        metadata={
            "name": "powerThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    median_factor: int = field(
        metadata={
            "name": "medianFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    estimation_valid_values_limits: MinMaxTypeWithUnit = field(
        metadata={
            "name": "estimationValidValuesLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    compression_options: CompressionOptionsL2A = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )
