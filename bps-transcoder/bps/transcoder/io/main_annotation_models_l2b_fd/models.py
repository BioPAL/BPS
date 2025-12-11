# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models L2b FD
---------------------------------
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
    OperationalModeType,
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
        fd: Optional["CompressionOptionsL2A.Mds.Fd"] = field(
            default=None,
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        probability_ofchange: Optional["CompressionOptionsL2A.Mds.ProbabilityOfchange"] = field(
            default=None,
            metadata={
                "name": "probabilityOFChange",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        cfm: Optional["CompressionOptionsL2A.Mds.Cfm"] = field(
            default=None,
            metadata={
                "name": "CFM",
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
        class ProbabilityOfchange:
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
        acm: Optional["CompressionOptionsL2A.Ads.Acm"] = field(
            default=None,
            metadata={
                "name": "ACM",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        number_of_averages: Optional["CompressionOptionsL2A.Ads.NumberOfAverages"] = field(
            default=None,
            metadata={
                "name": "numberOfAverages",
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
        class Acm:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor valid for all the ACM ADS LUT layers. From 1 to 9.
            least_significant_digit
                For all the layers of ACM LUT ADS, define exactly how lossy the ZLIB compression algorithm is
                allowed to be, specifying the power of ten of the smallest decimal place in the data that is a
                reliable value. Zero means loss-less compression.
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
            least_significant_digit: Optional[int] = field(
                default=None,
                metadata={
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

        @dataclass
        class NumberOfAverages:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the numberOfAverages ADS. From 1 to 9.
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
        fd: Optional["CompressionOptionsL2B.Mds.Fd"] = field(
            default=None,
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        probability_of_change: Optional["CompressionOptionsL2B.Mds.ProbabilityOfChange"] = field(
            default=None,
            metadata={
                "name": "probabilityOfChange",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        cfm: Optional["CompressionOptionsL2B.Mds.Cfm"] = field(
            default=None,
            metadata={
                "name": "CFM",
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
                ZSTD algorithm compression factor for the FNF image MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the HeatMap image MDS. From 1 to 9.
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
    compression_options
        Configurable compression options for all the L2b MDS.
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
        General processing flags (not specific to the Forest Disturbance processing).
    emphasized_forest_height
        Ground cancellation: value of the height [m] which has been emphasized during ground cancellation.
    operational_mode
        Ground Cancellation method used between the followings. “single reference”: direct computation with a
        preliminary automatic reference image selection. “insar pair”: debug operational mode, perform computation
        with only the two images specified in the optional element imagesPairSelection. Note: in case of only two
        images available, the operationalMode is automatically set to “insar pair” (without the need of
        imagesPairSelection element).
    images_pair_selection
        If operationalMode is “insar pair” and if there are more than two images, than this element is present and
        ground cancellation has being performed using only the two images specified here.
    disable_ground_cancellation_flag
        True, if ground cancellation has been disabled. False, if the ground cancellation has been performed.
        (Optional, default is False).
    significance_level
        Level of significance used in the change detection algorithm.
    product_resolution
        Product resolution in [m].
    numerical_determinant_limit
        Numerical determinant limit
    upsampling_factor
        Upwnsampling factor.
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
    emphasized_forest_height: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "emphasizedForestHeight",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    operational_mode: Optional[OperationalModeType] = field(
        default=None,
        metadata={
            "name": "operationalMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    images_pair_selection: Optional[AcquisitionListType] = field(
        default=None,
        metadata={
            "name": "imagesPairSelection",
            "type": "Element",
            "namespace": "",
        },
    )
    disable_ground_cancellation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "disableGroundCancellationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        },
    )
    significance_level: Optional[float] = field(
        default=None,
        metadata={
            "name": "significanceLevel",
            "type": "Element",
            "namespace": "",
            "required": True,
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
    numerical_determinant_limit: Optional[float] = field(
        default=None,
        metadata={
            "name": "numericalDeterminantLimit",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    upsampling_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "upsamplingFactor",
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


@dataclass
class MainAnnotationType:
    """
    Parameters
    ----------
    product
        Product L2b and L3 DSR. This DSR contains the L2b/L3 product information.
    raster_image
        Raster image DSR. This DSR contains all the necessary information to exploit the L2b/L3 raster images.
    input_information
        Input Information L2b/L3 DSR. This DSR contains the necessary information to identify the L2a products in
        input to L2b or to L3 processor and also a list of all the L1c acquisitions used to generate those L2a
        products.
    processing_parameters
        Processing parameters DSR. This DSR contains the description of L2b/L3 processing parameters.
    """

    class Meta:
        name = "mainAnnotationType"

    product: Optional[ProductL2BL3Type] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raster_image: Optional[RasterImageType] = field(
        default=None,
        metadata={
            "name": "rasterImage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    input_information: Optional[InputInformationL2BL3ListType] = field(
        default=None,
        metadata={
            "name": "inputInformation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_parameters: Optional[ProcessingParametersL2BType] = field(
        default=None,
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class MainAnnotation(MainAnnotationType):
    """
    BIOMASS L2b FD product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
