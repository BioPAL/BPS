# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models L2a FD
---------------------------------
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
        fd: CompressionOptionsL2A.Mds.Fd = field(
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
            },
        )
        probability_ofchange: CompressionOptionsL2A.Mds.ProbabilityOfchange = field(
            metadata={
                "name": "probabilityOFChange",
                "type": "Element",
                "namespace": "",
            },
        )
        cfm: CompressionOptionsL2A.Mds.Cfm = field(
            metadata={
                "name": "CFM",
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
    class Ads:
        fnf: CompressionOptionsL2A.Ads.Fnf = field(
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
            },
        )
        acm: CompressionOptionsL2A.Ads.Acm = field(
            metadata={
                "name": "ACM",
                "type": "Element",
                "namespace": "",
            },
        )
        number_of_averages: CompressionOptionsL2A.Ads.NumberOfAverages = field(
            metadata={
                "name": "numberOfAverages",
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

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )
            least_significant_digit: int = field(
                metadata={
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
        class NumberOfAverages:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the numberOfAverages ADS. From 1 to 9.
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
        fd: CompressionOptionsL2B.Mds.Fd = field(
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
            },
        )
        probability_of_change: CompressionOptionsL2B.Mds.ProbabilityOfChange = field(
            metadata={
                "name": "probabilityOfChange",
                "type": "Element",
                "namespace": "",
            },
        )
        cfm: CompressionOptionsL2B.Mds.Cfm = field(
            metadata={
                "name": "CFM",
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
                ZSTD algorithm compression factor for the FNF image MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the HeatMap image MDS. From 1 to 9.
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
    emphasized_forest_height: FloatWithUnit = field(
        metadata={
            "name": "emphasizedForestHeight",
            "type": "Element",
            "namespace": "",
        },
    )
    operational_mode: OperationalModeType = field(
        metadata={
            "name": "operationalMode",
            "type": "Element",
            "namespace": "",
        },
    )
    images_pair_selection: None | AcquisitionListType = field(
        default=None,
        metadata={
            "name": "imagesPairSelection",
            "type": "Element",
            "namespace": "",
        },
    )
    disable_ground_cancellation_flag: None | str = field(
        default=None,
        metadata={
            "name": "disableGroundCancellationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        },
    )
    significance_level: float = field(
        metadata={
            "name": "significanceLevel",
            "type": "Element",
            "namespace": "",
        },
    )
    product_resolution: FloatWithUnit = field(
        metadata={
            "name": "productResolution",
            "type": "Element",
            "namespace": "",
        },
    )
    numerical_determinant_limit: float = field(
        metadata={
            "name": "numericalDeterminantLimit",
            "type": "Element",
            "namespace": "",
        },
    )
    upsampling_factor: int = field(
        metadata={
            "name": "upsamplingFactor",
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


@dataclass(kw_only=True)
class MainAnnotationType:
    """
    Parameters
    ----------
    product
        Product L2a DSR. This DSR contains the L2a product information.
    raster_image
        Raster image DSR. This DSR contains all the necessary information to exploit the raster image products.
    input_information
        Input Information DSR. This DSR contains the necessary information to identify the input data set to the L2a
        processing, mainly the acquisitions configuration.
    processing_parameters
        Processing parameters DSR. This DSR contains the description of L2a processing parameters.
    annotation_lut
        Annotation LUT DSR. This DSR contains the list of Look-Up Tables (LUTs) complementing product main
        annotations.
    """

    class Meta:
        name = "mainAnnotationType"

    product: ProductL2AType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    raster_image: RasterImageType = field(
        metadata={
            "name": "rasterImage",
            "type": "Element",
            "namespace": "",
        },
    )
    input_information: InputInformationL2AType = field(
        metadata={
            "name": "inputInformation",
            "type": "Element",
            "namespace": "",
        },
    )
    processing_parameters: ProcessingParametersL2AType = field(
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    annotation_lut: LayerListType = field(
        metadata={
            "name": "annotationLUT",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class MainAnnotation(MainAnnotationType):
    """
    BIOMASS L2a product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
