# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models L2b AGB
----------------------------------
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
    AgbIndexingType,
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
        gn: Optional["CompressionOptionsL2A.Mds.Gn"] = field(
            default=None,
            metadata={
                "name": "GN",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

        @dataclass
        class Gn:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the ADS. From 1 to 9.
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
        local_incidence_angle: Optional["CompressionOptionsL2A.Ads.LocalIncidenceAngle"] = field(
            default=None,
            metadata={
                "name": "localIncidenceAngle",
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
                ZLIB algorithm compression factor. From 1 to 9.
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
        class LocalIncidenceAngle:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor. From 1 to 9.
            least_significant_digit
                define exactly how lossy the ZLIB compression algorithm is allowed to be, specifying the power of
                ten of the smallest decimal place in the data that is a reliable value. Zero means loss-less
                compression.
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
        agb: Optional["CompressionOptionsL2B.Mds.Agb"] = field(
            default=None,
            metadata={
                "name": "AGB",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        agbstandard_deviation: Optional["CompressionOptionsL2B.Mds.AgbstandardDeviation"] = field(
            default=None,
            metadata={
                "name": "AGBstandardDeviation",
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
        class Agb:
            """
            Parameters
            ----------
            compression_factor
                ZSTD algorithm compression factor for the images MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the images MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the images MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the image MDS. From 1 to 9.
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


class EstimatedParametersPolarisationTypePolarisations(Enum):
    HH_VH_VV = "HH VH VV"
    HH = "HH"
    VH = "VH"
    VV = "VV"


class ReferenceSelectionType(Enum):
    REF_ONLY = "refOnly"
    FIRST_ITERATION_ONLY = "firstIterationOnly"
    WEIGHTED_MEAN = "weightedMean"


@dataclass
class EstimatedParametersPolarisationType:
    class Meta:
        name = "estimatedParametersPolarisationType"

    lcm: list["EstimatedParametersPolarisationType.Lcm"] = field(
        default_factory=list,
        metadata={
            "name": "LCM",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    polarisations: Optional[EstimatedParametersPolarisationTypePolarisations] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )

    @dataclass
    class Lcm:
        date: list["EstimatedParametersPolarisationType.Lcm.Date"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 1,
            },
        )
        classes: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            },
        )

        @dataclass
        class Date:
            n: Optional["EstimatedParametersPolarisationType.Lcm.Date.N"] = field(
                default=None,
                metadata={
                    "name": "N",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            a: Optional["EstimatedParametersPolarisationType.Lcm.Date.A"] = field(
                default=None,
                metadata={
                    "name": "A",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            l: Optional["EstimatedParametersPolarisationType.Lcm.Date.L"] = field(
                default=None,
                metadata={
                    "name": "L",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            dates: Optional[str] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                },
            )

            @dataclass
            class N:
                mean: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )
                std: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )

            @dataclass
            class A:
                mean: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )
                std: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )

            @dataclass
            class L:
                mean: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )
                std: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )


@dataclass
class EstimatedParametersL2BAgb:
    """
    Parameters
    ----------
    rho
        AGB estimation Rho value: logarithmic bias correction as the ratio of the average reference AGB to the
        average estimated AGB
    polarisation
        Parameters estimated during AGB processing
    """

    class Meta:
        name = "estimatedParametersL2bAGB"

    rho: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: list[EstimatedParametersPolarisationType] = field(
        default_factory=list,
        metadata={"name": "Polarisation", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 3},
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
    forest_masking_flag
        True if forest/non-forest map masking has been performed during L2a products merging, false otherwise. Also,
        if True, the used forest/non-forest mask is specified in the maskFromForestDisturbanceFlag.
    minimum_l2a_coverage
        Minimum coverage in percentage of the output tile to enable L2b processing
    rejected_landcover_classes
        Set of landcover class indices to ignore
    backscatter_limits
        Lower and upper limits on backscatter in linear units. Default values are 0.0001,100 for all polarisations
    angle_limits
        Lower and upper limits on the local incidence angle in radians. Default values are 0, π/2
    mean_agblimits
        Lower and upper limits on the AGB mean in t/ha. Default values are 10^-3,10^3
    std_agblimits
        Lower and upper limits on the AGB std in t/ha. Default values are 10^-3,10^3
    relative_agblimits
        Lower and upper limits on AGB standard deviation relative AGB mean (coefficient of variability), in linear
        units. Default values are 0, 0.3
    reference_selection
        Selection of reference data (second iteration only). String.
    indexing_l
        L parameter variability with polarization ’p’, date ’j’, forest class ’k’. Default is ‘pj’
    indexing_a
        A parameter variability with polarization ’p’, date ’j’, forest class ’k’. Default is ‘pk’
    indexing_n
        N parameter variability with polarization ’p’, date ’j’, forest class ’k’. Default is ‘p’
    use_constant_n
        If True, indexingN=p is forced. Default is false
    values_constant_n
        Values to use if useConstantN=True. Default values is 1.0 for all polarizations
    regression_solver
        Computational complexity used for the regression:“double”,“float”
    regression_matrix_subsampling_factor
        Subsampling factor used during AGB regression, computed live, basing on available RAM remaining (starting
        from job order Amount_of_RAM)
    minimum_percentage_of_fillable_voids
        The minimum [%] of invalid pixels that triggers a new iteration.Default value is 5.0
    estimated_parameters
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
    minimum_l2a_coverage: Optional[float] = field(
        default=None,
        metadata={
            "name": "minimumL2aCoverage",
            "type": "Element",
            "namespace": "",
            "required": True,
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
    regression_matrix_subsampling_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "regressionMatrixSubsamplingFactor",
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
    estimated_parameters: Optional[EstimatedParametersL2BAgb] = field(
        default=None,
        metadata={
            "name": "estimatedParameters",
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
        General processing flags (not specific to the ground cancellation).
    emphasized_forest_height
        Single value of the height [m] for all the pixels and polarizations, which has been emphasized during ground
        cancellation [AD8].
    operational_mode
        Ground Cancellation method used between the followings. “multi reference”: multiple data computation using
        each image as reference, followed by data averaging. “insar pair”: debug operational mode, perform
        computation with only the two images specified in the optional element imagesPairSelection. Note: in case of
        only two images available, the operationalMode is automatically set to “insar pair” (without the need of
        imagesPairSelection element).
    compute_gnpower_flag
        True if the returned GN L2a image is an absolute squared value (power), False if it is the complex value.
        The flag is ineffective in case of multi reference operational mode (in this case the result is real only)
    radiometric_calibration_flag
        True if the incidence angle radiometric calibration has been applied, False otherwise.
    images_pair_selection
        If operationalMode is “insar pair” and if there are more than two images, than this element is present and
        ground cancellation has being performed using only the two images specified here.
    disable_ground_cancellation_flag
        True, if ground cancellation has been disabled. False, if the ground cancellation has been performed.
        (Optional, default is False).
    product_resolution
        Multi-look windows size in [m] (product resolution) used during ground cancellation.
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
    compute_gnpower_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "computeGNPowerFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    radiometric_calibration_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "radiometricCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
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
    product_resolution: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "productResolution",
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
    BIOMASS L2b AGB product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
