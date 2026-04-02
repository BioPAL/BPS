# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD processing parameters annotation models L2 AGB
--------------------------------------------------
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
        gn: CompressionOptionsL2A.Mds.Gn = field(
            metadata={
                "name": "GN",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
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
        local_incidence_angle: CompressionOptionsL2A.Ads.LocalIncidenceAngle = field(
            metadata={
                "name": "localIncidenceAngle",
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
                ZLIB algorithm compression factor. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
                },
            )

        @dataclass(kw_only=True)
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
        agb: CompressionOptionsL2B.Mds.Agb = field(
            metadata={
                "name": "AGB",
                "type": "Element",
                "namespace": "",
            },
        )
        agbstandard_deviation: CompressionOptionsL2B.Mds.AgbstandardDeviation = field(
            metadata={
                "name": "AGBstandardDeviation",
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
                ZSTD algorithm compression factor for the images MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the images MDS. From 1 to 9.
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
                ZSTD algorithm compression factor for the image MDS. From 1 to 9.
            """

            compression_factor: int = field(
                metadata={
                    "name": "compressionFactor",
                    "type": "Element",
                    "namespace": "",
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


@dataclass(kw_only=True)
class EstimatedParametersPolarisationType:
    class Meta:
        name = "estimatedParametersPolarisationType"

    lcm: list[EstimatedParametersPolarisationType.Lcm] = field(
        default_factory=list,
        metadata={
            "name": "LCM",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    polarisations: EstimatedParametersPolarisationTypePolarisations = field(
        metadata={
            "type": "Attribute",
        },
    )

    @dataclass(kw_only=True)
    class Lcm:
        date: list[EstimatedParametersPolarisationType.Lcm.Date] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 1,
            },
        )
        classes: str = field(
            metadata={
                "type": "Attribute",
            },
        )

        @dataclass(kw_only=True)
        class Date:
            n: EstimatedParametersPolarisationType.Lcm.Date.N = field(
                metadata={
                    "name": "N",
                    "type": "Element",
                    "namespace": "",
                },
            )
            a: EstimatedParametersPolarisationType.Lcm.Date.A = field(
                metadata={
                    "name": "A",
                    "type": "Element",
                    "namespace": "",
                },
            )
            l: EstimatedParametersPolarisationType.Lcm.Date.L = field(
                metadata={
                    "name": "L",
                    "type": "Element",
                    "namespace": "",
                },
            )
            dates: str = field(
                metadata={
                    "type": "Attribute",
                },
            )

            @dataclass(kw_only=True)
            class N:
                mean: float = field(
                    metadata={
                        "type": "Element",
                        "namespace": "",
                    },
                )
                std: float = field(
                    metadata={
                        "type": "Element",
                        "namespace": "",
                    },
                )

            @dataclass(kw_only=True)
            class A:
                mean: float = field(
                    metadata={
                        "type": "Element",
                        "namespace": "",
                    },
                )
                std: float = field(
                    metadata={
                        "type": "Element",
                        "namespace": "",
                    },
                )

            @dataclass(kw_only=True)
            class L:
                mean: float = field(
                    metadata={
                        "type": "Element",
                        "namespace": "",
                    },
                )
                std: float = field(
                    metadata={
                        "type": "Element",
                        "namespace": "",
                    },
                )


@dataclass(kw_only=True)
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

    rho: float = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: list[EstimatedParametersPolarisationType] = field(
        default_factory=list,
        metadata={"name": "Polarisation", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 3},
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
    forest_masking_flag: str = field(
        metadata={"name": "forestMaskingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    minimum_l2a_coverage: float = field(
        metadata={
            "name": "minimumL2aCoverage",
            "type": "Element",
            "namespace": "",
        },
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
    regression_matrix_subsampling_factor: int = field(
        metadata={
            "name": "regressionMatrixSubsamplingFactor",
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
    estimated_parameters: EstimatedParametersL2BAgb = field(
        metadata={
            "name": "estimatedParameters",
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
    compute_gnpower_flag: str = field(
        metadata={"name": "computeGNPowerFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    radiometric_calibration_flag: str = field(
        metadata={
            "name": "radiometricCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
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
    product_resolution: FloatWithUnit = field(
        metadata={
            "name": "productResolution",
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
