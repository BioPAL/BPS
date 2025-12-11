# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models L2a FH
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
        fh: Optional["CompressionOptionsL2A.Mds.Fh"] = field(
            default=None,
            metadata={
                "name": "FH",
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
        fh: Optional["CompressionOptionsL2B.Mds.Fh"] = field(
            default=None,
            metadata={
                "name": "FH",
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


class ModelInversionType(Enum):
    """
    Forest height inversion option.
    """

    SINGLE = "single"
    DUAL = "dual"


class VerticalProfileOptionType(Enum):
    """
    Vertical profile choice.
    """

    DEFAULT_PROFILE = "default profile"


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
    vertical_reflectivity_option
        Specify if a default profile (verticalReflectivityDefaultProfile) or a tomographic database (TBD) profile
        has been used.
    vertical_reflectivity_default_profile
        Vertical reflectivity profile function used if selected from verticalReflectivityOption, stored as a vector
        of values.
    model_inversion
        Model inversion algorithm used between single or dual baseline.
    spectral_decorrelation_compensation_flag
        True if spectral decorrelation compensation has been applied, False otherwise.
    snrdecorrelation_compensation
        True if SNR decorrelation compensation has been applied, False otherwise.
    correct_terrain_slopes_flag
        True if terrain slope correction has been applied, False otherwise.
    normalised_height_estimation_range
        Validity range of heights used for the canopy height estimation process, normalized from 0 to 1.
    normalised_wavenumber_estimation_range
        Validity range of wavenumbers used for the canopy height estimation process, normalized from 0 to 2π.
    ground_to_volume_ratio_range
        Validity ground to volume ratio range used for the canopy height estimation process.
    temporal_decorrelation_estimation_range
        Validity temporal decorrelation range used for the canopy height estimation process.
    temporal_decorrelation_ground_to_volume_ratio
        Ratio of temporal decorrelation between ground and volume (0.0 means no temporal decorrelation for ground,
        while 1.0 means ground and volume are equally impacted by temporal decorrelation)
    residual_decorrelation
        Residual decorrelation value used in error model computation.
    product_resolution
        Value in [m] used as the resolution on ground range map and also to perform the covariance averaging in
        radar coordinates.
    uncertainty_validvalues_limits
        Estimation valid values limits applied, in [m].
    vertical_wavenumber_validvalues_limits
        Vertical wavenumber valid values limits applied.
    lower_height_limit
        FH estimates lower this limit [m] were discarded and set to no data value.
    upsampling_factor
        Upsampling factor used for coherence.
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
    vertical_reflectivity_option: Optional[VerticalProfileOptionType] = field(
        default=None,
        metadata={
            "name": "verticalReflectivityOption",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vertical_reflectivity_default_profile: Optional[VerticalReflectivityProfileType] = field(
        default=None,
        metadata={
            "name": "verticalReflectivityDefaultProfile",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    model_inversion: Optional[ModelInversionType] = field(
        default=None,
        metadata={
            "name": "modelInversion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    spectral_decorrelation_compensation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "spectralDecorrelationCompensationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    snrdecorrelation_compensation: Optional[str] = field(
        default=None,
        metadata={
            "name": "SNRDecorrelationCompensation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    correct_terrain_slopes_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "correctTerrainSlopesFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    normalised_height_estimation_range: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "name": "normalisedHeightEstimationRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    normalised_wavenumber_estimation_range: Optional[MinMaxNumType] = field(
        default=None,
        metadata={
            "name": "normalisedWavenumberEstimationRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ground_to_volume_ratio_range: Optional[MinMaxNumType] = field(
        default=None,
        metadata={
            "name": "groundToVolumeRatioRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    temporal_decorrelation_estimation_range: Optional[MinMaxNumType] = field(
        default=None,
        metadata={
            "name": "temporalDecorrelationEstimationRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    temporal_decorrelation_ground_to_volume_ratio: Optional[float] = field(
        default=None,
        metadata={
            "name": "temporalDecorrelationGroundToVolumeRatio",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    residual_decorrelation: Optional[float] = field(
        default=None,
        metadata={
            "name": "residualDecorrelation",
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
    uncertainty_validvalues_limits: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "name": "uncertaintyValidvaluesLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vertical_wavenumber_validvalues_limits: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "name": "verticalWavenumberValidvaluesLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lower_height_limit: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "lowerHeightLimit",
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

    product: Optional[ProductL2AType] = field(
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
    input_information: Optional[InputInformationL2AType] = field(
        default=None,
        metadata={
            "name": "inputInformation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_parameters: Optional[ProcessingParametersL2AType] = field(
        default=None,
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    annotation_lut: Optional[LayerListType] = field(
        default=None,
        metadata={
            "name": "annotationLUT",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class MainAnnotation(MainAnnotationType):
    """
    BIOMASS L2a product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
