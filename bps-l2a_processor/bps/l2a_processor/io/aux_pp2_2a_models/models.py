# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PP2 2A models
-----------------
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
class AuxppacquisitionListType:
    """
    Parameters
    ----------
    acquisition_folder_name
        Folder name which univocally identifies an acquisition of the stack.
    count
    """

    class Meta:
        name = "AUXPPacquisitionListType"

    acquisition_folder_name: list["AuxppacquisitionListType.AcquisitionFolderName"] = field(
        default_factory=list,
        metadata={
            "name": "acquisitionFolderName",
            "type": "Element",
            "namespace": "",
            "min_occurs": 2,
            "max_occurs": 3,
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
    class AcquisitionFolderName:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        reference_image: Optional[str] = field(
            default=None,
            metadata={"name": "referenceImage", "type": "Attribute", "required": True, "pattern": r"(false)|(true)"},
        )
        average_wavenumber: Optional[float] = field(
            default=None,
            metadata={
                "name": "averageWavenumber",
                "type": "Attribute",
            },
        )


class ModelInversionType(Enum):
    """
    Default profile.
    """

    SINGLE = "single"
    DUAL = "dual"


class CalibrationScreenType(Enum):
    NONE = "none"
    GEOMETRY = "geometry"
    SKP = "skp"


@dataclass
class CompressionOptionsL2AAgb:
    """
    Parameters
    ----------
    mds
    ads
    mds_block_size
        Blocking size of all MDS.
    ads_block_size
        Blocking size of all ADS LUT.
    """

    class Meta:
        name = "compressionOptionsL2aAGB"

    mds: Optional["CompressionOptionsL2AAgb.Mds"] = field(
        default=None,
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ads: Optional["CompressionOptionsL2AAgb.Ads"] = field(
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
        gn: Optional["CompressionOptionsL2AAgb.Mds.Gn"] = field(
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
                ZLIB algorithm compression factor for the local incidence angle ADS. From 1 to 9.
            max_z_error
                For the ground cancelled backscatter images MDS, define exactly how lossy the LERC compression
                algorithm is allowed to be, specifying the absolute maximum error admitted. Zero means loss-less
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
        fnf: Optional["CompressionOptionsL2AAgb.Ads.Fnf"] = field(
            default=None,
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        local_incidence_angle: Optional["CompressionOptionsL2AAgb.Ads.LocalIncidenceAngle"] = field(
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
        class LocalIncidenceAngle:
            """
            Parameters
            ----------
            compression_factor
                ZLIB algorithm compression factor for the local incidence angle ADS. From 1 to 9.
            least_significant_digit
                For the local incidence angle ADS, define exactly how lossy the ZLIB compression algorithm is
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
class CompressionOptionsL2AFd:
    """
    Parameters
    ----------
    mds
    ads
    mds_block_size
        Blocking size of all MDS.
    ads_block_size
        Blocking size of all ADS LUT.
    """

    class Meta:
        name = "compressionOptionsL2aFD"

    mds: Optional["CompressionOptionsL2AFd.Mds"] = field(
        default=None,
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ads: Optional["CompressionOptionsL2AFd.Ads"] = field(
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
        fd: Optional["CompressionOptionsL2AFd.Mds.Fd"] = field(
            default=None,
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        probability_of_change: Optional["CompressionOptionsL2AFd.Mds.ProbabilityOfChange"] = field(
            default=None,
            metadata={
                "name": "probabilityOfChange",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        cfm: Optional["CompressionOptionsL2AFd.Mds.Cfm"] = field(
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
        fnf: Optional["CompressionOptionsL2AFd.Ads.Fnf"] = field(
            default=None,
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        acm: Optional["CompressionOptionsL2AFd.Ads.Acm"] = field(
            default=None,
            metadata={
                "name": "ACM",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        number_of_averages: Optional["CompressionOptionsL2AFd.Ads.NumberOfAverages"] = field(
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
                ZLIB algorithm compression factor valid for all the covariance ADS LUT layers. From 1 to 9.
            least_significant_digit
                For all the layers of covariance LUT ADS and for the number of averages ADS, define exactly how
                lossy the ZLIB compression algorithm is allowed to be, specifying the power of ten of the smallest
                decimal place in the data that is a reliable value. Zero means loss-less compression.
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
class CompressionOptionsL2AFh:
    """
    Parameters
    ----------
    mds
    ads
    mds_block_size
        Blocking size of all MDS.
    ads_block_size
        Blocking size of all ADS LUT.
    """

    class Meta:
        name = "compressionOptionsL2aFH"

    mds: Optional["CompressionOptionsL2AFh.Mds"] = field(
        default=None,
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ads: Optional["CompressionOptionsL2AFh.Ads"] = field(
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
        fh: Optional["CompressionOptionsL2AFh.Mds.Fh"] = field(
            default=None,
            metadata={
                "name": "FH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        quality: Optional["CompressionOptionsL2AFh.Mds.Quality"] = field(
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
        fnf: Optional["CompressionOptionsL2AFh.Ads.Fnf"] = field(
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
class CompressionOptionsL2ATfh:
    """
    Parameters
    ----------
    mds
    ads
    mds_block_size
        Blocking size of all MDS.
    ads_block_size
        Blocking size of all ADS LUT.
    """

    class Meta:
        name = "compressionOptionsL2aTFH"

    mds: Optional["CompressionOptionsL2ATfh.Mds"] = field(
        default=None,
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ads: Optional["CompressionOptionsL2ATfh.Ads"] = field(
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
        tfh: Optional["CompressionOptionsL2ATfh.Mds.Tfh"] = field(
            default=None,
            metadata={
                "name": "TFH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        quality: Optional["CompressionOptionsL2ATfh.Mds.Quality"] = field(
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
        fnf: Optional["CompressionOptionsL2ATfh.Ads.Fnf"] = field(
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
class MinMaxNumType:
    class Meta:
        name = "minMaxNumType"

    min: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    num: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


class OperationalModeType(Enum):
    MULTI_REFERENCE = "multi reference"
    SINGLE_REFERENCE = "single reference"
    INSAR_PAIR = "insar pair"


class SubsettingRuleType(Enum):
    GEOMETRY = "geometry"
    MAINTAIN_ALL = "maintain all"


class VerticalProfileOptionType(Enum):
    """
    Default profile.
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
class GeneralType:
    """
    Parameters
    ----------
    apply_calibration_screen
        Choose the phase calibration to be performed:“none”: no phase screen is applied“geometry”: only flattening
        phase screen is applied (i.e., as computed from acquisition geometry)“skp”: complete phase screen is applied
        (default)
    forest_coverage_threshold
        Minimum percentage forest coverage in L2a product footprint, triggering L2a processing.Range of values from
        0% to 100%, default 5%.
    forest_mask_interpolation_threshold
        This parameter is a threshold to fix rounding of pixels with decimal values originated from binary FNF
        interpolation onto L2a grid.This creates a safety buffer around forest border.Range of values from 0 to 1,
        default 0.5.
    subsetting_rule
        Select 3 acquisitions from the 7/8 of TOM phase, choosing, with a geometrical rule, the baselines
        corresponding to the ones of INT phase.Default value: “geometry”.
    """

    class Meta:
        name = "generalType"

    apply_calibration_screen: Optional[CalibrationScreenType] = field(
        default=None,
        metadata={
            "name": "applyCalibrationScreen",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    forest_coverage_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "forestCoverageThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    forest_mask_interpolation_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "forestMaskInterpolationThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    subsetting_rule: Optional[SubsettingRuleType] = field(
        default=None,
        metadata={
            "name": "subsettingRule",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class GroundCancellationTypeAgb:
    """
    Parameters
    ----------
    compute_gnpower_flag
        True to compute the power of ground cancelled data (absolute square), False to keep amplitude data value.
    radiometric_calibration_flag
        True if the incidence angle radiometric calibration has been applied, False otherwise.
    emphasized_forest_height
        Value of the height [m] to be emphasized during ground cancellation.
    operational_mode
        Choose the Ground Cancellation method to use. “multi reference”: multiple data computation using each image
        as reference, followed by data averaging. It is the default for AGB. “single reference”: direct computation
        with a preliminary automatic reference image selection. It is the default for FD. “insar pair”: debug
        operational mode, perform computation with only the two images specified in the element imagesPairSelection.
    images_pair_selection
        If operationalMode is “insar pair”, ground cancellation is performed using only the two images specified
        here, otherwise this element is ignored.
    disable_ground_cancellation_flag
        Disable ground cancellation for debug. (Optional, default is OFF).
    """

    class Meta:
        name = "GroundCancellationTypeAGB"

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
    images_pair_selection: Optional[AuxppacquisitionListType] = field(
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


@dataclass
class GroundCancellationTypeFd:
    """
    Parameters
    ----------
    emphasized_forest_height
        Value of the height [m] to be emphasized during ground cancellation.
    operational_mode
        Choose the Ground Cancellation method to use. “multi reference”: multiple data computation using each image
        as reference, followed by data averaging. It is the default for AGB. “single reference”: direct computation
        with a preliminary automatic reference image selection. It is the default for FD. “insar pair”: debug
        operational mode, perform computation with only the two images specified in the element imagesPairSelection.
    images_pair_selection
        If operationalMode is “insar pair”, ground cancellation is performed using only the two images specified
        here, otherwise this element is ignored.
    disable_ground_cancellation_flag
        Disable ground cancellation for debug. (Optional, default is OFF).
    """

    class Meta:
        name = "GroundCancellationTypeFD"

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
    images_pair_selection: Optional[AuxppacquisitionListType] = field(
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


@dataclass
class VerticalRangeWithUnitsType:
    class Meta:
        name = "verticalRangeWithUnitsType"

    min: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sampling: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class Agbtype:
    """
    Parameters
    ----------
    l2a_agbproduct_doi
        MDS COG blocking algorithm size and NetCDF ADS chunking algorithm size. Same value is used for both data
        array dimension
    product_id
        Product identifier: L2a AGB.
    enable_product_flag
        True to enable the AGB product computation, False to skip.
    ground_cancellation
        Ground Cancellation algorithm parameters.
    product_resolution
        Ground cancelled data averaging window size in [m] (product resolution) to be applied during multi looking.
    upsampling_factor
        Upsampling factor, default value is 2.
    compression_options
        Configurable compression options for all the L2a MDS and ADS NetCDF LUT variables.
    """

    class Meta:
        name = "AGBType"

    l2a_agbproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2aAGBProductDOI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_product_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "enableProductFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    ground_cancellation: Optional[GroundCancellationTypeAgb] = field(
        default=None,
        metadata={
            "name": "groundCancellation",
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
    upsampling_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "upsamplingFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    compression_options: Optional[CompressionOptionsL2AAgb] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class Fdtype:
    """
    Parameters
    ----------
    l2a_fdproduct_doi
        MDS COG blocking algorithm size and NetCDF ADS chunking algorithm size. Same value is used for both data
        array dimension
    product_id
        Product identifier: L2a FD.
    enable_product_flag
        True to enable the FD product computation, False to skip.
    ground_cancellation
        Ground Cancellation algorithm parameters.
    significance_level
        Confidence level to be applied in the change detection algorithm.
    product_resolution
        Product resolution in [m].
    numerical_determinant_limit
        Numerical determinant limit
    upsampling_factor
        Upsampling factor, default value is 2.
    compression_options
        Configurable compression options for all the L2a MDS and ADS NetCDF LUT variables.
    """

    class Meta:
        name = "FDType"

    l2a_fdproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2aFDProductDOI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_product_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "enableProductFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    ground_cancellation: Optional[GroundCancellationTypeFd] = field(
        default=None,
        metadata={
            "name": "groundCancellation",
            "type": "Element",
            "namespace": "",
            "required": True,
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
    compression_options: Optional[CompressionOptionsL2AFd] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class Fhtype:
    """
    Parameters
    ----------
    l2a_fhproduct_doi
        MDS COG blocking algorithm size and NetCDF ADS chunking algorithm size. Same value is used for both data
        array dimension
    product_id
        Product identifier: L2a FH.
    enable_product_flag
        True to enable the FH product computation, False to skip.
    vertical_reflectivity_option
        Specify which vertical reflectivity profile to use among the default profile
        verticalReflectivityDefaultProfile or a tomographic profile from external database (TBD).
    vertical_reflectivity_default_profile
        Default vertical reflectivity profile function, stored as a vector of float values, used if
        verticalReflectivityOption is set to “default profile”.
    model_inversion
        Model inversion algorithm to be used among single or dual baseline.
    spectral_decorrelation_compensation_flag
        True if spectral decorrelation compensation has to be performed, False otherwise.
    snrdecorrelation_compensation_flag
        True if SNR decorrelation compensation has to be performed, False otherwise.
    correct_terrain_slopes_flag
        True (Default) if terrain slope correction is to be applied, False otherwise.
    normalised_height_estimation_range
        Normalized range of height values from 0 to 1, where the canopy height estimation process has to be
        performed.
    normalised_wavenumber_estimation_range
        Normalized range of wavenumbers values from 0 to 2π, where the canopy height estimation process has to be
        performed.
    ground_to_volume_ratio_range
        Range of ground to volume ratio values to be used as valid ones [dB], during the canopy height estimation
        process, default [min, max, num] = [-20, 20, 30].
    temporal_decorrelation_estimation_range
        Range of temporal decorrelation values to be used as valid ones, during the canopy height estimation
        process.
    temporal_decorrelation_ground_to_volume_ratio
        Ratio of temporal decorrelation between ground and volume (0.0 means no temporal decorrelation for ground,
        while 1.0 means ground and volume are equally impacted by temporal decorrelation)
    residual_decorrelation
        Residual decorrelation value to be used in error model computation.
    product_resolution
        Value to be used as the resolution on ground map and also to perform the covariance averaging in radar
        coordinates. In [m].
    uncertainty_validvalues_limits
        Estimation valid values limits [m], values out of this limits are discarded and set to no data value.
    vertical_wavenumber_validvalues_limits
        Vertical wavenumber valid values limits [m], values of estimations out of this limits are discarded and set
        to no data value.
    lower_height_limit
        FH estimates lower this limit [m] are discarded and set to no data value. Default 10.0 [m]
    upsampling_factor
        Upsampling factor to decimate coherence, default value is 2.
    compression_options
        Configurable compression options for all the L2a MDS and ADS NetCDF LUT variables.
    """

    class Meta:
        name = "FHType"

    l2a_fhproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2aFHProductDOI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_product_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "enableProductFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
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
    snrdecorrelation_compensation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "SNRDecorrelationCompensationFlag",
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
    normalised_height_estimation_range: Optional["Fhtype.NormalisedHeightEstimationRange"] = field(
        default=None,
        metadata={
            "name": "normalisedHeightEstimationRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    normalised_wavenumber_estimation_range: Optional["Fhtype.NormalisedWavenumberEstimationRange"] = field(
        default=None,
        metadata={
            "name": "normalisedWavenumberEstimationRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ground_to_volume_ratio_range: Optional["Fhtype.GroundToVolumeRatioRange"] = field(
        default=None,
        metadata={
            "name": "groundToVolumeRatioRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    temporal_decorrelation_estimation_range: Optional["Fhtype.TemporalDecorrelationEstimationRange"] = field(
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
    uncertainty_validvalues_limits: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "uncertaintyValidvaluesLimits",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vertical_wavenumber_validvalues_limits: Optional[MinMaxTypeWithUnit] = field(
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
    compression_options: Optional[CompressionOptionsL2AFh] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class NormalisedHeightEstimationRange(MinMaxType):
        unit: Optional[object] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class NormalisedWavenumberEstimationRange(MinMaxNumType):
        unit: Optional[object] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class GroundToVolumeRatioRange(MinMaxNumType):
        unit: Optional[object] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class TemporalDecorrelationEstimationRange(MinMaxNumType):
        unit: Optional[object] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class Tfhtype:
    """
    Parameters
    ----------
    l2a_tfhproduct_doi
        MDS COG blocking algorithm size and NetCDF ADS chunking algorithm size. Same value is used for both data
        array dimension
    product_id
        Product identifier: L2a TOMO FH.
    enable_product_flag
        True to enable the FH product computation, False to skip.
    enable_super_resolution
        True to enable the TOMO FH super resolution algorithm.
    product_resolution
        Value to be used as the resolution on ground map and also to perform the covariance averaging in radar
        coordinates. In [m].
    regularization_noise_factor
        regularization Noise Factor
    power_threshold
        power threshold
    median_factor
        median Factor
    estimation_valid_values_limits
        Estimation valid values limits [m], values of estimations out of this limits are discarded and set to no
        data value
    vertical_range
        Vertical range minimum and maximum height [m], sampling
    compression_options
        Configurable compression options for all the L2a MDS and ADS NetCDF LUT variables.
    """

    class Meta:
        name = "TFHType"

    l2a_tfhproduct_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l2aTFHProductDOI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_product_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "enableProductFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
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
    vertical_range: Optional[VerticalRangeWithUnitsType] = field(
        default=None,
        metadata={
            "name": "verticalRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    compression_options: Optional[CompressionOptionsL2ATfh] = field(
        default=None,
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryL2AProcessingParametersType:
    """
    Parameters
    ----------
    general
        L2a common processing parameters shared by the four l2a products (AGB, FD, FH, TOMO FH).
    agb
        L2a processing parameters for the AGB product.
    fd
        L2a processing parameters for the FD product.
    fh
        L2a processing parameters for the FH product.
    tfh
        L2a processing parameters for the TOMO FH product.
    """

    class Meta:
        name = "auxiliaryL2aProcessingParametersType"

    general: Optional[GeneralType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    agb: Optional[Agbtype] = field(
        default=None,
        metadata={
            "name": "AGB",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    fd: Optional[Fdtype] = field(
        default=None,
        metadata={
            "name": "FD",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    fh: Optional[Fhtype] = field(
        default=None,
        metadata={
            "name": "FH",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tfh: Optional[Tfhtype] = field(
        default=None,
        metadata={
            "name": "TFH",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryL2AProcessingParameters(AuxiliaryL2AProcessingParametersType):
    """
    BIOMASS auxiliary L2a processing parameters for each product (AGB, FD, FH, TOMO
    FH) and common parameters for the four products..
    """

    class Meta:
        name = "auxiliaryL2aProcessingParameters"
