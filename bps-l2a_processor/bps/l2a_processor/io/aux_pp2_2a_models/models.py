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
    PolarisationCombinationMethodType,
    PolarisationType,
    SlantRangePolynomialType,
    StateType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
)


@dataclass(kw_only=True)
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

    acquisition_folder_name: list[AuxppacquisitionListType.AcquisitionFolderName] = field(
        default_factory=list,
        metadata={
            "name": "acquisitionFolderName",
            "type": "Element",
            "namespace": "",
            "min_occurs": 2,
            "max_occurs": 3,
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )

    @dataclass(kw_only=True)
    class AcquisitionFolderName:
        value: str = field(default="")
        reference_image: str = field(
            metadata={"name": "referenceImage", "type": "Attribute", "pattern": r"(false)|(true)"}
        )
        average_wavenumber: None | float = field(
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


@dataclass(kw_only=True)
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

    mds: CompressionOptionsL2AAgb.Mds = field(
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
        },
    )
    ads: CompressionOptionsL2AAgb.Ads = field(
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
        gn: CompressionOptionsL2AAgb.Mds.Gn = field(
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
                ZLIB algorithm compression factor for the local incidence angle ADS. From 1 to 9.
            max_z_error
                For the ground cancelled backscatter images MDS, define exactly how lossy the LERC compression
                algorithm is allowed to be, specifying the absolute maximum error admitted. Zero means loss-less
                compression.
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
        fnf: CompressionOptionsL2AAgb.Ads.Fnf = field(
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
            },
        )
        local_incidence_angle: CompressionOptionsL2AAgb.Ads.LocalIncidenceAngle = field(
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

    mds: CompressionOptionsL2AFd.Mds = field(
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
        },
    )
    ads: CompressionOptionsL2AFd.Ads = field(
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
        fd: CompressionOptionsL2AFd.Mds.Fd = field(
            metadata={
                "name": "FD",
                "type": "Element",
                "namespace": "",
            },
        )
        probability_of_change: CompressionOptionsL2AFd.Mds.ProbabilityOfChange = field(
            metadata={
                "name": "probabilityOfChange",
                "type": "Element",
                "namespace": "",
            },
        )
        cfm: CompressionOptionsL2AFd.Mds.Cfm = field(
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
        fnf: CompressionOptionsL2AFd.Ads.Fnf = field(
            metadata={
                "name": "FNF",
                "type": "Element",
                "namespace": "",
            },
        )
        acm: CompressionOptionsL2AFd.Ads.Acm = field(
            metadata={
                "name": "ACM",
                "type": "Element",
                "namespace": "",
            },
        )
        number_of_averages: CompressionOptionsL2AFd.Ads.NumberOfAverages = field(
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
                ZLIB algorithm compression factor valid for all the covariance ADS LUT layers. From 1 to 9.
            least_significant_digit
                For all the layers of covariance LUT ADS and for the number of averages ADS, define exactly how
                lossy the ZLIB compression algorithm is allowed to be, specifying the power of ten of the smallest
                decimal place in the data that is a reliable value. Zero means loss-less compression.
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

    mds: CompressionOptionsL2AFh.Mds = field(
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
        },
    )
    ads: CompressionOptionsL2AFh.Ads = field(
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
        fh: CompressionOptionsL2AFh.Mds.Fh = field(
            metadata={
                "name": "FH",
                "type": "Element",
                "namespace": "",
            },
        )
        quality: CompressionOptionsL2AFh.Mds.Quality = field(
            metadata={
                "name": "Quality",
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
    class Ads:
        fnf: CompressionOptionsL2AFh.Ads.Fnf = field(
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

    mds: CompressionOptionsL2ATfh.Mds = field(
        metadata={
            "name": "MDS",
            "type": "Element",
            "namespace": "",
        },
    )
    ads: CompressionOptionsL2ATfh.Ads = field(
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
        tfh: CompressionOptionsL2ATfh.Mds.Tfh = field(
            metadata={
                "name": "TFH",
                "type": "Element",
                "namespace": "",
            },
        )
        quality: CompressionOptionsL2ATfh.Mds.Quality = field(
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
        fnf: CompressionOptionsL2ATfh.Ads.Fnf = field(
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
class MinMaxNumType:
    class Meta:
        name = "minMaxNumType"

    min: float = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    max: float = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    num: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
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
    default profile.
    """

    DEFAULT_PROFILE = "default profile"


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
    polarisation_combination_method
        Polarisations combination method: “HV” or “VH”, if just one of the two is selected (in addition to HH and VV
        ones); “Average”, if the average of HV and VH is computed and used; “None” if no combination is performed
        (all the four polarisations are used).
    """

    class Meta:
        name = "generalType"

    apply_calibration_screen: CalibrationScreenType = field(
        metadata={
            "name": "applyCalibrationScreen",
            "type": "Element",
            "namespace": "",
        },
    )
    forest_coverage_threshold: float = field(
        metadata={
            "name": "forestCoverageThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    forest_mask_interpolation_threshold: float = field(
        metadata={
            "name": "forestMaskInterpolationThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    subsetting_rule: SubsettingRuleType = field(
        metadata={
            "name": "subsettingRule",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation_combination_method: PolarisationCombinationMethodType = field(
        metadata={
            "name": "polarisationCombinationMethod",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    images_pair_selection: None | AuxppacquisitionListType = field(
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


@dataclass(kw_only=True)
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
    images_pair_selection: None | AuxppacquisitionListType = field(
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


@dataclass(kw_only=True)
class VerticalRangeWithUnitsType:
    class Meta:
        name = "verticalRangeWithUnitsType"

    min: FloatWithUnit = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    max: FloatWithUnit = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    sampling: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    l2a_agbproduct_doi: str = field(
        metadata={
            "name": "l2aAGBProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    product_id: str = field(
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_product_flag: str = field(
        metadata={"name": "enableProductFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    ground_cancellation: GroundCancellationTypeAgb = field(
        metadata={
            "name": "groundCancellation",
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
    upsampling_factor: int = field(
        metadata={
            "name": "upsamplingFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    compression_options: CompressionOptionsL2AAgb = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    l2a_fdproduct_doi: str = field(
        metadata={
            "name": "l2aFDProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    product_id: str = field(
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_product_flag: str = field(
        metadata={"name": "enableProductFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    ground_cancellation: GroundCancellationTypeFd = field(
        metadata={
            "name": "groundCancellation",
            "type": "Element",
            "namespace": "",
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
    compression_options: CompressionOptionsL2AFd = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    l2a_fhproduct_doi: str = field(
        metadata={
            "name": "l2aFHProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    product_id: str = field(
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_product_flag: str = field(
        metadata={"name": "enableProductFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    vertical_reflectivity_option: VerticalProfileOptionType = field(
        metadata={
            "name": "verticalReflectivityOption",
            "type": "Element",
            "namespace": "",
        },
    )
    vertical_reflectivity_default_profile: VerticalReflectivityProfileType = field(
        metadata={
            "name": "verticalReflectivityDefaultProfile",
            "type": "Element",
            "namespace": "",
        },
    )
    model_inversion: ModelInversionType = field(
        metadata={
            "name": "modelInversion",
            "type": "Element",
            "namespace": "",
        },
    )
    spectral_decorrelation_compensation_flag: str = field(
        metadata={
            "name": "spectralDecorrelationCompensationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    snrdecorrelation_compensation_flag: str = field(
        metadata={
            "name": "SNRDecorrelationCompensationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    correct_terrain_slopes_flag: str = field(
        metadata={"name": "correctTerrainSlopesFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    normalised_height_estimation_range: Fhtype.NormalisedHeightEstimationRange = field(
        metadata={
            "name": "normalisedHeightEstimationRange",
            "type": "Element",
            "namespace": "",
        },
    )
    normalised_wavenumber_estimation_range: Fhtype.NormalisedWavenumberEstimationRange = field(
        metadata={
            "name": "normalisedWavenumberEstimationRange",
            "type": "Element",
            "namespace": "",
        },
    )
    ground_to_volume_ratio_range: Fhtype.GroundToVolumeRatioRange = field(
        metadata={
            "name": "groundToVolumeRatioRange",
            "type": "Element",
            "namespace": "",
        },
    )
    temporal_decorrelation_estimation_range: Fhtype.TemporalDecorrelationEstimationRange = field(
        metadata={
            "name": "temporalDecorrelationEstimationRange",
            "type": "Element",
            "namespace": "",
        },
    )
    temporal_decorrelation_ground_to_volume_ratio: float = field(
        metadata={
            "name": "temporalDecorrelationGroundToVolumeRatio",
            "type": "Element",
            "namespace": "",
        },
    )
    residual_decorrelation: float = field(
        metadata={
            "name": "residualDecorrelation",
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
    uncertainty_validvalues_limits: MinMaxTypeWithUnit = field(
        metadata={
            "name": "uncertaintyValidvaluesLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    vertical_wavenumber_validvalues_limits: MinMaxTypeWithUnit = field(
        metadata={
            "name": "verticalWavenumberValidvaluesLimits",
            "type": "Element",
            "namespace": "",
        },
    )
    lower_height_limit: FloatWithUnit = field(
        metadata={
            "name": "lowerHeightLimit",
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
    compression_options: CompressionOptionsL2AFh = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class NormalisedHeightEstimationRange(MinMaxType):
        unit: None | object = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass(kw_only=True)
    class NormalisedWavenumberEstimationRange(MinMaxNumType):
        unit: None | object = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass(kw_only=True)
    class GroundToVolumeRatioRange(MinMaxNumType):
        unit: None | object = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass(kw_only=True)
    class TemporalDecorrelationEstimationRange(MinMaxNumType):
        unit: None | object = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass(kw_only=True)
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

    l2a_tfhproduct_doi: str = field(
        metadata={
            "name": "l2aTFHProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    product_id: str = field(
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_product_flag: str = field(
        metadata={"name": "enableProductFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
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
    vertical_range: VerticalRangeWithUnitsType = field(
        metadata={
            "name": "verticalRange",
            "type": "Element",
            "namespace": "",
        },
    )
    compression_options: CompressionOptionsL2ATfh = field(
        metadata={
            "name": "compressionOptions",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    general: GeneralType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    agb: Agbtype = field(
        metadata={
            "name": "AGB",
            "type": "Element",
            "namespace": "",
        },
    )
    fd: Fdtype = field(
        metadata={
            "name": "FD",
            "type": "Element",
            "namespace": "",
        },
    )
    fh: Fhtype = field(
        metadata={
            "name": "FH",
            "type": "Element",
            "namespace": "",
        },
    )
    tfh: Tfhtype = field(
        metadata={
            "name": "TFH",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryL2AProcessingParameters(AuxiliaryL2AProcessingParametersType):
    """
    BIOMASS auxiliary L2a processing parameters for each product (AGB, FD, FH, TOMO FH) and common parameters for
    the four products..
    """

    class Meta:
        name = "auxiliaryL2aProcessingParameters"
