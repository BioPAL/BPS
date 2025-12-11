# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD common annotation models L2
-------------------------------
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


class CalibrationScreenType(Enum):
    NONE = "none"
    GEOMETRY = "geometry"
    SKP = "skp"


@dataclass
class IntegerListType:
    """
    Parameters
    ----------
    val
        Integer numbers list.
    """

    class Meta:
        name = "integerListType"

    val: list[int] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


@dataclass
class MinMaxNumType:
    """
    Enumeration of min max.
    """

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


@dataclass
class NoDataValueChoiceType:
    """
    Parameters
    ----------
    float_no_data_value
        Pixel value in case of invalid data, for float32 data.
    int_no_data_value
        Pixel value in case of invalid data, for uint8 data.
    """

    class Meta:
        name = "noDataValueChoiceType"

    float_no_data_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "floatNoDataValue",
            "type": "Element",
            "namespace": "",
        },
    )
    int_no_data_value: Optional[int] = field(
        default=None,
        metadata={
            "name": "intNoDataValue",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class PercentPixelsType:
    """
    Parameters
    ----------
    percentage
        percentage value of valid pixels
    pixels
        absolute number of valid pixels
    """

    class Meta:
        name = "percentPixelsType"

    percentage: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixels: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class SelectedReferenceImageType:
    class Meta:
        name = "selectedReferenceImageType"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    acquisition_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "acquisitionID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class StringListType:
    """
    Parameters
    ----------
    id
        String (identifiers) list, used for the DGG tiles in input neighbourhood for AGB parameter estimation
    """

    class Meta:
        name = "stringListType"

    id: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ID",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


class SubsettingRuleType(Enum):
    """Enumeration of available rules for INT subsetting: select three acquisitions from all the ones in input (when input is TOM phase)."""

    GEOMETRY = "geometry"
    MAINTAIN_ALL = "maintain all"


@dataclass
class GnfilteredCoverageType:
    """
    Parameters
    ----------
    after_global_filtering
        Input GN product filtered coverage, percentage and number of pixel values after all filterings
        concatenate.[RD] BPS_AGB_ATBD v3.1.2, equation 4.5
    after_sigma_filtering
        Input GN product filtered coverage, disaggregated percentage and number of pixel values after filtering only
        for GN Sigma (backscatter) values limits.[RD] BPS_AGB_ATBD v3.1.2, equation 4.5, σ_p^0
    after_angle_filtering
        Input GN product filtered coverage, disaggregated percentage and number of pixel values after filtering only
        for GN Incidence Angle values limits. [RD] BPS_AGB_ATBD v3.1.2, equation 4.5, ϑ
    """

    class Meta:
        name = "GNfilteredCoverageType"

    after_global_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterGlobalFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    after_sigma_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterSigmaFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    after_angle_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterAngleFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class BackscatterLimitsType:
    class Meta:
        name = "backscatterLimitsType"

    hh: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vh: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vv: Optional[MinMaxType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class CalAbfilteredCoverageType:
    """
    Parameters
    ----------
    after_global_filtering
        CAL_AB filtered coverage, percentage and number of pixel values after all filterings concatenated.[RD]
        BPS_AGB_ATBD v3.1.2, equation 4.6
    after_agbvalue_filtering
        CAL_AB filtered coverage, disaggregated percentage and number of pixel values after filtering only for mean
        AGB values limits.[RD] BPS_AGB_ATBD v3.1.2, equation 4.6, W
    after_agbstd_filtering
        CAL_AB filtered coverage, disaggregated percentage and number of pixel values after filtering only for AGB
        STD limits.[RD] BPS_AGB_ATBD v3.1.2, equation 4.6, U
    after_agbrelative_std_filtering
        CAL_AB filtered coverage, disaggregated percentage and number of pixel values after filtering only for AGB
        relative STD limits.[RD] BPS_AGB_ATBD v3.1.2, equation 4.6, R
    after_lcmclass_filtering
        CAL_AB filtered coverage, disaggregated percentage and number of pixel values after filtering only for LCM
        class. RD] BPS_AGB_ATBD v3.1.2, equation 4.6, Ω_fcrej
    """

    class Meta:
        name = "calABfilteredCoverageType"

    after_global_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterGlobalFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    after_agbvalue_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterAGBvalueFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    after_agbstd_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterAGBstdFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    after_agbrelative_std_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterAGBrelativeStdFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    after_lcmclass_filtering: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "afterLCMClassFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class GeneralConfigurationParametersType:
    """
    Parameters
    ----------
    apply_calibration_screen
        The performed phase calibration:“none”: no phase screen has been applied; “geometry”: only flattening phase
        screen has been applied (i.e., as computed from acquisition geometry); “skp”: complete phase screen has been
        applied (default)
    forest_coverage_threshold
        Minimum percentage forest coverage in L2a product footprint, used to trigger L2a processing. Range of values
        from 0% to 100%, default 5%.
    forest_mask_interpolation_threshold
        This parameter is a threshold used to fix rounding of pixels with decimal values originated from binary FNF
        interpolation onto L2a grid.
    subsetting_rule
        In case of more than three acquisitions in input (TOM phase), this is the rule which has been used to select
        3 acquisitions from the 7/8 of TOM phase, choosing, the baselines corresponding to the ones of INT phase.
    """

    class Meta:
        name = "generalConfigurationParametersType"

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
class PixelRepresentationChoiceType:
    class Meta:
        name = "pixelRepresentationChoiceType"

    fd: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "FD",
            "type": "Element",
            "namespace": "",
        },
    )
    cfm: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "CFM",
            "type": "Element",
            "namespace": "",
        },
    )
    probability_of_change: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "probabilityOfChange",
            "type": "Element",
            "namespace": "",
        },
    )
    fd_heat_map: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "FD_heatMap",
            "type": "Element",
            "namespace": "",
        },
    )
    fh: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "FH",
            "type": "Element",
            "namespace": "",
        },
    )
    quality: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "Quality",
            "type": "Element",
            "namespace": "",
        },
    )
    fh_heat_map: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "FH_heatMap",
            "type": "Element",
            "namespace": "",
        },
    )
    gn: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "GN",
            "type": "Element",
            "namespace": "",
        },
    )
    agb: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "AGB",
            "type": "Element",
            "namespace": "",
        },
    )
    agb_standard_deviation: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "AGB_STANDARD_DEVIATION",
            "type": "Element",
            "namespace": "",
        },
    )
    agb_heat_map: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "AGB_heatMap",
            "type": "Element",
            "namespace": "",
        },
    )
    bps_fnf: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "BPS_FNF",
            "type": "Element",
            "namespace": "",
        },
    )
    acquisition_id_image: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "Acquisition_id_image",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class PixelTypeChoiceType:
    """
    Parameters
    ----------
    float_pixel_type
        Data type of output pixels within the image, for float32 data.
    int_pixel_type
        Data type of output pixels within the image, for uint8 data.
    """

    class Meta:
        name = "pixelTypeChoiceType"

    float_pixel_type: Optional[PixelTypeType] = field(
        default=None,
        metadata={
            "name": "floatPixelType",
            "type": "Element",
            "namespace": "",
        },
    )
    int_pixel_type: Optional[PixelTypeType] = field(
        default=None,
        metadata={
            "name": "intPixelType",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class PolarisationListType:
    """
    Parameters
    ----------
    polarisation
        Polarisation (HH, VH, VV).
    count
    """

    class Meta:
        name = "polarisationListType"

    polarisation: list[PolarisationType] = field(
        default_factory=list, metadata={"type": "Element", "namespace": "", "min_occurs": 3, "max_occurs": 3}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class StaQualityParametersType:
    """
    Parameters
    ----------
    invalid_l1a_data_samples
        Number of invalid L1a data samples (according to their mean and standard deviation), expressed as a
        normalized fraction of the total number of samples.
    rfi_decorrelation
        Decorrelation factor due to RFI strength.
    rfi_decorrelation_threshold
        Maximum decorrelation factor due to RFI strength admissible to use the image as the primary image.
    faraday_decorrelation
        Decorrelation factor due to Faraday residual.
    faraday_decorrelation_threshold
        Maximum decorrelation factor due to Faraday residual admissible to use the image as the primary image.
    invalid_residual_shifts_fraction
        Number of invalid residual coregistration shifts, expressed as a normalized fraction of the total number of
        residual coregistration shifts.
    residual_shifts_quality_threshold
        Threshold on residual coregistration shifts quality (between 0 and 1). Residual shifts with a quality lower
        than this threshold are not used for processing.
    invalid_ground_phases_screen_estimates_fraction
        Number of invalid ground phase screen estimates, expressed as a normalized fraction of the total number of
        ground phase screen estimates.
    ground_phases_screen_quality_threshold
        Threshold on ground phase screen estimation quality (between 0 and 1). Estimates with a quality lower than
        this threshold are not corrected. Used only in case skpPhaseCorrectionFlag is set to True.
    skp_decomposition_index
        Error code for the SKP decomposition. Positive if the SKP decomposition hit a nonblocking contingency case
        (e.g. SVD decomposition failure etc.) and the exported SKP-related LUTs contain values obtained from a
        contingency handling procedure. 0 otherwise.
    polarisation
    """

    class Meta:
        name = "staQualityParametersType"

    invalid_l1a_data_samples: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidL1aDataSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_decorrelation: Optional[float] = field(
        default=None,
        metadata={
            "name": "rfiDecorrelation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_decorrelation_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "rfiDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    faraday_decorrelation: Optional[float] = field(
        default=None,
        metadata={
            "name": "faradayDecorrelation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    faraday_decorrelation_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "faradayDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_residual_shifts_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidResidualShiftsFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    residual_shifts_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "residualShiftsQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_ground_phases_screen_estimates_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidGroundPhasesScreenEstimatesFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ground_phases_screen_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "groundPhasesScreenQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_decomposition_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "skpDecompositionIndex",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class GncoverageType:
    """
    Parameters
    ----------
    original_coverage
        Input GN product original coverage (current GN and current tile) percentage and number of pixel values
    filtered_coverage
        Input GN product filtered (current GN and current tile) percentage and number of pixel values after each
        filtering.[RD] BPS_AGB_ATBD v3.1.2, equation 4.5
    acquisition_id
    """

    class Meta:
        name = "GNcoverageType"

    original_coverage: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "originalCoverage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    filtered_coverage: Optional[GnfilteredCoverageType] = field(
        default=None,
        metadata={
            "name": "filteredCoverage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    acquisition_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "acquisitionID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class CalAbcoverageType:
    """
    Parameters
    ----------
    original_coverage
        CAL_AB original coverage, percentage and number of pixel values
    filtered_coverage
        CAL_AB filtered coverage, percentage and number of pixel values after each filtering. [RD] BPS_AGB_ATBD
        v3.1.2, equation 4.6
    id
    """

    class Meta:
        name = "calABcoverageType"

    original_coverage: Optional[PercentPixelsType] = field(
        default=None,
        metadata={
            "name": "originalCoverage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    filtered_coverage: Optional[CalAbfilteredCoverageType] = field(
        default=None,
        metadata={
            "name": "filteredCoverage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ProductL2AType:
    """
    Parameters
    ----------
    mission
        Mission (BIOMASS).
    tile_id
        List of DGG geographic tile identifiers.
    basin_id
        List of DGG basin identifiers.
    product_type
        Product type
    start_time
        Start time of the image [UTC].
    stop_time
        Stop time of the image [UTC].
    radar_carrier_frequency
        Radar carrier frequency [Hz].
    mission_phase_id
        Mission phase identifier (COMMISSIONING, TOMOGRAPHIC, INTERFEROMETRIC).
    sensor_mode
        Sensor mode (always "measurement")
    global_coverage_id
        Global coverage identifier.
    swath
        Swath (S1, S2, S3).
    major_cycle_id
        Major cycle identifier.
    absolute_orbit_number
        List of absolute orbit numbers at start time, one for each L1c.
    relative_orbit_number
        Relative orbit number (track) at start time.
    orbit_pass
        Orbit pass (Ascending, Descending).
    data_take_id
        List of data take identifiers, one for each L1c.
    frame
        Frame identifier.
    platform_heading
        Platform heading relative to North [deg].
    forest_coverage_percentage
        Forest coverage of the input STA products, compute as a percentage of all the pixels in the STA footprint
    selected_reference_image
        Index of the input STA acquisition selected as optimal reference image (guaranteeing the widest scene
        coverage), used during ground cancellation. Optional element, present only when ground cancellation is
        performed (FD, GN) and operationalMode configuration has been set to “single reference”
    """

    class Meta:
        name = "productL2aType"

    mission: Optional[MissionType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tile_id: Optional[StringListType] = field(
        default=None,
        metadata={
            "name": "tileID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    basin_id: Optional[StringListType] = field(
        default=None,
        metadata={
            "name": "basinID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_type: Optional[ProductType] = field(
        default=None,
        metadata={
            "name": "productType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    start_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "startTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    stop_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "stopTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    radar_carrier_frequency: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "radarCarrierFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    mission_phase_id: Optional[MissionPhaseIdtype] = field(
        default=None,
        metadata={
            "name": "missionPhaseID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sensor_mode: Optional[SensorModeType] = field(
        default=None,
        metadata={
            "name": "sensorMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    global_coverage_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "globalCoverageID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swath: Optional[SwathType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    major_cycle_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "majorCycleID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    absolute_orbit_number: Optional[IntegerListType] = field(
        default=None,
        metadata={
            "name": "absoluteOrbitNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    relative_orbit_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "relativeOrbitNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    orbit_pass: Optional[OrbitPassType] = field(
        default=None,
        metadata={
            "name": "orbitPass",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    data_take_id: Optional[IntegerListType] = field(
        default=None,
        metadata={
            "name": "dataTakeID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    frame: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    platform_heading: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "platformHeading",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    forest_coverage_percentage: Optional[float] = field(
        default=None,
        metadata={
            "name": "forestCoveragePercentage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    selected_reference_image: Optional[SelectedReferenceImageType] = field(
        default=None,
        metadata={
            "name": "selectedReferenceImage",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class RasterImageType:
    """
    Parameters
    ----------
    footprint
        Image footprint, expressed as a list of latitude, longitude tuples in NE,SE,SW,NW order [deg].
    first_latitude_value
        First latitude value of the image [deg].
    first_longitude_value
        First longitude value of the image [deg].
    latitude_spacing
        Latitude spacing between samples [deg].
    longitude_spacing
        Loigitude spacing between samples [deg].
    number_of_samples
        Total number of samples in the image.
    number_of_lines
        Total number of lines in the image.
    projection
        Projection of the image: latitude longitude based on DGG.
    datum
        Datum used during processing.
    pixel_representation
        Representation of the image pixels within the image.
    pixel_type
        Representation of the image pixels within the image.
    no_data_value
        Representation of the image pixels within the image.
    """

    class Meta:
        name = "rasterImageType"

    footprint: Optional[FloatArrayWithUnits] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    first_latitude_value: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "firstLatitudeValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    first_longitude_value: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "firstLongitudeValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    latitude_spacing: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "latitudeSpacing",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    longitude_spacing: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "longitudeSpacing",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_samples: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    number_of_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "numberOfLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    projection: Optional[ProjectionType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    datum: Optional[DatumType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_representation: Optional[PixelRepresentationChoiceType] = field(
        default=None,
        metadata={
            "name": "pixelRepresentation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_type: Optional[PixelTypeChoiceType] = field(
        default=None,
        metadata={
            "name": "pixelType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    no_data_value: Optional[NoDataValueChoiceType] = field(
        default=None,
        metadata={
            "name": "noDataValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class StaQualityParametersListType:
    """
    Parameters
    ----------
    sta_quality_parameters
        Quality parameters for the current polarisation.
    count
    """

    class Meta:
        name = "staQualityParametersListType"

    sta_quality_parameters: list[StaQualityParametersType] = field(
        default_factory=list,
        metadata={"name": "staQualityParameters", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class GncoverageListType:
    """
    Parameters
    ----------
    gn
        Input GN product coverage, current GN and current tile
    id
    """

    class Meta:
        name = "GNcoverageListType"

    gn: list[GncoverageType] = field(
        default_factory=list,
        metadata={
            "name": "GN",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class CalAbcoverageTilesListType:
    """
    Parameters
    ----------
    tile
        CAL_AB coverage, current tile
    """

    class Meta:
        name = "calABcoverageTilesListType"

    tile: list[CalAbcoverageType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


@dataclass
class StaQualityType:
    """
    Parameters
    ----------
    overall_product_quality_index
        Overall product quality index. This annotation is calculated based on specific quality parameters and gives
        an overall quality value to the product. Equal to 0 for valid products and to 1 for invalid ones.
    sta_quality_parameters_list
        Quality parameters list. The list contains an entry for each polarisation.
    """

    class Meta:
        name = "staQualityType"

    overall_product_quality_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "overallProductQualityIndex",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_quality_parameters_list: Optional[StaQualityParametersListType] = field(
        default=None,
        metadata={
            "name": "staQualityParametersList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class GncoverageTilesListType:
    """
    Parameters
    ----------
    tile
        Input GN products coverage, current tile
    """

    class Meta:
        name = "GNcoverageTilesListType"

    tile: list[GncoverageListType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


@dataclass
class AcquisitionType:
    """
    Parameters
    ----------
    folder_name
        Folder name which univocally identifies an acquisition of the stack.
    sta_quality
    reference_image
    average_wavenumber
    """

    class Meta:
        name = "acquisitionType"

    folder_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "FolderName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_quality: Optional[StaQualityType] = field(
        default=None,
        metadata={
            "name": "staQuality",
            "type": "Element",
            "namespace": "",
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


@dataclass
class AcquisitionListType:
    """
    Parameters
    ----------
    acquisition
        Info about acquisition of the stack.
    count
    """

    class Meta:
        name = "acquisitionListType"

    acquisition: list[AcquisitionType] = field(
        default_factory=list, metadata={"type": "Element", "namespace": "", "min_occurs": 2, "max_occurs": 8}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ProductL2BL3Type:
    """
    Parameters
    ----------
    mission
        Mission (BIOMASS).
    tile_id
        DGG geographic tile identifier.
    basin_id
        List of DGG basin identifiers.
    product_type
        Product type
    start_time
        Start time of the image [UTC].
    stop_time
        Stop time of the image [UTC].
    radar_carrier_frequency
        Radar carrier frequency [Hz].
    mission_phase_id
        Mission phase identifier (COMMISSIONING, TOMOGRAPHIC, INTERFEROMETRIC).
    sensor_mode
        Sensor mode (always "measurement")
    global_coverage_id
        Global coverage identifier.
    contributing_tiles
        List of all the DGG tiles in the input neighbourhood, contributing for AGB parameter estimation
    cal_abcoverage_per_tile
        CAL_AB coverage per tile [% and absolute numbers], for each of the tiles in the input neighbourhood for AGB
        parameter estimation
    gncoverage_per_tile
        Input GN products coverage per tile [% and absolute numbers], for each of the tiles in the input
        neighbourhood for AGB parameter estimation
    """

    class Meta:
        name = "productL2bL3Type"

    mission: Optional[MissionType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    tile_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "tileID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    basin_id: Optional[StringListType] = field(
        default=None,
        metadata={
            "name": "basinID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_type: Optional[ProductType] = field(
        default=None,
        metadata={
            "name": "productType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    start_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "startTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    stop_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "stopTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    radar_carrier_frequency: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "radarCarrierFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    mission_phase_id: Optional[MissionPhaseIdtype] = field(
        default=None,
        metadata={
            "name": "missionPhaseID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sensor_mode: Optional[SensorModeType] = field(
        default=None,
        metadata={
            "name": "sensorMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    global_coverage_id: Optional[int] = field(
        default=None,
        metadata={
            "name": "globalCoverageID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    contributing_tiles: Optional[StringListType] = field(
        default=None,
        metadata={
            "name": "contributingTiles",
            "type": "Element",
            "namespace": "",
        },
    )
    cal_abcoverage_per_tile: Optional[CalAbcoverageTilesListType] = field(
        default=None,
        metadata={
            "name": "calABcoveragePerTile",
            "type": "Element",
            "namespace": "",
        },
    )
    gncoverage_per_tile: Optional[GncoverageTilesListType] = field(
        default=None,
        metadata={
            "name": "GNcoveragePerTile",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class InputInformationL2AType:
    """
    Parameters
    ----------
    product_type
        Product type ("STA")
    overall_products_quality_index
        Quality index based on the "overallProductQualityIndex" of the single STA input data, giving a qualitative
        overall idea of the input stack. Equal to "0" if all STA feature "overallProductQualityIndex"=False and to
        "1" if at least one is degraded.
    nominal_stack
        True if it is a nominal stack (3 STA products for INT phase or 7 STA products for TOM phase), False
        otherwise
    polarisation_list
        List of polarisations.
    projection
        Projection of the input acquisitions, always Slant Range.
    footprint
        Image footprint, expressed as a list of latitude, longitude tuples in NE,SE,SW,NW order [deg].
    vertical_wavenumbers
        Minimum and maximum values of the vertical wavenumbers [rad/m] in the read and used Input STA products
    height_of_ambiguity
        Minimum and maximum values of the height of ambiguity (HoA) [m], computed from the vertical wavenumbers in
        the read and used Input STA products
    acquisition_list
        Folder name of each acquisition composing the stack.
    """

    class Meta:
        name = "InputInformationL2aType"

    product_type: Optional[ProductType] = field(
        default=None,
        metadata={
            "name": "productType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    overall_products_quality_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "overallProductsQualityIndex",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    nominal_stack: Optional[str] = field(
        default=None,
        metadata={
            "name": "nominalStack",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    polarisation_list: Optional[PolarisationListType] = field(
        default=None,
        metadata={
            "name": "polarisationList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    projection: Optional[ProjectionType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    footprint: Optional[FloatArrayWithUnits] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    vertical_wavenumbers: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "verticalWavenumbers",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    height_of_ambiguity: Optional[MinMaxTypeWithUnit] = field(
        default=None,
        metadata={
            "name": "heightOfAmbiguity",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    acquisition_list: Optional[AcquisitionListType] = field(
        default=None,
        metadata={
            "name": "acquisitionList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


class OperationalModeType(Enum):
    """Enumeration of available rules for INT subsetting: select three acquisitions from all the ones in input (when input is TOM phase)."""

    SINGLE_REFERENCE = "single reference"
    MULTI_REFERENCE = "multi reference"
    INSAR_PAIR = "insar pair"


class BpsFnfType(Enum):
    """Enumeration of Forest Mask: CFM from FD processor or FNF."""

    FNF = "FNF"
    CFM = "CFM"


class AgbIndexingType(Enum):
    """
    Enumeration of AGB algorithm indexings.
    """

    NONE = "none"
    P = "p"
    J = "j"
    K = "k"
    PJ = "pj"
    PK = "pk"
    JK = "jk"
    PJK = "pjk"


@dataclass
class InputInformationL2BL3ListType:
    class Meta:
        name = "InputInformationL2bL3ListType"

    l2a_inputs: list["InputInformationL2BL3ListType.L2AInputs"] = field(
        default_factory=list,
        metadata={
            "name": "L2aInputs",
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
    class L2AInputs:
        """
        Parameters
        ----------
        l2a_product_folder_name
            L2a DSR product folder name.
        l2a_product_date
            L2a product date.
        l1_inputs
        significance_level
        """

        l2a_product_folder_name: Optional[str] = field(
            default=None,
            metadata={
                "name": "L2aProductFolderName",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        l2a_product_date: Optional[str] = field(
            default=None,
            metadata={
                "name": "L2aProductDate",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        l1_inputs: Optional[InputInformationL2AType] = field(
            default=None,
            metadata={
                "name": "L1Inputs",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        significance_level: Optional[float] = field(
            default=None,
            metadata={
                "name": "significanceLevel",
                "type": "Attribute",
            },
        )
