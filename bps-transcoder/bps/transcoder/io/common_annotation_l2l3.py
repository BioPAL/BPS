# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common annotations l2
---------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common


class BpsFnfType(Enum):
    FNF = "FNF"
    CFM = "CFM"


@dataclass
class PixelRepresentationChoiceType:
    fd: common.PixelRepresentationType | None = None
    cfm: common.PixelRepresentationType | None = None
    probability_of_change: common.PixelRepresentationType | None = None
    fd_heat_map: common.PixelRepresentationType | None = None
    fh: common.PixelRepresentationType | None = None
    quality: common.PixelRepresentationType | None = None
    fh_heat_map: common.PixelRepresentationType | None = None
    gn: common.PixelRepresentationType | None = None
    agb: common.PixelRepresentationType | None = None
    agb_standard_deviation: common.PixelRepresentationType | None = None
    agb_heat_map: common.PixelRepresentationType | None = None
    bps_fnf: common.PixelRepresentationType | None = None
    acquisition_id_image: common.PixelRepresentationType | None = None


@dataclass
class PixelTypeChoiceType:
    float_pixel_type: common.PixelTypeType | None = None
    int_pixel_type: common.PixelTypeType | None = None


@dataclass
class NoDataValueChoiceType:
    float_no_data_value: float | None = None
    int_no_data_value: int | None = None


@dataclass
class InputInformationL2BL3ListType:
    @dataclass
    class L2AInputs:
        l2a_product_folder_name: str
        l2a_product_date: str
        l1_inputs: BIOMASSL2aMainADSInputInformation
        significance_level: float | None = None

    l2a_inputs: list[InputInformationL2BL3ListType.L2AInputs]
    count: int


@dataclass
class PercentPixelsType:
    percentage: float
    pixels: int


@dataclass
class CalAbfilteredCoverageType:
    after_global_filtering: PercentPixelsType
    after_agbvalue_filtering: PercentPixelsType
    after_agbstd_filtering: PercentPixelsType
    after_agbrelative_std_filtering: PercentPixelsType
    after_lcmclass_filtering: PercentPixelsType


@dataclass
class CalAbcoverageType:
    original_coverage: PercentPixelsType
    filtered_coverage: CalAbfilteredCoverageType
    id: str


@dataclass
class GnfilteredCoverageType:
    after_global_filtering: PercentPixelsType
    after_sigma_filtering: PercentPixelsType
    after_angle_filtering: PercentPixelsType


@dataclass
class GncoverageType:
    original_coverage: PercentPixelsType
    filtered_coverage: GnfilteredCoverageType
    acquisition_id: str


@dataclass
class GncoverageListType:
    gn: list[GncoverageType]
    id: str


@dataclass
class StaQualityParametersType:
    invalid_l1a_data_samples: float
    rfi_decorrelation: float
    rfi_decorrelation_threshold: float
    faraday_decorrelation: float
    faraday_decorrelation_threshold: float
    invalid_residual_shifts_fraction: float
    residual_shifts_quality_threshold: float
    invalid_ground_phases_screen_estimates_fraction: float
    ground_phases_screen_quality_threshold: float
    skp_decomposition_index: int
    polarisation: common.PolarisationType


@dataclass
class StaQualityType:
    overall_product_quality_index: int
    sta_quality_parameters_list: list[StaQualityParametersType]


@dataclass
class AcquisitionType:
    folder_name: str
    sta_quality: StaQualityType
    reference_image: bool
    average_wavenumber: float | None = None


AcquisitionListType = list[AcquisitionType]


class CalibrationScreenType(Enum):
    NONE = "none"
    GEOMETRY = "geometry"
    SKP = "skp"


class SubsettingRuleType(Enum):
    GEOMETRY = "geometry"
    MAINTAIN_ALL = "maintain all"


@dataclass
class GeneralConfigurationParametersType:
    apply_calibration_screen: CalibrationScreenType
    forest_coverage_threshold: float
    forest_mask_interpolation_threshold: float
    subsetting_rule: SubsettingRuleType
    polarisation_combination_method: common.PolarisationCombinationMethodType | None


@dataclass
class MinMaxNumType:
    min: float
    max: float
    num: int


@dataclass
class SelectedReferenceImageType:
    value: int
    acquisition_id: str


@dataclass
class BIOMASSL2aMainADSproduct:
    mission: common.MissionType
    tile_id_list: list[str]
    basin_id_list: list[str]
    product_type: common.ProductType
    start_time: PreciseDateTime
    stop_time: PreciseDateTime
    radar_carrier_frequency: float
    mission_phase_id: common.MissionPhaseIdtype
    sensor_mode: common.SensorModeType
    global_coverage_id: int
    swath: common.SwathType
    major_cycle_id: int
    absolute_orbit_number: list[int]
    relative_orbit_number: int
    orbit_pass: common.OrbitPassType
    datatake_id: list[int]
    frame: int
    platform_heading: float
    baseline: int
    forest_coverage_percentage: float
    selected_reference_image: SelectedReferenceImageType | None = None


@dataclass
class BIOMASSL2aMainADSRasterImage:
    footprint: list[float]
    first_latitude_value: float
    first_longitude_value: float
    latitude_spacing: float
    longitude_spacing: float
    number_of_samples: int
    number_of_lines: int
    projection: common.ProjectionType
    datum: common.DatumType
    pixel_representation: PixelRepresentationChoiceType
    pixel_type: PixelTypeChoiceType
    no_data_value: NoDataValueChoiceType


@dataclass
class BIOMASSL2aMainADSInputInformation:
    product_type: common.ProductType
    overall_products_quality_index: int
    nominal_stack: str
    polarisation_list: list[common.PolarisationType]
    projection: common.ProjectionType
    footprint: list[float]
    vertical_wavenumbers: common.MinMaxTypeWithUnit
    height_of_ambiguity: common.MinMaxTypeWithUnit
    acquisition_list: AcquisitionListType


@dataclass
class ProductL2bL3Type:
    mission: common.MissionType
    tile_id: str
    basin_id: list[str]
    product_type: common.ProductType
    start_time: PreciseDateTime
    stop_time: PreciseDateTime
    radar_carrier_frequency: float
    mission_phase_id: common.MissionPhaseIdtype
    sensor_mode: common.SensorModeType
    global_coverage_id: int
    contributing_tiles: list[str]
    cal_ab_coverage_per_tile: list[CalAbcoverageType]
    gncoverage_per_tile: list[GncoverageListType]
