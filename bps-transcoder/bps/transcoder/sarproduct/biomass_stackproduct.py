# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The BIOMASS L1c Product Object
------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.common.io.translate_common import (
    translate_bool,
    translate_datetime,
    translate_interferometric_pair_list,
    translate_interferometric_pair_list_to_model,
)
from bps.transcoder.io import common_annotation_models_l1 as main_annotation_models
from bps.transcoder.io import main_annotation_models_l1c as l1c_annotations
from bps.transcoder.io.main_annotation_models_l1c.models import StaQualityParametersType
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.utils.production_model_utils import encode_product_name_id_value
from bps.transcoder.utils.time_conversions import (
    no_zulu_isoformat,
    pdt_to_compact_date,
    pdt_to_compact_string,
)


# Handle an invalid L1c product.
class InvalidBIOMASSStackProductError(RuntimeError):
    """Handle an invalid L1c product."""


# List of LUT layers that are expected to be always present in a L1c products
# (e.g. because they are the output of the coregistrator or because they should
# be provided in the L1a products).
REQUIRED_LUT_LAYERS = [
    "azimuthCoregistrationShifts",
    "azimuthOrbitCoregistrationShifts",
    "coregistrationShiftsQuality",
    "elevationAngle",
    "flatteningPhaseScreen",
    "height",
    "incidenceAngle",
    "latitude",
    "longitude",
    "rangeCoregistrationShifts",
    "rangeOrbitCoregistrationShifts",
    "terrainSlope",
    "waveNumbers",
]

# All the LUT layers possibly available in a L1c product. Not required LUT
# layers are present depending on the chosen configurations of the stack and L1
# processor.
LUT_LAYERS = [
    *REQUIRED_LUT_LAYERS,
    "gammaNought",
    "denoisingHH",
    "denoisingHV",
    "denoisingXX",
    "denoisingVH",
    "denoisingVV",
    "skpCalibrationPhaseScreen",
    "skpCalibrationPhaseScreenQuality",
    "sigmaNought",
]


@dataclass
class BIOMASSStackProductConfiguration:
    """Data structure for the BIOMASS Stack product configuration."""

    frame_number: int
    frame_status: str
    product_nodata_value: float
    product_baseline: int | None = None
    product_doi: str | None = None
    product_compression_method_abs: str | None = None
    product_compression_method_phase: str | None = None
    product_max_z_error_abs: float | None = None
    product_max_z_error_phase: float | None = None


@dataclass
class BIOMASSStackProcessingParameters:
    """Data struct for the BIOMASS Stack Processor parameters."""

    processor_version: str
    product_generation_time: PreciseDateTime | str
    polarizations_used: int
    polarization_combination_method: common.PolarisationCombinationMethodType
    primary_image_selection_method: common.PrimaryImageSelectionMethodType
    coregistration_method: common.CoregistrationMethodType
    height_model: common.HeightModelType
    rfi_degradation_estimation_flag: bool
    azimuth_spectral_filtering_flag: bool
    polarization_used_for_slow_ionosphere_removal: common.PolarisationType
    polarization_used_for_phase_plane_removal: common.PolarisationType
    calibration_primary_image_flag: bool
    slow_ionosphere_removal_flag: bool
    in_sar_calibration_flag: bool
    skp_phase_calibration_flag: bool
    skp_phase_correction_flag: bool
    skp_phase_correction_flattening_only_flag: bool
    skp_estimation_window_size: float
    skp_median_filter_flag: bool
    skp_median_filter_window_size: float
    azimuth_spectral_filtering_use_32bit_flag: bool
    slow_ionosphere_removal_multi_baseline_threshold: float
    slow_ionosphere_removal_use_32bit_flag: bool
    in_sar_calibration_use_32bit_flag: bool
    skp_phase_calibration_use_32bit_flag: bool

    @classmethod
    def from_l1c_main_annotation(
        cls, l1c_annotation: l1c_annotations.MainAnnotation
    ) -> BIOMASSStackProcessingParameters:
        """Unmarshall from a L1c main annotation object."""
        annot = l1c_annotation.sta_processing_parameters
        return cls(
            processor_version=annot.processor_version,
            product_generation_time=translate_datetime(annot.product_generation_time),
            polarizations_used=annot.polarisations_used,
            polarization_combination_method=annot.polarisation_combination_method,
            primary_image_selection_method=annot.primary_image_selection_method,
            coregistration_method=annot.coregistration_method,
            height_model=annot.height_model,
            rfi_degradation_estimation_flag=translate_bool(annot.rfi_degradation_estimation_flag),
            azimuth_spectral_filtering_flag=translate_bool(annot.azimuth_spectral_filtering_flag),
            polarization_used_for_slow_ionosphere_removal=annot.polarisation_used_for_slow_ionosphere_removal,
            polarization_used_for_phase_plane_removal=annot.polarisation_used_for_phase_plane_removal,
            calibration_primary_image_flag=translate_bool(annot.calibration_primary_image_flag),
            slow_ionosphere_removal_flag=translate_bool(annot.slow_ionosphere_removal_flag),
            in_sar_calibration_flag=translate_bool(annot.in_sarcalibration_flag),
            skp_phase_calibration_flag=translate_bool(annot.skp_phase_calibration_flag),
            skp_phase_correction_flag=translate_bool(annot.skp_phase_correction_flag),
            skp_phase_correction_flattening_only_flag=translate_bool(annot.skp_phase_correction_flattening_only_flag),
            skp_estimation_window_size=annot.skp_estimation_window_size.value,
            skp_median_filter_flag=translate_bool(annot.skp_median_filter_flag),
            skp_median_filter_window_size=annot.skp_median_filter_window_size.value,
            slow_ionosphere_removal_multi_baseline_threshold=annot.slow_ionosphere_removal_multi_baseline_threshold,
            azimuth_spectral_filtering_use_32bit_flag=translate_bool(annot.azimuth_spectral_filtering_use32_bit_flag),
            slow_ionosphere_removal_use_32bit_flag=translate_bool(annot.slow_ionosphere_removal_use32_bit_flag),
            in_sar_calibration_use_32bit_flag=translate_bool(annot.in_sarcalibration_use32_bit_flag),
            skp_phase_calibration_use_32bit_flag=translate_bool(annot.skp_phase_calibration_use32_bit_flag),
        )

    def to_l1c_annotation(self) -> l1c_annotations.StaProcessingParametersType:
        """Marshall to related L1c annotation."""
        return l1c_annotations.StaProcessingParametersType(
            processor_version=self.processor_version,
            product_generation_time=no_zulu_isoformat(self.product_generation_time, timespec="microseconds"),
            polarisations_used=self.polarizations_used,
            polarisation_combination_method=self.polarization_combination_method,
            polarisation_used_for_slow_ionosphere_removal=self.polarization_used_for_slow_ionosphere_removal,
            polarisation_used_for_phase_plane_removal=self.polarization_used_for_phase_plane_removal,
            primary_image_selection_method=self.primary_image_selection_method,
            coregistration_method=self.coregistration_method,
            calibration_primary_image_flag=self.calibration_primary_image_flag,
            height_model=self.height_model,
            rfi_degradation_estimation_flag=self.rfi_degradation_estimation_flag,
            azimuth_spectral_filtering_flag=self.azimuth_spectral_filtering_flag,
            slow_ionosphere_removal_flag=self.slow_ionosphere_removal_flag,
            in_sarcalibration_flag=self.in_sar_calibration_flag,
            skp_phase_calibration_flag=self.skp_phase_calibration_flag,
            skp_phase_correction_flag=self.skp_phase_correction_flag,
            skp_phase_correction_flattening_only_flag=self.skp_phase_correction_flattening_only_flag,
            skp_estimation_window_size=main_annotation_models.FloatWithUnit(
                value=float(self.skp_estimation_window_size),
                units="m",
            ),
            skp_median_filter_flag=self.skp_median_filter_flag,
            skp_median_filter_window_size=main_annotation_models.FloatWithUnit(
                value=float(self.skp_median_filter_window_size),
                units="m",
            ),
            slow_ionosphere_removal_multi_baseline_threshold=self.slow_ionosphere_removal_multi_baseline_threshold,
            azimuth_spectral_filtering_use32_bit_flag=self.azimuth_spectral_filtering_use_32bit_flag,
            slow_ionosphere_removal_use32_bit_flag=self.slow_ionosphere_removal_use_32bit_flag,
            in_sarcalibration_use32_bit_flag=self.in_sar_calibration_use_32bit_flag,
            skp_phase_calibration_use32_bit_flag=self.skp_phase_calibration_use_32bit_flag,
        )


@dataclass
class BIOMASSStackCoregistrationParameters:
    """Data struct for the BIOMASS Coregistration Processor parameters."""

    datum: common.DatumType
    primary_image: str
    secondary_image: str
    primary_image_selection_information: common.PrimaryImageSelectionInformationType
    average_azimuth_coregistration_shift: float
    average_range_coregistration_shift: float
    normal_baseline: float
    range_spectral_filtering_flag: bool
    polarization_used: common.PolarisationType

    @classmethod
    def from_l1c_main_annotation(
        cls, l1c_annotation: l1c_annotations.MainAnnotation
    ) -> BIOMASSStackCoregistrationParameters:
        """Unmarshall from L1c main annotation object."""
        annot = l1c_annotation.sta_coregistration_parameters
        return cls(
            datum=annot.datum,
            primary_image=annot.primary_image,
            secondary_image=annot.secondary_image,
            primary_image_selection_information=annot.primary_image_selection_information,
            average_azimuth_coregistration_shift=annot.average_azimuth_coregistration_shift.value,
            average_range_coregistration_shift=annot.average_range_coregistration_shift.value,
            normal_baseline=annot.normal_baseline.value,
            range_spectral_filtering_flag=translate_bool(annot.range_spectral_filtering_flag),
            polarization_used=annot.polarisation_used,
        )

    def to_l1c_annotation(self) -> l1c_annotations.StaCoregistrationParametersType:
        """Marshall to related L1c annotation."""
        return l1c_annotations.StaCoregistrationParametersType(
            datum=self.datum,
            primary_image=self.primary_image,
            secondary_image=self.secondary_image,
            primary_image_selection_information=self.primary_image_selection_information,
            average_azimuth_coregistration_shift=main_annotation_models.FloatWithUnit(
                value=float(self.average_azimuth_coregistration_shift),
                units="m",
            ),
            average_range_coregistration_shift=main_annotation_models.FloatWithUnit(
                value=float(self.average_range_coregistration_shift),
                units="m",
            ),
            normal_baseline=main_annotation_models.FloatWithUnit(
                value=float(self.normal_baseline),
                units="m",
            ),
            range_spectral_filtering_flag=self.range_spectral_filtering_flag,
            polarisation_used=self.polarization_used,
        )


@dataclass
class BIOMASSStackInSARParameters:
    """Data struct for the BIOMASS Stack InSAR parameters."""

    calibration_primary_image: str
    azimuth_common_bandwidth: float
    azimuth_central_frequency: float
    slow_ionosphere_azimuth_phase_screen: float
    slow_ionosphere_range_phase_screen: float
    slow_ionosphere_quality: float
    slow_ionosphere_removal_interferometric_pairs: list[tuple[int, int]]
    azimuth_phase_slope: float
    range_phase_slope: float
    baseline_ordering_index: int
    skp_calibration_phase_screen_mean: float
    skp_calibration_phase_screen_std: float
    skp_calibration_phase_screen_var: float
    skp_calibration_phase_screen_mad: float

    @classmethod
    def from_l1c_main_annotation(cls, l1c_annotation: l1c_annotations.MainAnnotation) -> BIOMASSStackInSARParameters:
        """Unmarshall from L1c main annotation object."""
        annot = l1c_annotation.sta_in_sarparameters
        return cls(
            calibration_primary_image=annot.calibration_primary_image,
            azimuth_common_bandwidth=annot.azimuth_common_bandwidth.value,
            azimuth_central_frequency=annot.azimuth_central_frequency.value,
            slow_ionosphere_azimuth_phase_screen=annot.slow_ionosphere_azimuth_phase_screen.value,
            slow_ionosphere_range_phase_screen=annot.slow_ionosphere_range_phase_screen.value,
            slow_ionosphere_quality=annot.slow_ionosphere_quality,
            slow_ionosphere_removal_interferometric_pairs=translate_interferometric_pair_list(
                annot.slow_ionosphere_removal_interferometric_pairs
            ),
            azimuth_phase_slope=annot.azimuth_phase_slope.value,
            range_phase_slope=annot.range_phase_slope.value,
            baseline_ordering_index=annot.baseline_ordering_index,
            skp_calibration_phase_screen_mean=annot.skp_calibration_phase_screen_mean.value,
            skp_calibration_phase_screen_std=annot.skp_calibration_phase_screen_std,
            skp_calibration_phase_screen_var=annot.skp_calibration_phase_screen_var,
            skp_calibration_phase_screen_mad=annot.skp_calibration_phase_screen_mad.value,
        )

    def to_l1c_annotation(self) -> l1c_annotations.StaInSarparametersType:
        """Marshall to related L1c annotation."""
        return l1c_annotations.StaInSarparametersType(
            calibration_primary_image=self.calibration_primary_image,
            azimuth_common_bandwidth=main_annotation_models.FloatWithUnit(
                value=float(self.azimuth_common_bandwidth),
                units="Hz",
            ),
            azimuth_central_frequency=main_annotation_models.FloatWithUnit(
                value=float(self.azimuth_central_frequency),
                units="Hz",
            ),
            slow_ionosphere_azimuth_phase_screen=main_annotation_models.FloatWithUnit(
                value=float(self.slow_ionosphere_azimuth_phase_screen),
                units="rad/s",
            ),
            slow_ionosphere_range_phase_screen=main_annotation_models.FloatWithUnit(
                value=float(self.slow_ionosphere_range_phase_screen),
                units="rad/s",
            ),
            slow_ionosphere_quality=float(self.slow_ionosphere_quality),
            slow_ionosphere_removal_interferometric_pairs=translate_interferometric_pair_list_to_model(
                self.slow_ionosphere_removal_interferometric_pairs
            ),
            azimuth_phase_slope=main_annotation_models.FloatWithUnit(
                value=float(self.azimuth_phase_slope),
                units="rad/s",
            ),
            range_phase_slope=main_annotation_models.FloatWithUnit(
                value=float(self.range_phase_slope),
                units="rad/s",
            ),
            baseline_ordering_index=int(self.baseline_ordering_index),
            skp_calibration_phase_screen_mean=main_annotation_models.FloatWithUnit(
                value=float(self.skp_calibration_phase_screen_mean),
                units="rad",
            ),
            skp_calibration_phase_screen_std=float(self.skp_calibration_phase_screen_std),
            skp_calibration_phase_screen_var=float(self.skp_calibration_phase_screen_var),
            skp_calibration_phase_screen_mad=main_annotation_models.FloatWithUnit(
                value=float(self.skp_calibration_phase_screen_mad),
                units="rad",
            ),
        )


@dataclass
class BIOMASSStackQualityParameters:
    """Data struct for the BIOMASS Stack Quality parameters."""

    polarization: common.PolarisationType
    invalid_l1a_data_samples: float
    rfi_decorrelation: float
    rfi_decorrelation_threshold: float
    faraday_decorrelation: float
    faraday_decorrelation_threshold: float
    invalid_residual_shifts_ratio: float
    residual_shifts_quality_threshold: float
    invalid_skp_calibration_phase_screen_ratio: float
    skp_calibration_phase_screen_quality_threshold: float
    skp_decomposition_index: int

    @classmethod
    def from_l1c_main_annotation(cls, annot: StaQualityParametersType) -> BIOMASSStackQualityParameters:
        """Unmarshall from L1c main annotation object."""
        return cls(
            polarization=annot.polarisation,
            invalid_l1a_data_samples=annot.invalid_l1a_data_samples,
            rfi_decorrelation=annot.rfi_decorrelation,
            rfi_decorrelation_threshold=annot.rfi_decorrelation_threshold,
            faraday_decorrelation=annot.faraday_decorrelation,
            faraday_decorrelation_threshold=annot.faraday_decorrelation_threshold,
            invalid_residual_shifts_ratio=annot.invalid_residual_shifts_ratio,
            residual_shifts_quality_threshold=annot.residual_shifts_quality_threshold,
            invalid_skp_calibration_phase_screen_ratio=annot.invalid_skp_calibration_phase_screen_ratio,
            skp_calibration_phase_screen_quality_threshold=annot.skp_calibration_phase_screen_quality_threshold,
            skp_decomposition_index=annot.skp_decomposition_index,
        )

    def to_l1c_annotation(self) -> l1c_annotations.StaQualityParametersType:
        """Marshall to related L1c annotation."""
        return l1c_annotations.StaQualityParametersType(
            polarisation=self.polarization,
            invalid_l1a_data_samples=float(self.invalid_l1a_data_samples),
            rfi_decorrelation=float(self.rfi_decorrelation),
            rfi_decorrelation_threshold=float(self.rfi_decorrelation_threshold),
            faraday_decorrelation=float(self.faraday_decorrelation),
            faraday_decorrelation_threshold=float(self.faraday_decorrelation_threshold),
            invalid_residual_shifts_ratio=float(self.invalid_residual_shifts_ratio),
            residual_shifts_quality_threshold=float(self.residual_shifts_quality_threshold),
            invalid_skp_calibration_phase_screen_ratio=float(self.invalid_skp_calibration_phase_screen_ratio),
            skp_calibration_phase_screen_quality_threshold=float(self.skp_calibration_phase_screen_quality_threshold),
            skp_decomposition_index=int(self.skp_decomposition_index),
        )


@dataclass
class BIOMASSStackQuality:
    """Data struct for the BIOMASS Stack quality parameters."""

    overall_product_quality_index: int
    sta_quality_parameters_list: list[BIOMASSStackQualityParameters] = field(default_factory=list)

    @classmethod
    def from_l1c_main_annotation(cls, l1c_annotation: l1c_annotations.MainAnnotation) -> BIOMASSStackQuality:
        """Unmarshall from L1c main annotation object."""
        annot = l1c_annotation.sta_quality
        return cls(
            overall_product_quality_index=annot.overall_product_quality_index,
            sta_quality_parameters_list=[
                BIOMASSStackQualityParameters.from_l1c_main_annotation(n)
                for n in annot.sta_quality_parameters_list.sta_quality_parameters
            ],
        )

    def to_l1c_annotation(self) -> l1c_annotations.StaQualityType:
        """Marshall to related L1c annotation."""
        return l1c_annotations.StaQualityType(
            overall_product_quality_index=self.overall_product_quality_index,
            sta_quality_parameters_list=l1c_annotations.StaQualityParametersListType(
                sta_quality_parameters=[sta_q.to_l1c_annotation() for sta_q in self.sta_quality_parameters_list],
                count=len(self.sta_quality_parameters_list),
            ),
        )


class BIOMASSStackProduct(SARProduct):
    """Object that stores a BIOMASS Stack Product (L1c)."""

    def __init__(
        self,
        *,
        product: SARProduct | None = None,
        product_primary: SARProduct | None = None,
        stack_processing_parameters: BIOMASSStackProcessingParameters | None = None,
        stack_coregistration_parameters: (BIOMASSStackCoregistrationParameters | None) = None,
        stack_in_sarparameters: BIOMASSStackInSARParameters | None = None,
        stack_quality: BIOMASSStackQuality | None = None,
        is_monitoring: bool | None = False,
        configuration: BIOMASSStackProductConfiguration | None = None,
        mission_phase_id: int | None = None,
        datatake_id: int | None = None,
        orbit_number: int | None = None,
        global_coverage_id: int | None = None,
        repeat_cycle_id: int | None = None,
        major_cycle_id: int | None = None,
        track_number: int | None = None,
        platform_heading: float | None = None,
        first_sample_sr_time: float | None = None,
        first_line_az_time: float | None = None,
        rg_time_interval: float | None = None,
        az_time_interval: float | None = None,
        number_of_samples: int | None = None,
        number_of_lines: int | None = None,
        stack_footprint: list[list[float]] | None = None,
        stack_id: str | None = None,
    ):
        """
        Instantiate an L1c product.

        Parameters
        ----------
        product: Optional[SARProduct] = None
            The target product encoded as a general SAR product.

        product_primary: Optional[SARProduct] = None
            The primary product encoded as a general SAR product.

        stack_processing_parameters: Optional[BIOMASSStackProcessingParameters] = None
            The BIOMASS Stack Processor common parameters.

        stack_coregistration_parameters: Optional[BIOMASSStackCoregistrationParameters] = None
            The BIOMASS Coregistration Processor parameters.

        stack_in_sarparameters: Optional[BIOMASSStackInSARParameters] = None
            The BIOMASS stack InSAR parameters.

        stack_quality: Optional[BIOMASSStackQuality] = None
            The BIOMASS Stack quality parameters.

        is_monitoring: Optional[bool] = False
            As to whether the current product is a monitoring or standard one.

        configuration: Optional[BIOMASSStackProductConfiguration] = None
            THe BIOMASS Stack Product configuration parameters.

        mission_phase_id: Optional[int] = None,
            The ID of the mission phase.

        datatake_id: Optional[int] = None
            The ID of when/how the data was taken.

        orbit_number: Optional[int] = None
            The orbit number.

        global_coverage_id: Optional[int] = None
            The global coverage ID.

        major_cycle_id: Optional[int] = None
            The ID of the satellite major cycle.

        repeat_cycle_id: Optional[int] = None
            The ID of the repeat cycle.

        track_number: Optional[int] = None
            The track number.

        platform_heading: Optional[float] = None
            The platform heading.

        first_sample_sr_time: Optional[float] = None [s]
            Time of the first slant-range sample.

        first_line_az_time: Optional[float] = None [s]
            Time of the first azimuth line.

        rg_time_interval: Optional[float] = None [s]
            Time interval in slant-range component.

        az_time_interval: Optional[float] = None
            Time interval in azimuth component.

        number_of_samples: Optional[int] = None
            Number of slant-range samples.

        number_of_lines: Optional[int] = None
            Number of azimuth lines.

        stack_footprint: list[list[float]] | None = None,
            Optionally, a stack footprint.

        stack_id: str | None = None
            Optionally, the stack unique identifier.

        """
        SARProduct.__init__(self)
        # Possibly store some attributes from the input object.
        if product is not None:
            self.__dict__.update(product.__dict__)
        if product_primary is not None:
            self.product_primary = product_primary
            self.product_primary.type = "STA"

        # NOTE: These ought to be after incorporating the the product object.
        self.type = "STA"
        self.is_monitoring = is_monitoring

        # Store the products of the calibration stack.
        self.stack_processing_parameters = stack_processing_parameters
        self.stack_coregistration_parameters = stack_coregistration_parameters
        self.stack_in_sarparameters = stack_in_sarparameters
        self.stack_quality = stack_quality
        self.mission_phase_id = mission_phase_id
        self.datatake_id = datatake_id
        self.orbit_number = orbit_number
        self.global_coverage_id = global_coverage_id
        self.repeat_cycle_id = repeat_cycle_id
        self.major_cycle_id = major_cycle_id
        self.track_number = track_number
        self.platform_heading = platform_heading
        self.first_sample_sr_time = first_sample_sr_time
        self.first_line_az_time = first_line_az_time
        self.rg_time_interval = rg_time_interval
        self.az_time_interval = az_time_interval
        self.number_of_samples = number_of_samples
        self.number_of_lines = number_of_lines
        self.stack_footprint = stack_footprint
        self.stack_id = stack_id
        self.frame_status = None
        self.frame_number = None
        self.baseline_id = None
        self.product_doi = None
        self.product_nodata_value = None
        self.product_compression_method_abs = None
        self.product_compression_method_phase = None
        self.product_max_z_error_abs = None
        self.product_max_z_error_phase = None

        if configuration is not None:
            self.frame_number = configuration.frame_number
            self.frame_status = configuration.frame_status
            self.baseline_id = configuration.product_baseline
            self.product_doi = configuration.product_doi
            self.product_nodata_value = configuration.product_nodata_value
            self.product_compression_method_abs = configuration.product_compression_method_abs
            self.product_compression_method_phase = configuration.product_compression_method_phase
            self.product_max_z_error_abs = configuration.product_max_z_error_abs
            self.product_max_z_error_phase = configuration.product_max_z_error_phase

        self.__set_default_missing_attributes()
        self.__set_product_name()

    @property
    def is_coreg_primary(self) -> bool:
        """Whether the current product is the coregistration primary of the stack."""
        return (
            self.stack_coregistration_parameters.primary_image == self.stack_coregistration_parameters.secondary_image
        )

    def __set_default_missing_attributes(self):
        """Set default values to the (most important) attributes."""
        self.mission_phase_id = _value_or(self.mission_phase_id, "INTERFEROMETRIC")
        self.datatake_id = _value_or(self.datatake_id, 1)
        self.orbit_number = _value_or(self.orbit_number, 1)
        self.global_coverage_id = _value_or(self.global_coverage_id, 1)
        self.major_cycle_id = _value_or(self.major_cycle_id, 1)
        self.repeat_cycle_id = _value_or(self.repeat_cycle_id, 1)
        self.track_number = _value_or(self.track_number, 1)
        self.frame_number = _value_or(self.frame_number, 1)
        self.frame_status = _value_or(self.frame_status, "NOMINAL")
        self.baseline_id = _value_or(self.baseline_id, 1)
        self.product_nodata_value = _value_or(self.product_nodata_value, -9999.0)
        self.product_compression_method_abs = _value_or(self.product_compression_method_abs, "DEFAULT")
        self.product_compression_method_phase = _value_or(self.product_compression_method_phase, "DEFAULT")
        self.platform_heading = _value_or(self.platform_heading, 0.0)
        self.number_of_samples = _value_or(self.number_of_samples, 1)
        self.number_of_lines = _value_or(self.number_of_lines, 1)
        self.stack_footprint = _value_or(self.stack_footprint, [[0.0, 0.0]] * 4)

    def __set_product_name(self):
        """Initialize the product name according to the naming convention."""
        self.creation_date = PreciseDateTime.now()
        self.name = "_".join(
            [
                "BIO",
                self.swath_list[0],
                self.type,
                "_1S" if not self.is_monitoring else "_1M",
                pdt_to_compact_string(self.start_time),
                pdt_to_compact_string(self.stop_time),
                self.mission_phase_id[0],
                "G{}".format(encode_product_name_id_value(self.global_coverage_id, npad=2)),
                "M{}".format(encode_product_name_id_value(self.major_cycle_id, npad=2)),
                "C{}".format(encode_product_name_id_value(self.repeat_cycle_id, npad=2)),
                "T{}".format("___" if self.mission_phase_id[0] == "C" else f"{self.track_number:03d}"),
                "F{}".format("___" if self.frame_number == 0 else f"{self.frame_number:03d}"),
                f"{self.baseline_id:02d}",
                pdt_to_compact_date(self.creation_date),
            ]
        )
        return True


def _value_or(opt: Any | None, default_value: Any) -> Any:
    """Access the optional value if not None, otherwise return default."""
    return opt if opt is not None else default_value
