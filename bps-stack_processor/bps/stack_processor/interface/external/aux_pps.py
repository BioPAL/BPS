# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX-PPS Configuration Object and Utils
--------------------------------------
"""

import importlib.resources
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path

import bps.stack_processor
import numpy as np
from arepytools.io.metadata import EPolarization
from bps.common import bps_logger
from bps.common.io import common
from bps.common.io.parsing import ParsingError, validate
from bps.stack_cal_processor.configuration import (
    AZF_NAME,
    CAL_NAME,
    IOB_NAME,
    PPR_NAME,
    SKP_NAME,
    UNITS_CAL_CONFIG,
    BaselineMethodType,
)
from bps.stack_cal_processor.core.filtering import ConvolutionWindowType
from bps.stack_pre_processor.configuration import PrimaryImageSelectionConf

# Just a conversion from field name to module name.
_FIELD_TO_MODULE_NAME = {
    "azimuth_spectral_filtering": AZF_NAME,
    "phase_plane_removal": PPR_NAME,
    "slow_ionosphere_removal": IOB_NAME,
    "in_sar_calibration": CAL_NAME,
    "skp_phase_calibration": SKP_NAME,
}

# Units of AUX-PPS values that has units.
_UNITS_AUX_PPS_VALUES = {
    "azimuth_max_shift": " [px]",
    "azimuth_block_size": " [px]",
    "azimuth_min_overlap": " [px]",
    "range_max_shift": " [px]",
    "range_block_size": " [px]",
    "range_min_overlap": " [px]",
    "latitude_threshold": " [rad]",
    "sublook_window_azimuth_size": " [px]",
    "sublook_window_range_size": " [px]",
    "low_pass_filter_order": " [px]",
    "low_pass_filter_std_dev": " [px]",
    "fft2_peak_window_size": " [px]",
    "phase_max_z_error": " [rad]",
    "ql_range_decimation_factor": " [px]",
    "ql_range_averaging_factor": " [px]",
    "ql_azimuth_decimation_factor": " [px]",
    "ql_azimuth_averaging_factor": " [px]",
    **UNITS_CAL_CONFIG,
}


class AuxPPSInvalidSchema(ValueError):
    """Raised when the AUX-PPS has an invalid XSD schema."""


class AuxPPSNotSupportedError(ValueError):
    """Raised when the AUX-PPS has valid but not yet supported parameters."""


class AuxPPSInvalidValue(ValueError):
    """When invalid parameters are passed to the AUX-PPS."""


@dataclass
class GeneralConf:
    """The AUX-PPS general configuration."""

    polarization_combination_method: common.PolarisationCombinationMethodType
    outer_parallelization_max_cores: int
    allow_duplicate_images_flag: bool
    flattening_phase_bias_compensation_flag: bool


@dataclass
class CoregistrationConf:
    """Coregistration configuration."""

    coregistration_method: common.CoregistrationMethodType
    range_spectral_filtering_flag: bool
    residual_shift_quality_threshold: float
    polarization_used: EPolarization
    block_quality_threshold: float
    fitting_quality_threshold: float
    min_valid_blocks: int
    azimuth_max_shift: int
    azimuth_block_size: int
    azimuth_min_overlap: int
    range_max_shift: int
    range_block_size: int
    range_min_overlap: int
    model_based_fit_flag: bool
    low_pass_filter_type: str
    low_pass_filter_order: int
    low_pass_filter_std_dev: float
    export_debug_products_flag: bool
    coregistration_execution_policy: common.CoregistrationExecutionPolicyType


@dataclass
class RfiDegradationEstimationConf:
    """RFI degradation model configuration."""

    rfi_degradation_estimation_flag: bool


@dataclass
class AzimuthSpectralFilteringConf:
    """Configuration for the Azimuth Spectral Filtering (AZF)."""

    azimuth_spectral_filtering_flag: bool
    use_primary_weighting_window_flag: bool
    spectral_weighting_window: ConvolutionWindowType
    spectral_weighting_window_parameter: float
    use_32bit_flag: bool


@dataclass
class SlowIonosphereRemovalConf:
    """Configuration for the Background Ionosphere Removal (IOB)."""

    slow_ionosphere_removal_flag: bool
    primary_image_flag: bool
    polarization_used: EPolarization
    compensate_l1_iono_phase_screen_flag: bool
    range_look_bandwidth: float  # [Hz].
    range_look_frequency: float  # [Hz].
    phase_unwrapping_flag: bool
    latitude_threshold: float  # [rad].
    baseline_method: BaselineMethodType
    unweighted_multi_baseline_estimation: bool
    multi_baseline_critical_baseline_threshold: float
    sublook_window_azimuth_size: int  # [px].
    sublook_window_range_size: int  # [px]
    slow_ionosphere_quality_threshold: float
    min_coherence_threshold: float
    max_lh_phase_delta: float  # [rad].
    min_usable_pixel_ratio: float
    use_32bit_flag: bool


@dataclass
class InSarCalibrationConf:
    """Configuration of the InSAR calibration module."""

    in_sar_calibration_flag: bool
    polarization_used: EPolarization
    fft2_zero_padding_upsampling_factor: float
    fft2_peak_window_size: int
    use_32bit_flag: bool


@dataclass
class SkpPhaseCalibrationConf:
    """Configuration for the Sum-of-Kronecker-Product Calibration (SKP)."""

    class SkpPhaseCorrectionType(Enum):
        """Enumeration for the possible correction executed by SKP."""

        NONE = "NONE"
        FLATTENING_PHASE_SCREEN = "FLATTENING PHASE SCREEN"
        GROUND_PHASE_SCREEN = "GROUND PHASE SCREEN"

    skp_phase_estimation_flag: bool
    phase_correction: SkpPhaseCorrectionType
    estimation_window_size: float  # [m].
    skp_calibration_phase_screen_quality_threshold: float
    overall_product_quality_threshold: float
    median_filter_flag: bool
    median_filter_window_size: float  # [m].
    exclude_mpmb_polarization_cross_covariance_flag: bool
    use_32bit_flag: bool


@dataclass
class L1cProductExportConf:
    """Configuration of the L1c exporter."""

    class CompressionMethodType(Enum):
        """Enumeration for an image pixel type."""

        NONE = "NONE"
        DEFLATE = "DEFLATE"
        ZSTD = "ZSTD"
        LERC = "LERC"
        LERC_DEFLATE = "LERC_DEFLATE"
        LERC_ZSTD = "LERC_ZSTD"

    l1_product_doi: str
    pixel_representation: common.PixelRepresentationType
    pixel_quantity: common.PixelQuantityType
    abs_compression_method: CompressionMethodType | None
    abs_max_z_error: float
    phase_compression_method: CompressionMethodType | None
    phase_max_z_error: float
    no_pixel_value: float
    ql_range_decimation_factor: int
    ql_range_averaging_factor: int
    ql_azimuth_decimation_factor: int
    ql_azimuth_averaging_factor: int
    ql_absolute_scaling_factor: float


@dataclass
class AuxiliaryStaprocessingParameters:
    """The AUX-PPS class."""

    product_id: str
    general: GeneralConf
    primary_image_selection: PrimaryImageSelectionConf
    coregistration: CoregistrationConf
    rfi_degradation_estimation: RfiDegradationEstimationConf
    azimuth_spectral_filtering: AzimuthSpectralFilteringConf
    in_sar_calibration: InSarCalibrationConf
    slow_ionosphere_removal: SlowIonosphereRemovalConf
    skp_phase_calibration: SkpPhaseCalibrationConf
    l1c_product_export: L1cProductExportConf


def log_aux_pps_summary(aux_pps: AuxiliaryStaprocessingParameters):
    """Log a summary of the most relevant AUX-PPS parameters."""
    # Print the AUX-PPS configuration.
    bps_logger.info("Essential AUX-PPS summary:")

    # Flatten all out to a dictionary.
    aux_pps_dict = {
        param_name: param_value.__dict__
        for param_name, param_value in aux_pps.__dict__.items()
        if is_dataclass(param_value)
    }

    # General.
    bps_logger.info("  General")
    log_params(
        aux_pps_dict["general"],
        log_only=[
            "flattening_phase_bias_compensation_flag",
            "polarization_combination_method",
        ],
        indent=2,
    )

    # Coregistrator.
    bps_logger.info("  Coregistration")
    log_params(
        {
            **aux_pps_dict["primary_image_selection"],
            **aux_pps_dict["rfi_degradation_estimation"],
            **aux_pps_dict["coregistration"],
        },
        indent=2,
    )

    # Calibration.
    enabled_modules = []
    if aux_pps.azimuth_spectral_filtering.azimuth_spectral_filtering_flag:
        enabled_modules.append("azimuth_spectral_filtering")
    if aux_pps.slow_ionosphere_removal.slow_ionosphere_removal_flag:
        enabled_modules.append("slow_ionosphere_removal")
    if aux_pps.in_sar_calibration.in_sar_calibration_flag:
        enabled_modules.append("in_sar_calibration")
    if aux_pps.skp_phase_calibration.skp_phase_estimation_flag:
        enabled_modules.append("skp_phase_calibration")

    for enabled_module in enabled_modules:
        bps_logger.info("  %s", _FIELD_TO_MODULE_NAME[enabled_module])
        log_params(aux_pps_dict[enabled_module], indent=2)

    # L1c export configuration.
    bps_logger.info("  L1c export configuration")
    log_params(aux_pps_dict["l1c_product_export"], indent=2)


def log_params(
    parameters: dict,
    *,
    log_only: list[str] | None = None,
    indent: int = 1,
):
    """Pretty print the parameters."""
    if log_only is None:
        log_only = list(parameters.keys())

    for param_name in log_only:
        param_value = parameters[param_name]
        bps_logger.info(
            "%s%s%s: %s",
            "  " * indent,
            param_name,
            _UNITS_AUX_PPS_VALUES.get(param_name, ""),
            param_value.value if isinstance(param_value, Enum) else param_value,
        )


def validate_aux_pps_xsd_schema(aux_pps_xml_path: Path):
    """
    Validate the AUX-PPS's XSD schema.

    Parameters
    ----------
    aux_pps_xml_file: Path
        Path to the AUX-PPS xml file.

    Raises
    ------
    AuxPPSInvalidSchema

    """
    main_folder = importlib.resources.files(bps.stack_processor)
    package_xsd_dir = Path(main_folder).joinpath("xsd")

    if not package_xsd_dir.is_dir():
        raise NotADirectoryError(f"broken package, expected an XSD folder {package_xsd_dir}")

    aux_pps_xsd_schema_path = package_xsd_dir / "biomass-xsd" / "bio-aux-pps.xsd"
    if not aux_pps_xsd_schema_path.exists():
        raise FileNotFoundError(f"broken package, expected AUX-PPS schema {aux_pps_xsd_schema_path}")

    try:
        validate(xml_file=aux_pps_xml_path, schema=aux_pps_xsd_schema_path)
    except ParsingError as error:
        raise AuxPPSInvalidSchema(error) from error


def validate_aux_pps_parameters(aux_pps: AuxiliaryStaprocessingParameters):
    """
    Validate the AUX-PPS input parameters.

    Parameters
    ----------
    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS structure.

    Raises
    ------
    AuxPPSInvalidValue

    """
    # Preliminary check on the general configurations.
    if aux_pps.general.outer_parallelization_max_cores < 0:
        raise AuxPPSInvalidValue("outerParallelizationMaxCores cannot be negative")

    # Preliminary check on the coregistration parameters.
    if not 0 <= aux_pps.coregistration.residual_shift_quality_threshold <= 1:
        raise AuxPPSInvalidValue("residualShiftQualityThreshold must be between 0 and 1")
    if not 0 <= aux_pps.coregistration.block_quality_threshold <= 1:
        raise AuxPPSInvalidValue("blockQualityThreshold must be between 0 and 1")
    if not 0 <= aux_pps.coregistration.fitting_quality_threshold <= 1:
        raise AuxPPSInvalidValue("fittingQualityThreshold must be between 0 and 1")
    if aux_pps.coregistration.min_valid_blocks > 100:
        raise AuxPPSInvalidValue("minValidBlocks must be at most 100")
    if aux_pps.coregistration.azimuth_max_shift <= 0:
        raise AuxPPSInvalidValue("azimuthMaxShift must be a positive integer")
    if aux_pps.coregistration.azimuth_block_size <= 0:
        raise AuxPPSInvalidValue("azimuthBlockSize must be a positive integer")
    if aux_pps.coregistration.azimuth_min_overlap < 0:
        raise AuxPPSInvalidValue("azimuthMinOverlap must be a non-negative integer")
    if aux_pps.coregistration.range_max_shift <= 0:
        raise AuxPPSInvalidValue("rangeMaxShift must be a positive integer")
    if aux_pps.coregistration.range_block_size <= 0:
        raise AuxPPSInvalidValue("rangeBlockSize must be a positive integer")
    if aux_pps.coregistration.range_min_overlap < 0:
        raise AuxPPSInvalidValue("rangeMinOverlap must be a non-negative integer")
    if aux_pps.coregistration.low_pass_filter_type not in {"Average", "Gaussian"}:
        raise AuxPPSInvalidValue("lowPassFilterTYpe must be either 'Average' or 'Gaussian'")
    if aux_pps.coregistration.low_pass_filter_order <= 0:
        raise AuxPPSInvalidValue("lowPassFilterOrder must be a positive integer")
    if aux_pps.coregistration.low_pass_filter_std_dev <= 0:
        raise AuxPPSInvalidValue("lowPassFilterStdDev must be positive")
    if aux_pps.coregistration.coregistration_execution_policy is not common.CoregistrationExecutionPolicyType.NOMINAL:
        raise AuxPPSNotSupportedError("Only Nominal execution is supported for the coregistration at the moment.")

    # Preliminary check on the AZF parameters.
    if aux_pps.azimuth_spectral_filtering.spectral_weighting_window_parameter <= 0:
        raise AuxPPSInvalidValue("spectralWeightingWindowParameter must be positive")

    # Preliminary check on the IOB parameters.
    if not 0 <= aux_pps.slow_ionosphere_removal.range_look_bandwidth <= 1:
        raise AuxPPSInvalidValue("rangeLookBandwidth (IOB) must be between 0 and 1")
    if not 0 <= aux_pps.slow_ionosphere_removal.range_look_frequency <= 0.5:
        raise AuxPPSInvalidValue("rangeLookFrequency (IOB) must be between 0 and 0.5")
    if not 0 <= aux_pps.slow_ionosphere_removal.latitude_threshold <= np.pi / 2:
        raise AuxPPSInvalidValue("latitudeThreshold (IOB) must be between 0 and 90 degrees")
    if not 0 <= aux_pps.slow_ionosphere_removal.slow_ionosphere_quality_threshold <= 1:
        raise AuxPPSInvalidValue("slowIonosphereQualityThreshold (IOB) must be between 0 and 1")
    if aux_pps.slow_ionosphere_removal.sublook_window_azimuth_size <= 0:
        raise AuxPPSInvalidValue("sublookWindowAzimuthSize (IOB) must be at least 1")
    if aux_pps.slow_ionosphere_removal.sublook_window_range_size <= 0:
        raise AuxPPSInvalidValue("sublookWindowRangeSize (IOB) must be at least 1")
    if aux_pps.slow_ionosphere_removal.multi_baseline_critical_baseline_threshold <= 0.0:
        raise AuxPPSInvalidValue("multiBaselineCriticalBaselineThreshold (IOB) must be positive")
    if not 0 <= aux_pps.slow_ionosphere_removal.min_usable_pixel_ratio <= 1:
        raise AuxPPSInvalidValue("minUsablePixelRatio (IOB) must be between 0 and 1")
    if not 0 <= aux_pps.slow_ionosphere_removal.min_coherence_threshold <= 1:
        raise AuxPPSInvalidValue("minCoherenceThreshold (IOB) must be between 0 and 1")

    # Preliminary check on the PPR.
    if aux_pps.in_sar_calibration.fft2_peak_window_size % 2 == 0:
        raise AuxPPSInvalidValue("fft2PeakWindowSize (PPR) must be an odd integer")

    # Preliminary check on the SKP.
    if aux_pps.skp_phase_calibration.estimation_window_size <= 0.0:
        raise AuxPPSInvalidValue("estimationWindowSize (SKP) must be positive")
    if aux_pps.skp_phase_calibration.median_filter_window_size <= 0.0:
        raise AuxPPSInvalidValue("medianFilterWindowSize (SKP) must be positive")
    if not 0 <= aux_pps.skp_phase_calibration.skp_calibration_phase_screen_quality_threshold <= 1:
        raise AuxPPSInvalidValue("skpCalibrationPhaseScreenQualityThreshold (SKP) must be between 0 and 1")
    if not 0 <= aux_pps.skp_phase_calibration.overall_product_quality_threshold <= 1:
        raise AuxPPSInvalidValue("overallProductQualityTreshold (SKP) must be between 0 and 1")

    # L1c product export parameters.
    if aux_pps.l1c_product_export.ql_range_decimation_factor <= 0:
        raise AuxPPSInvalidValue("qlRangeDecimationFactor must be a positive integer")
    if aux_pps.l1c_product_export.ql_range_averaging_factor <= 0:
        raise AuxPPSInvalidValue("qlRangeAveragingFactor must be a positive integer")
    if aux_pps.l1c_product_export.ql_azimuth_decimation_factor <= 0:
        raise AuxPPSInvalidValue("qlAzimuthDecimationFactor must be a positive integer")
    if aux_pps.l1c_product_export.ql_azimuth_averaging_factor <= 0:
        raise AuxPPSInvalidValue("qlAzimuthAveragingFactor must be a positive integer")
    if aux_pps.l1c_product_export.pixel_quantity != common.PixelQuantityType.BETA_NOUGHT:
        raise AuxPPSInvalidValue("pixelQuantity must be 'Beta-Nought'")
    if aux_pps.l1c_product_export.pixel_representation != common.PixelRepresentationType.ABS_PHASE:
        raise AuxPPSInvalidValue("pixelRepresentation must be 'Abs Phase'")
