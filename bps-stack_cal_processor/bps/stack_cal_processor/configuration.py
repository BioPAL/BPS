# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Calibration Configuration Utilities
-----------------------------------------
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import read_metadata
from arepytools.io.productfolder2 import ProductFolder2
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from bps.stack_cal_processor.core.filtering import ConvolutionWindowType
from bps.stack_cal_processor.core.utils import (
    SubLookFilterType,
    compute_earth_radius,
    compute_incidence_angle_from_orbit,
    compute_look_angle_from_orbit,
    compute_satellite_altitude,
    compute_satellite_ground_speed,
    compute_satellite_state,
    compute_target_ground_speed,
)

# The names of the modules.
AZF_NAME = "azimuthSpectralFilter"
IOB_NAME = "slowIonosphereRemoval"
CAL_NAME = "inSarPhaseCalibration"
PPR_NAME = "phasePlaneRemoval"
SKP_NAME = "skpPhaseCalibration"


# Units of calibration configurations that has units.
UNITS_CAL_CONFIG = {
    # Common.
    "ionosphere_latitude_threshold": " [rad]",
    # IOB.
    "max_lh_phase_delta": " [rad]",
    "sublook_window_sizes": " [px]",
    "split_spectrum_decimation_factors": " [px]",
    # PPR.
    "fft2_peak_window_size": " [px]",
    # SKP.
    "estimation_window_size": " [m]",
    "output_azimuth_subsampling_step": " [px]",
    "output_range_subsampling_step": " [px]",
    "median_filter_window_size": " [m]",
}


# The single-baseline and multi-baseline approach for IOB and IOS.
class BaselineMethodType(Enum):
    """Baseline selection for some of the calibration modules."""

    SINGLE_BASELINE = "SINGLE_BASELINE"
    MULTI_BASELINE = "MULTI_BASELINE"


@dataclass
class StackCalConf:
    """Configuration of the calibration module."""

    class StackCalConfError(ValueError):
        """Handle a generic value error for the stack cal configuration."""

    @dataclass
    class AzfConf:
        """
        Configuration of the Azimuth Spectral Filtering (AZF)."""

        class AzfValueError(ValueError):
            """Handle a generic invalid value of the AZF configuration."""

            def __init__(self, message: str):
                super().__init__(f"[{AZF_NAME}] {message}")

        use_primary_spectral_weighting_window_flag: bool
        """If true, use spectral window from coregistration primary."""

        window_type: ConvolutionWindowType | None = None
        """The window applied by the AZF. If None, use that in the L1a data."""

        window_parameter: float | None = None
        """The window parameter."""

        use_32bit_precision: bool = True
        """Use 32-bit precision (aka complex64 and float32) for model estimations."""

    @dataclass
    class IobConf:
        """Configuration of the Background Ionosphere Removal (IOB)."""

        class IobValueError(ValueError):
            """Handle a generic invalid value in the IOB configuration."""

            def __init__(self, message: str = ""):
                super().__init__(f"[{IOB_NAME}] {message}")

        phase_unwrapping_flag: bool = True
        """If true, phase unwrapping is performed during single-base interferometry."""

        min_coherence_threshold: float = 0.0  # [%].
        """Threshold on the coherence to consider a pixel usable for fitting the phase plane."""

        max_lh_phase_delta: float = np.inf  # [rad].
        """Phase delta between low and high interferograms shall not exceeds this value."""

        min_usable_pixel_ratio: float = np.inf  # [rad].
        """Minimum ratio of pixels that are usable for fitting the phase plane."""

        compensate_l1_iono_phase_screen_flag: bool = True
        """If true, the ionospheric phase screen from L1 will be applied back."""

        range_look_band: float = 0.3  # [adim]
        """Percentage of range band used in the IOB."""

        range_look_frequency: float = 0.3  # [adim]
        """Percentage of range frequency used in the IOB."""

        sublook_window_sizes: tuple[int, int] = (501, 41)  # [px].
        """Window sizes to generate the sub-looks interferograms."""

        sublook_filter_type: SubLookFilterType = SubLookFilterType.GAUSSIAN
        """Filter type used to compute the sub-looks."""

        sublook_filter_param: float | int = 7.0
        """The filter parameter used by the sub-look's filter."""

        split_spectrum_decimation_factors: tuple[int, int] = (5, 5)  # [px]
        """Decimation factors for subsampling the interferograms."""

        polarization_index: int = 0
        """Reference polarization index used for single- and multi-baseline."""

        ionosphere_latitude_threshold: float = np.deg2rad(0.0)  # [rad}
        """Skip when satellite latitude is below this threshold."""

        baseline_method: BaselineMethodType = BaselineMethodType.MULTI_BASELINE
        """As to whether multi-baseline or single-baseline should be used."""

        multi_baseline_uniform_weighting: bool = False
        """Use equal weighting for multi-baseline estimation."""

        multi_baseline_cb_ratio_threshold: float = np.inf
        """Maximum allowed percentage of CB displacement for multi-baseline pairs."""

        quality_threshold: float = 0.0
        """Estimated phase ramps with quality below this threshold will be ignored."""

        use_32bit_precision: bool = True
        """Use 32-bit precision (aka complex64 and float32) for model estimations."""

    @dataclass
    class PprConf:
        """Configuration of the Phase Plane Removal (PPR)."""

        class PprValueError(ValueError):
            """Handle a generic invalid value in the IOB configuration."""

            def __init__(self, message: str = ""):
                super().__init__(f"[{PPR_NAME}] {message}")

        polarization_index: int = 0
        """Reference polarization index used for estimating the coherences/inteferograms."""

        fft2_zero_padding_upsampling_factor: float = 1.2
        """Multiplicative factor that controls zero-padding before FFT2 to increasing frequency-domain sampling density."""

        fft2_peak_window_size: int = 3
        """Size of the local window used for parabolic interpolation of the FFT2 around the max/peak."""

        use_32bit_precision: bool = True
        """Use 32-bit precision (aka complex64 and float32) for model estimation."""

    @dataclass
    class SkpConf:
        """Configuration of the Sum-of-Kronecker-Products module (SKP)."""

        class SkpValueError(ValueError):
            """Handle a generic invalid value in the SKP configuration."""

            def __init__(self, message: str):
                super().__init__(f"[{SKP_NAME}] {message}")

        estimation_window_size: float = 500.0  # [m]
        """Resolution of the estimation grid."""

        output_azimuth_subsampling_step: int = 7  # [px].
        """Azimuth decimation factor for the output grid."""

        output_range_subsampling_step: int = 1  # [px].
        """Range decimation factor for the output grid."""

        skp_phase_correction_flag: bool = True
        """Whether the SKP calibration phase should be corrected."""

        only_flattening_phase_correction_flag: bool = False
        """Whether only the DSI should be corrected (required if phase correction is on)."""

        skp_calibration_phase_screen_quality_threshold: float = 0.0
        """Estimations with a lower quality than this value will not be corrected."""

        overall_product_quality_threshold: float = 0.0
        """Percentage of valid qualities that are required to accept the estimation."""

        nyquist_window_bounds: tuple[float, float] = (0.5, 0.9)  # [0-1]
        """Control the ratio between SKP estimation window and L1a LUT resolution."""

        median_filter_flag: bool = False
        """Post-process the estimated SKP phase screens with a median filter."""

        median_filter_window_size: float = 1000.0  # [m].
        """Size of the median filter's window (same in range and azimuth)."""

        exclude_mpmb_polarization_cross_covariance_flag: bool = False
        """If true, exclude the polarization cross-covariances when computing the MPMB coherence matrix."""

        use_32bit_precision: bool = True
        """Use 32-bit precision (aka complex64 and float32) for model estimations."""

    azf_conf: AzfConf | None = None
    """The Azimuth Filtering (AZF) configurations."""

    iob_conf: IobConf | None = None
    """The Background Ionosphere Removal (IOB) configurations."""

    ppr_conf: PprConf | None = None
    """Phase Plane Removal (PPR) configurations."""

    skp_conf: SkpConf | None = None
    """The Sum-of-Kronecker-Product (SKP) configurations."""


@dataclass
class StackDataSpecs:
    """
    This class stores the specifications of a BPS data stack. In
    particular, we will assume

      - PRF and and range bandwidth to be the same for all images and
        polarizations.
      - Central frequencies to be the same for all images as well.
      - Azimuth bandwidths can vary from stack image to stack image, though
        constant with respect to different polarizations.

    Note that not all parameters are required by each calibration module.
    Therefore, the related parameters can be set to None when not required.
    """

    azimuth_compression_window_types: tuple[ConvolutionWindowType | None, ...]
    """The window type applied to the azimuth spectrum."""

    azimuth_compression_window_parameters: tuple[float | None, ...]  # [adim]
    """Azimuth parameters used to preprocess the image (1 per image)."""

    azimuth_compression_window_bands: tuple[float | None, ...]  # [adim]
    """Azimuth parameters used to preprocess the image (1 per image)."""

    azimuth_bandwidths: tuple[float, ...]  # [Hz]
    """The azimuth look bandwidths (1 per image)."""

    azimuth_sampling_step: float  # [s]
    """The azimuth sampling step, i.e. 1/PRF in azimuth. Common to all images."""

    azimuth_sampling_starts: tuple[PreciseDateTime, ...]  # [date]
    """Absolute start time for the azimuth component (1 per image)."""

    range_compression_window_types: tuple[ConvolutionWindowType | None, ...]
    """The window type applied to the range spectrum."""

    range_compression_window_parameters: tuple[float | None, ...]  # [adim]
    """Range parameters used to preprocess each image (1 per image)."""

    range_compression_window_bands: tuple[float | None, ...]  # [adim]
    """Range parameters used to preprocess the image (1 per image)."""

    range_bandwidth: float  # [Hz]
    """The range look bandwidth (common to all frames)."""

    range_sampling_step: float  # [Hz]
    """The range sampling step, i.e. 1 / PRF in range. Common to all images."""

    range_sampling_starts: tuple[float, ...]  # [s]
    """Absolute start time for the range component (1 per image)."""

    central_frequency: float  # [Hz]
    """The pulse central frequency. Common to all images."""

    satellite_positions: tuple[float, ...]  # [m]
    """Satellite ECEF positions."""

    satellite_altitudes: tuple[float, ...]  # [m]
    """Satellite alititudes wrt the WGS84 ellispsoid (1 per image)."""

    satellite_ground_speeds: tuple[float, ...]  # [m/s]
    """Satellite speeds on the ground (1 per image)."""

    satellite_orbital_speeds: tuple[float, ...]  # [m/s]
    """Satellite tangential speeds at the orbit (1 per image)."""

    target_ground_speeds: tuple[float, ...]  # [m/s]
    """Target speeds at the ground (1 per image)."""

    earth_radii: tuple[float, ...]  # [m]
    """Earth's radii at the projection of the satellite on the WGS84 (1 per image)."""

    look_angles: tuple[float, ...]  # [rad]
    """Look angles (1 per image)."""

    incidence_angles: tuple[float, ...]  # [rad]
    """Incidence angles (1 per image)."""

    azimuth_central_frequency: float = 0.0  # [Hz]
    """The azimuth spectral shift with respect to the central (dominant) frequency."""


def fill_stack_data_specs(
    *,
    coreg_products: tuple[ProductFolder2, ...],
    coreg_primary_image_index: int,
    window_compression_parameters: tuple[dict[str, float], ...],
    roi: RegionOfInterest | None = None,
    azimuth_central_frequency: float | None = None,
) -> StackDataSpecs:
    """
    Populate the stack specs from the input data.

    Parameters
    ----------
    coreg_products: tuple[ProductFolder2, ...]
        The stack product folders.

    coreg_primary_image_index: int
        The coregistration primary image.

    window_compression_parameters: tuple[dict[str, float], ...]
        The range compression parameters used for processing the data.

    roi: Optional[RegionOfInterest] = None
        Optionally, a ROI in the data.

    azimuth_central_frequency: Optional[float] = None [Hz]
        Optionally, the central frequency in azimuth.

    Raises
    ------
    ValueError

    Return
    ------
    StackDataSpecs
        The stack's data specs.

    """
    # We use the coreg primary as reference rasters for common quantities.
    coreg_primary_product = coreg_products[coreg_primary_image_index]
    coreg_primary_channel = read_metadata(coreg_primary_product.get_channel_metadata(1))
    coreg_primary_raster_info = coreg_primary_channel.get_raster_info()
    coreg_primary_sampling_constants = coreg_primary_channel.get_sampling_constants()
    coreg_primary_dataset_info = coreg_primary_channel.get_dataset_info()

    azimuth_sampling_step = float(1 / coreg_primary_sampling_constants.faz_hz)
    range_bandwidth = float(coreg_primary_sampling_constants.brg_hz)
    range_sampling_step = float(1 / coreg_primary_sampling_constants.frg_hz)
    central_frequency = float(coreg_primary_dataset_info.fc_hz)

    # Handle the optional ROI.
    if roi is None:
        roi = [0, 0, coreg_primary_raster_info.lines, coreg_primary_raster_info.samples]
    raise_if_roi_is_invalid(coreg_primary_raster_info, roi)

    # We populate the image-dependent quantities.
    azimuth_compression_window_types = []
    azimuth_compression_window_parameters = []
    azimuth_compression_window_bands = []
    azimuth_bandwidths = []
    azimuth_sampling_starts = []
    range_compression_window_types = []
    range_compression_window_parameters = []
    range_compression_window_bands = []
    range_sampling_starts = []
    satellite_altitudes = []
    satellite_ground_speeds = []
    satellite_orbital_speeds = []
    satellite_positions = []
    target_ground_speeds = []
    earth_radii = []
    look_angles = []
    incidence_angles = []

    for coreg_product, win_param in zip(coreg_products, window_compression_parameters):
        # These are independent from polarization, so we use the first channel.
        channel = read_metadata(coreg_product.get_channel_metadata(1))
        raster_info = channel.get_raster_info()
        sampling_constants = channel.get_sampling_constants()
        dataset_info = channel.get_dataset_info()
        state_vectors = channel.get_state_vectors()
        sar_orbit = create_general_sar_orbit(state_vectors)

        azimuth_bandwidths.append(float(sampling_constants.baz_hz))
        azimuth_sampling_starts.append(raster_info.lines_start + roi[0] * raster_info.lines_step)
        azimuth_compression_window_types.append(ConvolutionWindowType(win_param["azimuth_window_type"].upper()))
        azimuth_compression_window_parameters.append(float(win_param["azimuth_window_coefficient"]))
        azimuth_compression_window_bands.append(
            float(
                win_param["azimuth_window_bandwidth"]
                / win_param.get("azimuth_window_total_bandwidth", sampling_constants.faz_hz)
            )
        )
        range_sampling_starts.append(float(raster_info.samples_start + roi[1] * raster_info.samples_step))
        range_compression_window_types.append(ConvolutionWindowType(win_param["range_window_type"].upper()))
        range_compression_window_parameters.append(float(win_param["range_window_coefficient"]))
        range_compression_window_bands.append(
            float(
                win_param["range_window_bandwidth"]
                / win_param.get("range_window_total_bandwidth", sampling_constants.frg_hz)
            )
        )

        (
            satellite_position,
            satellite_velocity,
            target_position,
        ) = compute_satellite_state(
            sar_orbit,
            raster_info,
            dataset_info,
            roi,
        )

        satellite_positions.append(satellite_position)
        satellite_altitudes.append(
            compute_satellite_altitude(satellite_position),
        )
        satellite_orbital_speeds.append(float(np.linalg.norm(satellite_velocity)))
        satellite_ground_speeds.append(float(compute_satellite_ground_speed(sar_orbit, raster_info, dataset_info, roi)))
        target_ground_speeds.append(
            float(compute_target_ground_speed(satellite_position, satellite_velocity, target_position))
        )
        earth_radii.append(float(compute_earth_radius(satellite_position)))
        look_angles.append(float(compute_look_angle_from_orbit(raster_info, sar_orbit, dataset_info, roi)))
        incidence_angles.append(float(compute_incidence_angle_from_orbit(sar_orbit, raster_info, dataset_info)))

    return StackDataSpecs(
        azimuth_compression_window_types=azimuth_compression_window_types,
        azimuth_compression_window_parameters=azimuth_compression_window_parameters,
        azimuth_compression_window_bands=azimuth_compression_window_bands,
        azimuth_central_frequency=azimuth_central_frequency,
        azimuth_bandwidths=azimuth_bandwidths,
        azimuth_sampling_step=azimuth_sampling_step,
        azimuth_sampling_starts=azimuth_sampling_starts,
        range_compression_window_types=range_compression_window_types,
        range_compression_window_parameters=range_compression_window_parameters,
        range_compression_window_bands=range_compression_window_bands,
        range_bandwidth=range_bandwidth,
        range_sampling_step=range_sampling_step,
        range_sampling_starts=range_sampling_starts,
        central_frequency=central_frequency,
        satellite_positions=satellite_positions,
        satellite_altitudes=satellite_altitudes,
        satellite_ground_speeds=satellite_ground_speeds,
        satellite_orbital_speeds=satellite_orbital_speeds,
        target_ground_speeds=target_ground_speeds,
        earth_radii=earth_radii,
        look_angles=look_angles,
        incidence_angles=incidence_angles,
    )


def log_calibration_params(
    conf_dict: dict,
    indent: int = 1,
):
    """Pretty print calibration parameters.

    Parameters
    ----------
    conf_dict: dict
        Configuration of the calibration module as dictionary.

    indent: int = 2
        Number of indentations to use in the logging.

    """
    # Log the parameters.
    bps_logger.info(
        "%sParameters:",
        "  " * (indent - 1),
    )
    for param_name, param_value in conf_dict.items():
        bps_logger.info(
            "%s%s%s: %s",
            "  " * indent,
            param_name,
            UNITS_CAL_CONFIG.get(param_name, ""),
            param_value.value if isinstance(param_value, Enum) else param_value,
        )
