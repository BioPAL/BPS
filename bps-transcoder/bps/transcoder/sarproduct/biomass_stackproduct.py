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

from dataclasses import dataclass
from typing import Any

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.io.main_annotation_l1c import (
    BIOMASSStackCoregistrationParameters,
    BIOMASSStackInSARParameters,
    BIOMASSStackProcessingParameters,
    BIOMASSStackQuality,
)
from bps.transcoder.sarproduct.sarproduct import SARProduct
from bps.transcoder.utils.production_model_utils import encode_product_name_id_value
from bps.transcoder.utils.time_conversions import (
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
    "rangeTerrainSlope",
    "azimuthTerrainSlope",
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
    "coherence",
    "interferogram",
    "ionospherePhaseScreens",
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


class BIOMASSStackProduct(SARProduct):
    """Object that stores a BIOMASS Stack Product (L1c)."""

    def __init__(
        self,
        *,
        product: SARProduct | None = None,
        product_primary: SARProduct | None = None,
        stack_processing_parameters: BIOMASSStackProcessingParameters | None = None,
        stack_coregistration_parameters: BIOMASSStackCoregistrationParameters | None = None,
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
