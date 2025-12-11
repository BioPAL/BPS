# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Utilities for L1 annotation xml file management"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arepytools.io import (
    iter_channels,
    metadata,
    open_product_folder,
    read_metadata,
    read_raster_with_raster_info,
)
from arepytools.math.genericpoly import create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import Polarization
from bps.common.io import common
from bps.transcoder.io import common_annotation_l1
from scipy.constants import speed_of_light


@dataclass
class FrequencyMaskStats:
    """Frequency mask statistics"""

    polarization: Polarization

    isolated_affected_lines_percentage: float
    isolated_max_affected_bandwidth_percentage: float
    isolated_avg_affected_bandwidth_percentage: float

    persistent_affected_lines_percentage: float
    persistent_max_affected_bandwidth_percentage: float
    persistent_avg_affected_bandwidth_percentage: float


@dataclass
class TimeMaskStats:
    """Time mask statistics"""

    polarization: Polarization
    affected_lines_percentage: float
    average_affected_samples_percentage: float
    max_affected_samples_percentage: float


@dataclass
class RFIMasksStatistics:
    """RFI masks statistics"""

    time_stats: list[TimeMaskStats]
    freq_stats: list[FrequencyMaskStats]


def translate_polarisation_to_model(
    pol: Polarization,
) -> common.PolarisationType:
    """BPS common Polarization type to annotation type"""
    return common.PolarisationType(pol.name)


def translate_bps_polarisation(
    pol: Polarization,
) -> common.PolarisationType:
    """BPS common Polarization type to annotation type"""
    return common.PolarisationType(pol.name)


def fill_rfi_isolated_fm_report_item(
    freq_stats: FrequencyMaskStats,
) -> common_annotation_l1.RfiIsolatedFmreportType:
    """Fill isolated RI report item"""
    return common_annotation_l1.RfiIsolatedFmreportType(
        percentage_affected_lines=freq_stats.isolated_affected_lines_percentage,
        max_percentage_affected_bw=freq_stats.isolated_max_affected_bandwidth_percentage,
        avg_percentage_affected_bw=freq_stats.isolated_avg_affected_bandwidth_percentage,
        polarisation=translate_bps_polarisation(freq_stats.polarization),
    )


def fill_rfi_persistent_fm_report_item_to_model(
    freq_stats: FrequencyMaskStats,
) -> common_annotation_l1.RfiPersistentFmreportType:
    """Fill persistent RI report item"""
    return common_annotation_l1.RfiPersistentFmreportType(
        polarisation=translate_polarisation_to_model(freq_stats.polarization),
        max_percentage_affected_bw=freq_stats.persistent_max_affected_bandwidth_percentage,
        avg_percentage_affected_bw=freq_stats.persistent_avg_affected_bandwidth_percentage,
        percentage_affected_lines=freq_stats.persistent_affected_lines_percentage,
    )


def translate_rfi_freq_stats_to_model(
    freq_stats: list[FrequencyMaskStats],
) -> tuple[
    list[common_annotation_l1.RfiIsolatedFmreportType],
    list[common_annotation_l1.RfiPersistentFmreportType],
]:
    """Translate RFI Frequency statistics to report list"""

    isolated = [fill_rfi_isolated_fm_report_item(item) for item in freq_stats]

    persistent = [fill_rfi_persistent_fm_report_item_to_model(item) for item in freq_stats]

    return isolated, persistent


def translate_rfi_time_stats(
    time_stats: list[TimeMaskStats],
) -> list[common_annotation_l1.RfiTmreportType]:
    """Translate RFI Time statistics to report list"""

    def translate_rfi_time_stats_item(
        time_stats: TimeMaskStats,
    ) -> common_annotation_l1.RfiTmreportType:
        return common_annotation_l1.RfiTmreportType(
            polarisation=translate_polarisation_to_model(time_stats.polarization),
            percentage_affected_lines=time_stats.affected_lines_percentage,
            avg_percentage_affected_samples=time_stats.average_affected_samples_percentage,
            max_percentage_affected_samples=time_stats.max_affected_samples_percentage,
        )

    return [translate_rfi_time_stats_item(item) for item in time_stats]


def _read_geometric_dc_product(
    product: Path,
) -> dict[metadata.EPolarization, metadata.DopplerCentroidVector]:
    """Read geometric dc product"""

    geometric_dc: dict[metadata.EPolarization, metadata.DopplerCentroidVector] = {}

    for file in product.iterdir():
        if file.suffix == ".xml":
            content = read_metadata(file)
            geometric_dc[content.get_swath_info().polarization] = content.get_doppler_centroid()

    return geometric_dc


def _read_dc_grid_product(
    product: Path,
) -> tuple[dict[metadata.EPolarization, np.ndarray], list[float]]:
    """Read geometric dc product"""
    values: dict[metadata.EPolarization, np.ndarray] = {}

    pf = open_product_folder(product)
    for channel_idx, channel_md in iter_channels(pf):
        raster_info = channel_md.get_raster_info()
        polarization = channel_md.get_swath_info().polarization
        data = read_raster_with_raster_info(pf.get_channel_data(channel_idx), raster_info)
        values[polarization] = data.squeeze()

        assert isinstance(raster_info.samples_start, float)
        rg_axis = [
            raster_info.samples_start + rg_index * raster_info.samples_step for rg_index in range(raster_info.samples)
        ]

    return values, rg_axis


@dataclass
class DCAnnotations:
    """Additional DC info, when combined estimation is enabled"""

    geometric_dc: dict[metadata.EPolarization, metadata.DopplerCentroidVector]
    dc_grid: dict[metadata.EPolarization, np.ndarray]
    rg_axis: list[float]

    @classmethod
    def from_products(cls, geometric_dc_product: Path, combined_dc_grid: Path) -> DCAnnotations:
        """Fill information from products"""
        geometric_dc = _read_geometric_dc_product(geometric_dc_product)
        dc_grid, rg_axis = _read_dc_grid_product(combined_dc_grid)

        return DCAnnotations(
            geometric_dc=geometric_dc,
            dc_grid=dc_grid,
            rg_axis=rg_axis,
        )


def get_list_of_dc_poly(
    polylist: metadata.DopplerCentroidVector | None,
) -> list[metadata.DopplerCentroid]:
    """Get list of dc poly"""
    if polylist is None:
        return []

    return [polylist.get_poly(index) for index in range(polylist.get_number_of_poly())]


def get_coefficients_from_poly(poly: metadata._Poly2D | None) -> list[float]:
    """Get the coefficients from the poly"""
    if poly is None:
        return []

    assert poly.coefficients is not None
    return np.concatenate((poly.coefficients[0:2], poly.coefficients[4:])).tolist()


def _compute_average_combined_dc_values(
    dc_grid: dict[metadata.EPolarization, np.ndarray],
) -> np.ndarray:
    dc_values = list(dc_grid.values())
    assert len(dc_values) > 0

    avg_dc_value = dc_values[0].copy()
    for dc_value in dc_values[1:]:
        avg_dc_value += dc_value
    avg_dc_value /= len(dc_values)

    return avg_dc_value


@dataclass
class CombinedDCStatistics:
    slant_range_times: list[float]
    values: list[float]
    rmse: float


def build_list_of_combined_dc_stats_from_annotations_and_poly(
    dc_annotations: DCAnnotations,
    combined_dc: metadata.DopplerCentroidVector,
) -> list[CombinedDCStatistics]:
    """Compute combined poly statistics"""
    grid_values = _compute_average_combined_dc_values(dc_annotations.dc_grid)
    composite_polynomial = create_sorted_poly_list(combined_dc)
    combined_stats_list = []
    for index in range(combined_dc.get_number_of_poly()):
        poly_values = composite_polynomial.evaluate(
            (
                combined_dc.get_poly(index).t_ref_az,
                np.array(dc_annotations.rg_axis),
            )
        )
        rmse = np.sqrt(np.mean((poly_values - grid_values) ** 2))
        combined_stats_list.append(
            CombinedDCStatistics(
                slant_range_times=dc_annotations.rg_axis,
                values=grid_values.tolist(),
                rmse=rmse,
            )
        )
    return combined_stats_list


def fill_empty_coordinate_conversion_type(
    azimuth_time: PreciseDateTime,
) -> common_annotation_l1.CoordinateConversionType:
    """Fill an empty coordinate conversion type"""
    return common_annotation_l1.CoordinateConversionType(
        azimuth_time=azimuth_time,
        t0=0.0,
        sr0=0.0,
        slant_to_ground_coefficients=[],
        gr0=0.0,
        ground_to_slant_coefficients=[],
    )


def fill_coordinate_conversion_type(
    ground_to_slant: metadata.GroundToSlant,
    slant_to_ground: metadata.SlantToGround | None,
) -> common_annotation_l1.CoordinateConversionType:
    """Fill coordinate conversion section"""
    azimuth_time: PreciseDateTime = ground_to_slant.t_ref_az  # type: ignore
    if slant_to_ground is None:
        t0 = 0.0
        sr0 = 0.0
        slant_to_ground_coefficients = []
    else:
        assert slant_to_ground.coefficients is not None
        t0: float = slant_to_ground.t_ref_rg  # type: ignore
        sr0 = t0 * speed_of_light / 2
        slant_to_ground_coefficients = (
            np.multiply(get_coefficients_from_poly(slant_to_ground), speed_of_light / 2)
        ).tolist()

    gr0: float = ground_to_slant.t_ref_rg  # type: ignore
    ground_to_slant_coefficients = get_coefficients_from_poly(ground_to_slant)

    return common_annotation_l1.CoordinateConversionType(
        azimuth_time,
        t0,
        sr0,
        slant_to_ground_coefficients,
        gr0,
        ground_to_slant_coefficients,
    )


def fill_tx_pulse(pulse: metadata.Pulse, azimuth_time: PreciseDateTime) -> common_annotation_l1.TxPulseType:
    assert pulse.bandwidth is not None
    chirp_rate = pulse.bandwidth / pulse.pulse_length
    chirp_rate = chirp_rate if pulse.pulse_direction == metadata.EPulseDirection.up else -chirp_rate
    assert pulse.pulse_length is not None
    assert pulse.pulse_start_frequency is not None
    assert pulse.pulse_start_phase is not None
    return common_annotation_l1.TxPulseType(
        azimuth_time=azimuth_time,
        tx_pulse_length=pulse.pulse_length,
        tx_pulse_start_frequency=pulse.pulse_start_frequency,
        tx_pulse_start_phase=pulse.pulse_start_phase,
        tx_pulse_ramp_rate=chirp_rate,
    )


def fill_raster_info_from_sar_image_slc(
    sar_image: common_annotation_l1.SarImageType,
) -> metadata.RasterInfo:
    """Fill raster info"""

    raster_info = metadata.RasterInfo(
        lines=sar_image.number_of_lines,
        samples=sar_image.number_of_samples,
        celltype="FLOAT_COMPLEX",
        filename=None,
        header_offset_bytes=0,
        row_prefix_bytes=0,
        byteorder="LITTLEENDIAN",
    )

    raster_info.set_samples_axis(
        sar_image.first_sample_slant_range_time,
        "s",
        sar_image.range_time_interval,
        "s",
    )

    raster_info.set_lines_axis(
        sar_image.first_line_azimuth_time,
        "Utc",
        sar_image.azimuth_time_interval,
        "s",
    )

    return raster_info


def fill_raster_info_from_sar_image_grd(
    sar_image: common_annotation_l1.SarImageType,
) -> metadata.RasterInfo:
    """Fill raster info"""

    raster_info = metadata.RasterInfo(
        lines=sar_image.number_of_lines,
        samples=sar_image.number_of_samples,
        celltype="FLOAT32",
        filename=None,
        header_offset_bytes=0,
        row_prefix_bytes=0,
        byteorder="LITTLEENDIAN",
    )

    raster_info.set_samples_axis(
        sar_image.range_coordinate_conversion[0].gr0,
        "m",
        sar_image.range_pixel_spacing,
        "m",
    )

    raster_info.set_lines_axis(
        sar_image.first_line_azimuth_time,
        "Utc",
        sar_image.azimuth_time_interval,
        "s",
    )

    return raster_info
