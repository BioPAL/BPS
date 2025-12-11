# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""A generic SAR product"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    iter_channels,
    metadata,
    open_product_folder,
    read_raster_with_raster_info,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.math import genericpoly
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.transcoder.sarproduct.footprint_utils import (
    compute_footprint,
    compute_ground_corner_points_on_wgs84,
)
from bps.transcoder.sarproduct.platform_heading_utils import compute_platform_heading


class SARProduct:
    """SAR Product"""

    def __init__(self) -> None:
        """fields"""
        self.name: str | None = None
        self.mission: str | None = None
        self.acquisition_mode: str | None = None
        self.type: Literal["SLC", "GRD"] | None = None
        self.swath_list: list[str] = []
        self.polarization_list: list[Literal["H/H", "H/V", "V/H", "V/V"]] = []
        self.start_time: PreciseDateTime | None = None
        self.stop_time: PreciseDateTime | None = None
        self.orbit_number: int | None = None
        self.orbit_direction: Literal["ASCENDING", "DESCENDING"] | None = None
        self.anx_time: PreciseDateTime | None = None
        self.footprint: list[list[float]] = []  # 4 elements, each lat, lon in deg
        self.platform_heading: float | None = None
        self.gcp_list: list[list] = []  # N elements, each x,y,z,row,col

        self.channels: int | None = None

        self.data_list: list[np.ndarray] = []

        self.raster_info_list: list[metadata.RasterInfo] = []
        self.burst_info_list: list[metadata.BurstInfo] = []
        self.dataset_info: list[metadata.DataSetInfo] = []
        self.swath_info_list: list[metadata.SwathInfo] = []
        self.sampling_constants_list: list[metadata.SamplingConstants] = []
        self.acquisition_timeline_list: list[metadata.AcquisitionTimeLine] = []
        self.data_statistics_list: list[metadata.DataStatistics] = []
        self.general_sar_orbit: list[metadata.StateVectors] = []
        self.dc_vector_list: list[metadata.DopplerCentroidVector] = []
        self.dc_eff_vector_list: list[metadata.DopplerCentroidVector] = []
        self.dr_vector_list: list[metadata.DopplerRateVector] = []
        self.slant_to_ground_list: list[metadata.SlantToGroundVector] = []
        self.ground_to_slant_list: list[metadata.GroundToSlantVector] = []
        self.attitude_info: list[metadata.AttitudeInfo] = []
        self.pulse_list: list[metadata.Pulse] = []

    @classmethod
    def read(cls, product_path: Path) -> SARProduct:
        """Read from product path"""

        bps_logger.debug("Reading SAR product..")

        pf = open_product_folder(product_path)
        first_channel = pf.get_channels_list()[0]

        product = SARProduct()
        product.name = product_path.name
        product.channels = len(pf.get_channels_list())

        for channel, ch in iter_channels(pf):
            data_file = pf.get_channel_data(channel)
            data = read_raster_with_raster_info(data_file, ch.get_raster_info())
            product.data_list.append(data)

            product.raster_info_list.append(ch.get_raster_info())
            product.burst_info_list.append(ch.get_burst_info())
            if channel == first_channel:
                product.dataset_info.append(ch.get_dataset_info())
            product.swath_info_list.append(ch.get_swath_info())
            product.sampling_constants_list.append(ch.get_sampling_constants())
            product.acquisition_timeline_list.append(ch.get_acquisition_time_line())
            product.data_statistics_list.append(ch.get_data_statistics())
            if channel == first_channel:
                product.general_sar_orbit.append(ch.get_state_vectors())
            product.dc_vector_list.append(ch.get_doppler_centroid())
            product.dc_eff_vector_list.append(ch.get_doppler_centroid())
            product.dr_vector_list.append(ch.get_doppler_rate())
            product.slant_to_ground_list.append(ch.get_slant_to_ground())
            product.ground_to_slant_list.append(ch.get_ground_to_slant())
            if channel == first_channel:
                product.attitude_info.append(ch.get_attitude_info())
            product.pulse_list.append(ch.get_pulse())

        # Set remaining product attributes
        product.mission = product.dataset_info[0].sensor_name
        product.acquisition_mode = product.dataset_info[0].acquisition_mode

        if product.dataset_info[0].image_type in (
            "AZIMUTH FOCUSED",
            "AZIMUTH FOCUSED RANGE COMPENSATED",
        ):
            product.type = "SLC"
        elif product.dataset_info[0].image_type == "MULTILOOK":
            product.type = "GRD"

        for si in product.swath_info_list:
            assert si.swath is not None
            assert si.polarization.value is not None
            product.swath_list.append(si.swath)
            product.polarization_list.append(si.polarization.value)

        starts: list[PreciseDateTime] = []
        stops: list[PreciseDateTime] = []
        for ri in product.raster_info_list:
            assert ri.lines_start is not None
            assert isinstance(ri.lines_start, PreciseDateTime)
            starts.append(ri.lines_start)
            stops.append(ri.lines_start + (ri.lines - 1) * ri.lines_step)

        product.start_time = min(starts)
        product.stop_time = max(stops)

        product.orbit_number = product.general_sar_orbit[0].orbit_number
        product.orbit_direction = product.general_sar_orbit[0].orbit_direction.value

        gso = create_general_sar_orbit(product.general_sar_orbit[0], ignore_anx_after_orbit_start=True)
        if gso.anx_times is not None and gso.anx_times.size > 0:
            product.anx_time = gso.anx_times[0]

        assert product.dataset_info[0].side_looking is not None
        look_direction = product.dataset_info[0].side_looking.value

        time_corners = product.compute_time_corners()
        product.footprint.extend(
            compute_footprint(
                time_corners,
                gso,
                look_direction,
            )
        )
        product.platform_heading = compute_platform_heading(time_corners, gso, look_direction)

        range_times, aximuth_times = product.compute_time_axis()
        product.gcp_list.extend(compute_ground_corner_points_on_wgs84(range_times, aximuth_times, gso, look_direction))

        bps_logger.debug("..done")
        return product

    def write(self, product_path: Path | str, *, use_eff_dc_vectors: bool = False) -> None:
        """Write a sar product"""
        bps_logger.debug(f"Writing SAR product {product_path}")

        product_path = Path(product_path)

        if product_path.exists():
            raise FileExistsError(f"Folder {product_path} already exists.")

        pf = create_product_folder(product_path)

        assert self.channels is not None
        for channel in range(self.channels):
            meta = create_new_metadata()

            raster_info = self.raster_info_list[channel]
            raster_info.file_name = pf.get_channel_data(channel).name
            meta.insert_element(raster_info)
            meta.insert_element(self.burst_info_list[channel])
            meta.insert_element(self.dataset_info[0])
            meta.insert_element(self.swath_info_list[channel])
            meta.insert_element(self.sampling_constants_list[channel])
            meta.insert_element(self.acquisition_timeline_list[channel])
            meta.insert_element(self.data_statistics_list[channel])
            meta.insert_element(self.general_sar_orbit[0])
            meta.insert_element(
                self.dc_eff_vector_list[channel] if use_eff_dc_vectors else self.dc_vector_list[channel]
            )
            meta.insert_element(self.dr_vector_list[channel])
            if self.slant_to_ground_list:
                meta.insert_element(self.slant_to_ground_list[channel])
            if self.ground_to_slant_list:
                meta.insert_element(self.ground_to_slant_list[channel])
            meta.insert_element(self.attitude_info[0])
            meta.insert_element(self.pulse_list[channel])

            write_metadata(meta, pf.get_channel_metadata(channel))
            write_raster_with_raster_info(pf.get_channel_data(channel), self.data_list[channel], raster_info)

        bps_logger.debug("..done")

    def compute_time_corners(
        self,
    ) -> tuple[float, float, PreciseDateTime, PreciseDateTime]:
        """Returns slant range time axis"""
        raster_info = self.raster_info_list[0]
        ground_to_slant_poly = (
            genericpoly.create_sorted_poly_list(self.ground_to_slant_list[0]) if self.type == "GRD" else None
        )
        return _compute_time_corners(raster_info, ground_to_slant_poly)

    def compute_time_axis(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns samples_start, samples_stop, lines_start, lines_stop"""
        raster_info = self.raster_info_list[0]
        ground_to_slant_poly = (
            genericpoly.create_sorted_poly_list(self.ground_to_slant_list[0]) if self.type == "GRD" else None
        )
        return _compute_time_axis(raster_info, ground_to_slant_poly)


def _compute_time_axis(
    raster_info: metadata.RasterInfo,
    ground_to_slant_poly: genericpoly.SortedPolyList | None,
) -> tuple[np.ndarray, np.ndarray]:
    range_axis = raster_info.samples_start + raster_info.samples_step * np.arange(0, raster_info.samples)
    azimuth_axis = raster_info.lines_start + raster_info.lines_step * np.arange(0, raster_info.lines)

    if ground_to_slant_poly:
        range_axis = ground_to_slant_poly.evaluate((azimuth_axis[0], range_axis))

    return range_axis, azimuth_axis


def _compute_time_corners(
    raster_info: metadata.RasterInfo,
    ground_to_slant_poly: genericpoly.SortedPolyList | None,
) -> tuple[float, float, PreciseDateTime, PreciseDateTime]:
    samples_start = raster_info.samples_start
    samples_stop = raster_info.samples_start + raster_info.samples_step * (raster_info.samples - 1)
    lines_start = raster_info.lines_start
    lines_stop = raster_info.lines_start + raster_info.lines_step * (raster_info.lines - 1)

    if ground_to_slant_poly:
        samples_start = ground_to_slant_poly.evaluate((raster_info.lines_start, samples_start))
        samples_stop = ground_to_slant_poly.evaluate((raster_info.lines_start, samples_stop))

    return samples_start, samples_stop, lines_start, lines_stop  # type: ignore
