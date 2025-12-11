# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to add footprint information
--------------------------------------
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import (
    create_new_metadata,
    metadata,
    open_product_folder,
    read_metadata,
    write_metadata,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_processor.settings.intermediate_names import (
    FOOTPRINT_FILE_NAME,
    IntermediateProductID,
)
from bps.l1_processor.settings.l1_intermediates import L1CoreProcessorOutputProducts
from bps.transcoder.sarproduct.dem_footprint_utils import read_from_dem_lut
from bps.transcoder.sarproduct.generic_product import GenericProduct


@dataclass
class FootprintPoints:
    """Corners and center"""

    @dataclass
    class GeographicPoint:
        """Geographic point information"""

        latitude_degree: float
        longitude_degree: float
        height: float
        incidence_angle_degree: float = 0.0
        look_angle_degree: float = 0.0

        @classmethod
        def from_point(cls, point: np.ndarray) -> FootprintPoints.GeographicPoint:
            """From point"""
            return FootprintPoints.GeographicPoint(
                latitude_degree=float(np.rad2deg(point[0])),
                longitude_degree=float(np.rad2deg(point[1])),
                height=float(point[2]),
            )

        def to_metadata(self) -> metadata.GeoPoint:
            """To aresys metadata"""
            return metadata.GeoPoint(
                lat=self.latitude_degree,
                lon=self.longitude_degree,
                height=self.height,
                theta_inc=self.incidence_angle_degree,
                theta_look=self.look_angle_degree,
            )

    north_east: GeographicPoint
    north_west: GeographicPoint
    south_east: GeographicPoint
    south_west: GeographicPoint
    center: GeographicPoint

    @classmethod
    def from_points(
        cls,
        corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        center: np.ndarray,
    ) -> FootprintPoints:
        """From points"""
        sorted_by_lat = np.array(corners)
        sorted_by_lat = np.sort(sorted_by_lat, axis=0)

        def _split_west_east(corner):
            west_point = corner[0]
            east_point = corner[1]
            if west_point[1] > east_point[1]:
                west_point, east_point = east_point, west_point
            return west_point, east_point

        north_corners = sorted_by_lat[2:]
        north_west_point, north_east_point = _split_west_east(north_corners)

        south_corners = sorted_by_lat[0:2]
        south_west_point, south_east_point = _split_west_east(south_corners)

        return FootprintPoints(
            center=FootprintPoints.GeographicPoint.from_point(center),
            north_east=FootprintPoints.GeographicPoint.from_point(north_east_point),
            north_west=FootprintPoints.GeographicPoint.from_point(north_west_point),
            south_east=FootprintPoints.GeographicPoint.from_point(south_east_point),
            south_west=FootprintPoints.GeographicPoint.from_point(south_west_point),
        )

    def to_metadata(self) -> metadata.GroundCornerPoints:
        """To metadata"""
        ground_corner_points = metadata.GroundCornerPoints()
        ground_corner_points.ne_point = self.north_east.to_metadata()
        ground_corner_points.se_point = self.south_east.to_metadata()
        ground_corner_points.nw_point = self.north_west.to_metadata()
        ground_corner_points.sw_point = self.south_west.to_metadata()
        ground_corner_points.center_point = self.center.to_metadata()
        return ground_corner_points


@dataclass
class FootprintSarPoints:
    """Corners and center"""

    @dataclass
    class SarPoint:
        """Point in sar coordinate"""

        line: PreciseDateTime
        sample: float

    corners: tuple[SarPoint, SarPoint, SarPoint, SarPoint]
    center: SarPoint

    @classmethod
    def from_raster_info(cls, raster_info: metadata.RasterInfo) -> FootprintSarPoints:
        """From metadata raster info"""
        range_length = (raster_info.samples - 1) * raster_info.samples_step
        near_range = raster_info.samples_start
        far_range = near_range + range_length
        central_range = near_range + range_length / 2

        azimuth_length = (raster_info.lines - 1) * raster_info.lines_step
        start_azimuth: PreciseDateTime = raster_info.lines_start  # type: ignore
        stop_azimuth = start_azimuth + azimuth_length
        central_azimuth = start_azimuth + azimuth_length / 2

        return FootprintSarPoints(
            corners=(
                FootprintSarPoints.SarPoint(sample=near_range, line=start_azimuth),
                FootprintSarPoints.SarPoint(sample=near_range, line=stop_azimuth),
                FootprintSarPoints.SarPoint(sample=far_range, line=start_azimuth),
                FootprintSarPoints.SarPoint(sample=far_range, line=stop_azimuth),
            ),
            center=FootprintSarPoints.SarPoint(sample=central_range, line=central_azimuth),
        )


def _compute_from_dem(
    points: FootprintSarPoints, dem_product: Path
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    dem_lut = GenericProduct.read_from_product_path(dem_product)

    corners_llh = tuple(
        np.array(read_from_dem_lut(dem_lut, sample=point.sample, line=point.line)) for point in points.corners
    )
    assert len(corners_llh) == 4

    center_llh = np.array(read_from_dem_lut(dem_lut, sample=points.center.sample, line=points.center.line))
    return corners_llh, center_llh


def _compute_from_ellipsoid(
    points: FootprintSarPoints, reference_metadata: metadata.MetaData
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    side_looking = reference_metadata.get_dataset_info().side_looking
    assert side_looking is not None
    look_direction = side_looking.value

    gso = create_general_sar_orbit(reference_metadata.get_state_vectors(), ignore_anx_after_orbit_start=True)

    corners_llh = tuple(xyz2llh(gso.sat2earth(point.line, point.sample, look_direction)) for point in points.corners)
    assert len(corners_llh) == 4

    center_llh = xyz2llh(gso.sat2earth(points.center.line, points.center.sample, look_direction))
    return corners_llh, center_llh


def _retrieve_reference_metadata(product: Path) -> metadata.MetaData:
    pf = open_product_folder(product)
    return read_metadata(pf.get_channel_metadata(pf.get_channels_list()[0]))


def _compute_footprint(reference_metadata: metadata.MetaData, footprint_computer) -> FootprintPoints:
    """Compute footprint from reference product"""
    points = FootprintSarPoints.from_raster_info(reference_metadata.get_raster_info())

    corners_llh, center_llh = footprint_computer(points)

    return FootprintPoints.from_points(corners_llh, center_llh)


def _write_footprint_file(corners: metadata.GroundCornerPoints, footprint_file: Path):
    ground_corner_points_metadata = create_new_metadata()
    ground_corner_points_metadata.insert_element(corners)
    write_metadata(ground_corner_points_metadata, footprint_file)


def _copy_file_to_destinations(source_file: Path, destinations: list[Path]) -> None:
    for destination in destinations:
        shutil.copy2(str(source_file), str(destination))


def _write_footprint_file_into_products(products: list[Path], corners: metadata.GroundCornerPoints):
    footprint_files = [product.joinpath(FOOTPRINT_FILE_NAME) for product in products if product.is_dir()]

    if not footprint_files:
        return

    _write_footprint_file(corners, footprint_files[0])

    _copy_file_to_destinations(footprint_files[0], footprint_files[1:])


def add_footprint_file_to_intermediate_products(
    core_outputs: L1CoreProcessorOutputProducts,
    additional_products: list[Path] | None = None,
):
    """Add Ground Corner Points metadata to l1 core proc outputs"""
    additional_products = additional_products or []

    dem_lut = core_outputs.output_products.get(IntermediateProductID.SLANT_DEM)
    assert dem_lut is not None

    assert core_outputs.main_slc_id
    slc_product = core_outputs.output_products.get(core_outputs.main_slc_id)
    assert slc_product is not None

    reference_metadata = _retrieve_reference_metadata(slc_product.path)

    footprint_computer = (
        partial(_compute_from_dem, dem_product=dem_lut.path)
        if dem_lut.path.exists()
        else partial(_compute_from_ellipsoid, reference_metadata=reference_metadata)
    )

    footprint = _compute_footprint(reference_metadata, footprint_computer)

    _write_footprint_file_into_products(core_outputs.list_products() + additional_products, footprint.to_metadata())
