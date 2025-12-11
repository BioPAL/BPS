# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Post Processor interface
---------------------------
"""

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.sarproduct.footprint_utils import gcp_axis_sampling
from bps.transcoder.sarproduct.generic_product import GenericProduct
from bps.transcoder.sarproduct.sarproduct import SARProduct


def _find_index(value, array):
    return np.abs(array - value).argmin()


def _read_dem_point_to_ecef(xyz_data: list[np.ndarray], *, line_ind: int, sample_ind: int) -> list[float]:
    return [
        xyz_data[0][line_ind, sample_ind],
        xyz_data[1][line_ind, sample_ind],
        xyz_data[2][line_ind, sample_ind],
    ]


def _read_dem_point_to_llh_rad(xyz_data: list[np.ndarray], *, line_ind: int, sample_ind: int) -> list[float]:
    return xyz2llh(_read_dem_point_to_ecef(xyz_data, line_ind=line_ind, sample_ind=sample_ind)).squeeze().tolist()


def read_from_dem_lut(dem_lut: GenericProduct, sample: float, line: PreciseDateTime) -> list[float]:
    """Read value from dem lut"""
    sample_ind = _find_index(sample, dem_lut.samples_axis_list[0])
    line_ind = _find_index(line, dem_lut.lines_axis_list[0])

    xyz_data = dem_lut.data_list[0:3]
    return _read_dem_point_to_llh_rad(xyz_data, line_ind=line_ind, sample_ind=sample_ind)


def read_from_dem_lut_ecef(dem_lut: GenericProduct, sample: float, line: PreciseDateTime) -> list[float]:
    """Read value from dem lut"""
    sample_ind = _find_index(sample, dem_lut.samples_axis_list[0])
    line_ind = _find_index(line, dem_lut.lines_axis_list[0])

    xyz_data = dem_lut.data_list[0:3]
    return _read_dem_point_to_ecef(xyz_data, line_ind=line_ind, sample_ind=sample_ind)


def compute_footprint_from_dem_lut(product: SARProduct, dem_lut: GenericProduct) -> list[list[float]]:
    """Compute footprint by reading DEM look up table"""
    samples_start, samples_stop, lines_start, lines_stop = product.compute_time_corners()

    def get_lat_lon_deg(line, sample):
        return np.rad2deg(read_from_dem_lut(dem_lut, line=line, sample=sample)[0:2]).tolist()

    return [
        get_lat_lon_deg(line=lines_stop, sample=samples_start),
        get_lat_lon_deg(line=lines_stop, sample=samples_stop),
        get_lat_lon_deg(line=lines_start, sample=samples_stop),
        get_lat_lon_deg(line=lines_start, sample=samples_start),
    ]


def compute_gcp_from_dem_lut(
    product: SARProduct,
    dem_lut: GenericProduct,
    samples_sub_sampling: int | None = None,
    lines_sub_sampling: int | None = None,
):
    """Compute GCP from DEM LUT"""
    range_times, azimuth_times = product.compute_time_axis()
    samples_indexes, lines_indexes = gcp_axis_sampling(
        range_times, azimuth_times, samples_sub_sampling, lines_sub_sampling
    )

    range_times = range_times[samples_indexes]
    azimuth_times = azimuth_times[lines_indexes]

    gcp_list = []
    for sample, sample_index in zip(range_times, samples_indexes):
        for line, line_index in zip(azimuth_times, lines_indexes):
            gcp_list.append(read_from_dem_lut_ecef(dem_lut, sample, line) + [int(sample_index), int(line_index)])

    return gcp_list
