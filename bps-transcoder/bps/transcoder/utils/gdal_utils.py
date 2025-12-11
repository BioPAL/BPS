# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""GDAL utilities"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from arepytools.geometry.conversions import llh2xyz, xyz2llh
from osgeo import gdal

gdal.UseExceptions()


def get_wgs84_spatial_reference(epsg_code: int = 4979) -> gdal.osr.SpatialReference:
    """Retrieve the wgs84 spatial reference"""
    spatial_reference: gdal.osr.SpatialReference = gdal.osr.SpatialReference()
    spatial_reference.ImportFromEPSG(epsg_code)
    return spatial_reference


def get_wgs84_wkt(epsg_code: int = 4979) -> str:
    """Get the string related to the wgs84 spatial reference in wkt format (well known text)"""
    return get_wgs84_spatial_reference(epsg_code).ExportToWkt()


@dataclass
class GeotiffConf:
    """Configuration for geotiff writing"""

    compression_schema: str = "LERC"

    max_z_error: float = 0.0001

    zstd_level: int | None = None

    nodata_value: float = -9999.0

    block_size: int = 512

    output_driver: Literal["COG", "GTiff"] = "GTiff"

    overview_resampling: str = "NEAREST"

    overview_levels: list[int] = field(default_factory=lambda: [2, 5, 10, 20])

    epsg_code: int = 4979

    gdal_num_threads: int = 1


@dataclass
class GeotiffMetadata:
    """Metadata for geotiff annotations"""

    creation_date: str
    """Creation data"""
    swath: str | None
    """Swath name"""
    software: str
    """Processing software"""
    matrix_representation: Literal["SCATTERING", "SYMMETRIZED_SCATTERING"] | None
    """Matrix representation"""
    description: str
    """Product description"""

    polarizations: list[str] | None
    """Polarizations list e.g. ["HH", "HV", "VH", "VV"] (meaningful for SCATTERING or SYMMETRIZED_SCATTERING matrix representation)"""

    additional_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class GeotiffData:
    """Helper class that store the content of BIOMASS Geotiff data."""

    data_list: list[np.ndarray]
    """The raster data stored in the Geotiff."""

    nodata_values: list[float]
    """The value assigned to pixels that miss data."""

    gcp_list: list[list[float]]
    """The ground control points, expressed in ECEF coordinates."""

    geotransform: np.ndarray | None = None
    """The geo-transformation."""


def _retrieve_polarimetric_interpretation(
    metadata: GeotiffMetadata,
) -> list[str] | None:
    """Retrieve polarimetric interpretation based on matrix_representation"""

    if metadata.matrix_representation is None:
        return None

    if metadata.matrix_representation in ("SCATTERING", "SYMMETRIZED_SCATTERING"):
        if not metadata.polarizations:
            raise RuntimeError(
                "Polarizations missing, it is required for "
                + "'SCATTERING' and 'SYMMETRIZED_SCATTERING' matrix representation"
            )

    return metadata.polarizations


def _retrieve_data_type(data: np.ndarray):
    if data.dtype == np.float32:
        return gdal.GDT_Float32
    elif data.dtype == np.uint8:
        return gdal.GDT_Byte
    else:
        raise ValueError(f"data type not valid: {data.dtype}")


def write_geotiff(
    output_geotiff_file: Path,
    data_list: list[np.ndarray],
    nodata_mask: np.ndarray | None,
    ecef_gcp_list: list[list[float]] | None,
    geotiff_metadata: GeotiffMetadata,
    geotiff_conf: GeotiffConf,
    geotransform: list[float] | None = None,
):
    """Write one raster file"""
    number_of_bands = len(data_list)
    if number_of_bands == 0:
        raise RuntimeError("Empty data_list")

    dataset_shape = data_list[0].shape
    if len(dataset_shape) != 2:
        raise RuntimeError(f"Unexpected dataset shape: {dataset_shape}")

    for data in data_list:
        if data.shape != dataset_shape:
            raise RuntimeError(f"Unexpected data shape: {data.shape} != {dataset_shape}")

    data_type = _retrieve_data_type(data_list[0])
    driver: gdal.Driver = gdal.GetDriverByName("MEM")
    mem_dataset: gdal.Dataset = driver.Create(
        "",
        xsize=dataset_shape[1],
        ysize=dataset_shape[0],
        bands=number_of_bands,
        eType=data_type,
    )
    mem_dataset.SetSpatialRef(get_wgs84_spatial_reference(geotiff_conf.epsg_code))

    polarimetric_interpretations = _retrieve_polarimetric_interpretation(geotiff_metadata)

    if polarimetric_interpretations is None:
        polarimetric_interpretations = number_of_bands * [None]
    for band_index, (band_data, polarimetric_interpretation) in enumerate(zip(data_list, polarimetric_interpretations)):
        if nodata_mask is not None:
            band_data[nodata_mask] = geotiff_conf.nodata_value
        gdal_band_index = band_index + 1
        gdal_band: gdal.Band = mem_dataset.GetRasterBand(gdal_band_index)
        gdal_band.WriteArray(band_data)
        gdal_band.SetNoDataValue(geotiff_conf.nodata_value)
        if polarimetric_interpretation is not None:
            gdal_band.SetMetadataItem("POLARIMETRIC_INTERP", polarimetric_interpretation)
        del gdal_band

    raster_metadata = {
        "TIFFTAG_SOFTWARE": geotiff_metadata.software,
        "TIFFTAG_DATETIME": geotiff_metadata.creation_date,
        "TIFFTAG_IMAGEDESCRIPTION": geotiff_metadata.description,
        "MAX_Z_ERROR": str(geotiff_conf.max_z_error),
    }
    raster_metadata.update(geotiff_metadata.additional_metadata)

    if geotiff_metadata.swath is not None:
        raster_metadata["Swath"] = geotiff_metadata.swath

    if geotiff_metadata.matrix_representation is not None:
        raster_metadata["MATRIX_REPRESENTATION"] = geotiff_metadata.matrix_representation
        if geotiff_metadata.matrix_representation == "SCATTERING":
            assert geotiff_metadata.polarizations is not None
            raster_metadata["PolarisationsSequence"] = " ".join(geotiff_metadata.polarizations)

    mem_dataset.SetDescription(geotiff_metadata.description)

    if ecef_gcp_list is not None:
        gcp_list = [_to_gdal_gcp(gcp) for gcp in ecef_gcp_list]
        longitudes = [gcp.GCPX for gcp in gcp_list]
        if max(longitudes) - min(longitudes) > 180:
            for gcp in gcp_list:
                if gcp.GCPX < 0:
                    gcp.GCPX += 360.0

        mem_dataset.SetGCPs(gcp_list, mem_dataset.GetProjection())
    elif geotransform is not None:
        mem_dataset.SetGeoTransform(geotransform)

    mem_dataset.SetMetadata(raster_metadata)

    creation_options = [
        f"COMPRESS={geotiff_conf.compression_schema}",
        f"MAX_Z_ERROR={geotiff_conf.max_z_error}",
        f"NUM_THREADS={geotiff_conf.gdal_num_threads}",
    ]

    if geotiff_conf.output_driver == "COG":
        driver_cog: gdal.Driver = gdal.GetDriverByName("COG")

        creation_options.extend(
            [
                f"BLOCKSIZE={geotiff_conf.block_size}",
                f"OVERVIEW_RESAMPLING={geotiff_conf.overview_resampling}",
            ]
        )
        cog_dataset = driver_cog.CreateCopy(str(output_geotiff_file), mem_dataset, options=creation_options)
        cog_dataset.FlushCache()
        del cog_dataset
    elif geotiff_conf.output_driver == "GTiff":
        mem_dataset.BuildOverviews(geotiff_conf.overview_resampling, geotiff_conf.overview_levels)

        driver_gtiff: gdal.Driver = gdal.GetDriverByName("GTiff")

        creation_options.extend(
            [
                f"BLOCKXSIZE={geotiff_conf.block_size}",
                f"BLOCKYSIZE={geotiff_conf.block_size}",
                "INTERLEAVE=BAND",
                "Tiled=YES",
                "COPY_SRC_OVERVIEWS=YES",
            ]
        )
        if geotiff_conf.zstd_level is not None:
            creation_options.append(f"ZSTD_LEVEL={geotiff_conf.zstd_level}")

        gtiff_dataset: gdal.Dataset = driver_gtiff.CreateCopy(
            str(output_geotiff_file),
            mem_dataset,
            options=creation_options,
        )

        gtiff_dataset.FlushCache()
        del gtiff_dataset

    mem_dataset.FlushCache()
    del mem_dataset


def read_geotiff(geotiff_file: Path, skip_data: bool = False) -> GeotiffData:
    """
    Read a GeoTIFF file of a BIOMASS product.

    Parameters
    ----------
    geotiff_file: Path
        The input GeoTIFF file.

    skip_data: bool
        Do not read the raster data. It defaults to False.

    Raises
    ------
    FileNotFoundError

    Return
    ------
    GeotiffData
        The content of the Geotiff file.

    """
    if not Path(geotiff_file).is_file():
        raise FileNotFoundError(geotiff_file)

    ds = gdal.Open(str(geotiff_file))

    data_list = []
    nodata_values = []

    # Optionally, read the data and no-data values.
    if not skip_data:
        for band_index in range(ds.RasterCount):
            raster_band = ds.GetRasterBand(band_index + 1)
            data_list.append(raster_band.ReadAsArray())
            nodata_values.append(raster_band.GetNoDataValue())

    # Read the list of ground control points.
    #
    # NOTE: In BIOMASS products, the GCP are assigned IDs '1', '2', ... so it
    # is enough to sort them to be sure that when the will be written by
    # write_geotiff, they will be assigned the same IDs.
    gcp_list = [_from_gdal_gcp(gcp) for gcp in sorted(ds.GetGCPs(), key=lambda p: int(p.Id))]

    return GeotiffData(
        data_list=data_list,
        nodata_values=nodata_values,
        gcp_list=gcp_list,
        geotransform=np.array(ds.GetGeoTransform()),
    )


def _to_gdal_gcp(gcp: list[float]) -> gdal.GCP:
    """
    Convert a GCP to a gdal object.

    Parameters
    ----------
    gcp: list[float]
        A GCP encoded as [X, Y, Z, line, sample].

    Return
    ------
    gdal.GCP
        The serializable gdal object with Lon/Lat/Height as
        X/Y/Z members.

    """
    llh = xyz2llh([gcp[0], gcp[1], gcp[2]]).squeeze()
    return gdal.GCP(np.rad2deg(llh[1]), np.rad2deg(llh[0]), llh[2], gcp[3], gcp[4])


def _from_gdal_gcp(gcp: gdal.GCP) -> list[float]:
    """
    Convert a gdal GCP object to internal GCP format. We expect
    the gdal GCP object to have been serialized so that X/Y/Z
    components store respectively Lon/Lat/Height.

    Parameters
    ----------
    gcp: gdal.GCP
        A serializable gdal GCP object.

    Return
    ------
    list[float]
        A GCP encoded as [X, Y, Z, line, sample].

    """
    xyz = llh2xyz([np.deg2rad(gcp.GCPY), np.deg2rad(gcp.GCPX), gcp.GCPZ]).squeeze()
    return [xyz[0], xyz[1], xyz[2], gcp.GCPPixel, gcp.GCPLine]
