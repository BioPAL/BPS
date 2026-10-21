# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
from bps.common import bps_logger
from bps.transcoder.io.translate_main_annotation_l2a_fd import read_main_annotation_l2a_fd
from bps.transcoder.io.translate_main_annotation_l2a_fh import read_main_annotation_l2a_fh
from bps.transcoder.io.translate_main_annotation_l2a_gn import read_main_annotation_l2a_gn
from bps.transcoder.io.translate_main_annotation_l2a_tfh import read_main_annotation_l2a_tfh
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aLutAdsFD,
    BIOMASSL2aLutAdsFH,
    BIOMASSL2aLutAdsGN,
    BIOMASSL2aLutAdsTOMOFH,
    BIOMASSL2aProduct,
    BIOMASSL2aProductFD,
    BIOMASSL2aProductFH,
    BIOMASSL2aProductGN,
    BIOMASSL2aProductMeasurement,
    BIOMASSL2aProductStructure,
    BIOMASSL2aProductTOMOFH,
)
from bps.transcoder.sarproduct.mph import read_product_doi
from bps.transcoder.utils.product_name import parse_l2aproduct_name
from netCDF4 import Dataset
from osgeo import gdal
from pystac import Item

gdal.UseExceptions()


class BIOMASSL2aProductReader:
    def __init__(
        self, product_path: Path
    ) -> BIOMASSL2aProductFD | BIOMASSL2aProductFH | BIOMASSL2aProductGN | BIOMASSL2aProductTOMOFH:  # type: ignore
        product_folder_name = product_path.name
        product_type = product_folder_name[4:14]
        if product_type not in ["FP_FD__L2A", "FP_FH__L2A", "FP_GN__L2A", "FP_TFH_L2A"]:
            bps_logger.error(
                "Not recognized input L2a Product from job order, file name not valid: {}".format(product_path)
            )

        self.product_path = product_path
        self.parsed_product_name = parse_l2aproduct_name(self.product_path.name)
        self.product: BIOMASSL2aProduct
        self.product_type = product_type
        self.product_structure = BIOMASSL2aProductStructure(self.product_path, self.product_type)

    def _read_mph_file(self):
        pass

    def read_stac_file(self):
        item = Item.from_file(self.product_structure.stac_file)

        return item

    def _read_measurement_files(self):
        # Read measurement files
        data_dict = {}
        metadata_dict = {}
        latitude_vec = None
        longitude_vec = None

        assert self.product_structure.measurement_files is not None
        for idx, file_path in enumerate(self.product_structure.measurement_files):
            if not Path(file_path).exists():
                raise RuntimeError(f"L2A product {file_path} file does not exist.")

            data_driver = gdal.Open(file_path)

            if idx == 0:
                # geotransform is [longitude start, longitude step, 0, latitude start, 0, latitude step]
                geotransform = data_driver.GetGeoTransform()

                # proj = data_driver.GetProjection()

                # RasterXSize: longitude size
                # RasterYSize: latitude size
                longitude_vec = geotransform[0] + geotransform[1] * np.arange(data_driver.RasterXSize)
                latitude_vec = geotransform[3] + geotransform[5] * np.arange(data_driver.RasterYSize)

            keys = ["fd"]
            if "_i_cfm.tiff" in file_path:
                keys = ["cfm"]
            if "_i_probability.tiff" in file_path:
                keys = ["probability_ofchange"]
            if "_i_fh.tiff" in file_path:
                keys = ["fh"]
            if "_i_quality.tiff" in file_path:
                keys = ["quality"]

            key = keys[0]

            if "_i_gn.tiff" in file_path:
                keys = ["HH", "VH", "VV"]
                key = "gn"

            # Read COG metadata
            metadata_read = data_driver.GetMetadata()
            metadata_read["COMPRESSION"] = data_driver.GetMetadata("IMAGE_STRUCTURE")["COMPRESSION"]

            ### TO BE REMOVED, RETRO COMPATIBILITY for OLD GEOTIFF TAGS
            if "TIFFTAG_IMAGEDESCRIPTION" not in metadata_read.keys():
                metadata_read["TIFFTAG_IMAGEDESCRIPTION"] = metadata_read["ImageDescription"]
                metadata_read["TIFFTAG_SOFTWARE"] = metadata_read["Software"]
                metadata_read["TIFFTAG_DATETIME"] = metadata_read["DateTime"]
            ###

            if "GeographicTypeGeoKey" not in metadata_read.keys():
                metadata_read["GeographicTypeGeoKey"] = 4030

            metadata_dict[key] = BIOMASSL2aProductMeasurement.MetadataCOG(
                metadata_read["Swath"],
                metadata_read["tileID"].replace("'", "").replace("[", "").replace("]", "").split(", "),
                [metadata_read["basinID"].replace("'", "").replace("[", "").replace("]", "").split(", ")],
                metadata_read["COMPRESSION"],
                metadata_read["TIFFTAG_IMAGEDESCRIPTION"],
                metadata_read["TIFFTAG_SOFTWARE"],
                metadata_read["TIFFTAG_DATETIME"],
            )

            # Read bands: GN has 3 bands, all the other only one
            band_number = data_driver.RasterCount

            for band_index, key in zip(range(band_number), keys):
                band = data_driver.GetRasterBand(band_index + 1)
                data_dict[key] = band.ReadAsArray()

        # Eveltually close the driver:
        data_driver = None
        assert latitude_vec is not None
        assert longitude_vec is not None
        self.measurement = BIOMASSL2aProductMeasurement(
            latitude_vec,
            longitude_vec,
            data_dict,
            metadata_dict,
        )

    def _read_main_annotation_file(self):
        assert self.product_structure.main_annotation_file is not None
        main_annotation_path = Path(self.product_structure.main_annotation_file)

        if self.product_type == "FP_FD__L2A":
            main_annotation = read_main_annotation_l2a_fd(main_annotation_path)
        elif self.product_type == "FP_FH__L2A":
            main_annotation = read_main_annotation_l2a_fh(main_annotation_path)
        elif self.product_type == "FP_GN__L2A":
            main_annotation = read_main_annotation_l2a_gn(main_annotation_path)
        elif self.product_type == "FP_TFH_L2A":
            main_annotation = read_main_annotation_l2a_tfh(main_annotation_path)
        else:
            raise ValueError(f"Invalid product L2a: {self.product_type}")

        self.main_ads_product = main_annotation.product
        self.main_ads_raster_image = main_annotation.raster_image
        self.main_ads_input_information = main_annotation.input_information
        self.main_ads_processing_parameters = main_annotation.processing_parameters

    def _read_lut_annotation_file(self):
        if self.product_type == "FP_FD__L2A":
            lut_class = BIOMASSL2aLutAdsFD
        elif self.product_type == "FP_FH__L2A":
            lut_class = BIOMASSL2aLutAdsFH
        elif self.product_type == "FP_GN__L2A":
            lut_class = BIOMASSL2aLutAdsGN
        elif self.product_type == "FP_TFH_L2A":
            lut_class = BIOMASSL2aLutAdsTOMOFH
        else:
            raise ValueError(f"Product type non supportato: {self.product_type}")

        ncfile_read = Dataset(self.product_structure.lut_annotation_file, mode="r")

        nc_groups = [group for group in ncfile_read.groups]
        for group_name in nc_groups:
            nc_group_vars = [var for var in ncfile_read[group_name].variables]

            for var_name in nc_group_vars:
                if var_name == "FNF":
                    lut_fnf = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "localIncidenceAngle":
                    lut_local_incidence_angle = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "numberOfAverages":
                    lut_number_of_averages = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "height":
                    lut_dem = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "latitude":
                    lut_latitude = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "longitude":
                    lut_longitude = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "elevationAngle":
                    lut_elevation_angle = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "incidenceAngle":
                    lut_incidence_angle = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "rangeTerrainSlope":
                    lut_range_dem_slope = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "azimuthTerrainSlope":
                    lut_azimuth_dem_slope = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name in [  # ACM
                    "layer1",
                    "layer2",
                    "layer3",
                    "layer4",
                    "layer5",
                    "layer6",
                    "layer7",
                    "layer8",
                    "layer9",
                ]:
                    idx = int(var_name[-1]) - 1
                    if "lut_acm" not in locals():
                        n, m = np.array(ncfile_read[group_name].variables[var_name][:]).shape
                        lut_acm = np.zeros((9, n, m))

                    lut_acm[idx, :, :] = np.array(ncfile_read[group_name].variables[var_name][:])

                if var_name == "firstSample":
                    first_sample = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "firstLine":
                    first_line = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "samplesInterval":
                    samples_interval = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "linesInterval":
                    lines_interval = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "pixelType":
                    pixelType = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "noDataValue":
                    no_data_value = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "projection":
                    projection = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "coordinateReferenceSystem":
                    coordinateReferenceSystem = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "geodeticReferenceFrame":
                    geodeticReferenceFrame = ncfile_read[group_name].variables[var_name][:][0]

                if var_name == "least_significant_digit":
                    least_significant_digit = ncfile_read[group_name].variables[var_name][:][0]

            if group_name == "FNF":
                lut_fnf_metadata = lut_class.LutMetadata(
                    first_sample,
                    first_line,
                    samples_interval,
                    lines_interval,
                    pixelType,
                    no_data_value,
                    projection,
                    coordinateReferenceSystem,
                    geodeticReferenceFrame,
                )

            if group_name == "ACM":
                lut_acm_metadata = lut_class.LutMetadata(
                    first_sample,
                    first_line,
                    samples_interval,
                    lines_interval,
                    pixelType,
                    no_data_value,
                    projection,
                    coordinateReferenceSystem,
                    geodeticReferenceFrame,
                    least_significant_digit,
                )
            if group_name == "numberOfAverages":
                lut_number_of_averages_metadata = lut_class.LutMetadata(
                    first_sample,
                    first_line,
                    samples_interval,
                    lines_interval,
                    pixelType,
                    no_data_value,
                    projection,
                    coordinateReferenceSystem,
                    geodeticReferenceFrame,
                )
            if group_name in [
                "height",
                "latitude",
                "longitude",
                "incidenceAngle",
                "elevationAngle",
                "rangeTerrainSlope",
                "azimuthTerrainSlope",
                "localIncidenceAngle",  # GN: incidence - slope
            ]:
                lut_geometric_metadata = lut_class.LutMetadata(
                    first_sample,
                    first_line,
                    samples_interval,
                    lines_interval,
                    pixelType,
                    no_data_value,
                    projection,
                    coordinateReferenceSystem,
                    geodeticReferenceFrame,
                    least_significant_digit,
                )

        if self.product_type == "FP_FD__L2A":
            self.lut_ads = lut_class(
                lut_fnf,
                lut_acm,
                lut_number_of_averages,
                lut_latitude,
                lut_longitude,
                lut_dem,
                lut_incidence_angle,
                lut_elevation_angle,
                lut_range_dem_slope,
                lut_azimuth_dem_slope,
                lut_fnf_metadata,
                lut_acm_metadata,
                lut_number_of_averages_metadata,
                lut_geometric_metadata,
            )

        elif self.product_type == "FP_FH__L2A":
            self.lut_ads = lut_class(
                lut_fnf,
                lut_latitude,
                lut_longitude,
                lut_dem,
                lut_incidence_angle,
                lut_elevation_angle,
                lut_range_dem_slope,
                lut_azimuth_dem_slope,
                lut_fnf_metadata,
                lut_geometric_metadata,
            )

        elif self.product_type == "FP_GN__L2A":
            self.lut_ads = lut_class(
                lut_fnf,
                lut_local_incidence_angle,
                lut_latitude,
                lut_longitude,
                lut_dem,
                lut_incidence_angle,
                lut_elevation_angle,
                lut_range_dem_slope,
                lut_azimuth_dem_slope,
                lut_fnf_metadata,
                lut_geometric_metadata,
            )

        elif self.product_type == "FP_TFH_L2A":
            self.lut_ads = lut_class(
                lut_fnf,
                lut_latitude,
                lut_longitude,
                lut_dem,
                lut_incidence_angle,
                lut_elevation_angle,
                lut_range_dem_slope,
                lut_azimuth_dem_slope,
                lut_fnf_metadata,
                lut_geometric_metadata,
            )
        ncfile_read.close()

    def read(self):
        bps_logger.info(f"Reading BIOMASS L2a product {self.product_path.name}")

        # # Read MPH file
        # bps_logger.info("..MPH file")
        # self._read_mph_file()
        assert self.product_structure.mph_file is not None
        self.product_doi = read_product_doi(Path(self.product_structure.mph_file))

        # # Read STAC file
        # bps_logger.info("..STAC file")
        # self._read_stac_file()

        # Read measurement files
        bps_logger.debug("..measurement files")
        self._read_measurement_files()

        # Read annotation files
        bps_logger.debug("..main annotation file")
        self._read_main_annotation_file()
        bps_logger.debug("..LUT annotation file")
        self._read_lut_annotation_file()

        bps_logger.debug("..done")

        l2a_product = None
        if self.product_type == "FP_FD__L2A":
            l2a_product = BIOMASSL2aProductFD(
                self.measurement,
                self.main_ads_product,
                self.main_ads_raster_image,
                self.main_ads_input_information,
                self.main_ads_processing_parameters,  # type: ignore
                self.lut_ads,  # type: ignore
                self.product_doi,
            )
        if self.product_type == "FP_FH__L2A":
            l2a_product = BIOMASSL2aProductFH(
                self.measurement,
                self.main_ads_product,
                self.main_ads_raster_image,
                self.main_ads_input_information,
                self.main_ads_processing_parameters,  # type: ignore
                self.lut_ads,  # type: ignore
                self.product_doi,
            )
        if self.product_type == "FP_GN__L2A":
            l2a_product = BIOMASSL2aProductGN(
                self.measurement,
                self.main_ads_product,
                self.main_ads_raster_image,
                self.main_ads_input_information,
                self.main_ads_processing_parameters,  # type: ignore
                self.lut_ads,  # type: ignore
                self.product_doi,
            )
        if self.product_type == "FP_TFH_L2A":
            l2a_product = BIOMASSL2aProductFH(
                self.measurement,
                self.main_ads_product,
                self.main_ads_raster_image,
                self.main_ads_input_information,
                self.main_ads_processing_parameters,  # type: ignore
                self.lut_ads,  # type: ignore
                self.product_doi,
            )

        assert l2a_product is not None
        # Restore read product name (BIOMASSL2aProductXX constructors set a new name)
        l2a_product.name = self.product_path.name
        return l2a_product


def str_to_bool(tag: str) -> bool:
    """Safe string to bool tag content conversion"""
    tag = tag.lower()
    if tag == "true":
        return True
    if tag == "false":
        return False
    raise InvalidBoolTagContent(tag)


class InvalidBoolTagContent(RuntimeError):
    """Raised when input bool tag content is different from true or false"""
