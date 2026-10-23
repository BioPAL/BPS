# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
from bps.common import bps_logger
from bps.transcoder.io.translate_main_annotation_l2b_fh import read_main_annotation_l2b_fh
from bps.transcoder.sarproduct.biomass_l2bfhproduct import (
    BIOMASSL2bFHProduct,
    BIOMASSL2bFHProductMeasurement,
    BIOMASSL2bFHProductStructure,
)
from bps.transcoder.sarproduct.mph import read_product_doi
from bps.transcoder.utils.product_name import parse_l2bproduct_name
from osgeo import gdal
from pystac import Item

gdal.UseExceptions()


class BIOMASSL2bFHProductReader:
    def __init__(self, product_path: Path) -> BIOMASSL2bFHProduct:  # type: ignore
        product_folder_name = product_path.name
        product_type = product_folder_name[4:14]
        if product_type not in "FP_FH__L2B":
            bps_logger.error(
                "Not recognized input L2b Product from job order, file name not valid: {}".format(product_path)
            )

        self.product_path = product_path
        self.parsed_product_name = parse_l2bproduct_name(self.product_path.name)
        self.product: BIOMASSL2bFHProduct
        self.product_type = product_type
        self.product_structure = BIOMASSL2bFHProductStructure(self.product_path)

    def _read_mph_file(self):
        pass

    def _read_stac_file(self):
        # TBD
        item = Item.from_file(self.product_structure.stac_file)

        return item

    def _read_measurement_files(self):
        # Read measurement files
        data_dict = {}
        metadata_dict = {}
        latitude_vec = None
        longitude_vec = None

        for idx, file_path in enumerate(
            [
                self.product_structure.fh_file,
                self.product_structure.fh_quality_file,
                self.product_structure.bps_fnf_file,
                self.product_structure.heatmap_file,
                self.product_structure.acquisition_id_image_file,
            ]
        ):
            assert file_path is not None
            data_driver = gdal.Open(file_path)

            if idx == 0:
                # geotransform is [longitude start, longitude step, 0, latitude start, 0, latitude step]
                geotransform = data_driver.GetGeoTransform()

                # proj = data_driver.GetProjection()

                # RasterXSize: longitude size
                # RasterYSize: latitude size
                longitude_vec = geotransform[0] + geotransform[1] * np.arange(data_driver.RasterXSize)
                latitude_vec = geotransform[3] + geotransform[5] * np.arange(data_driver.RasterYSize)

            key = "fh"
            if "_i_bps_fnf.tiff" in file_path:
                key = "bps_fnf"
            if "_i_fhquality.tiff" in file_path:
                key = "quality"
            if "_i_heatmap.tiff" in file_path:
                key = "heat_map"
            if "i_acquisition_id_image.tiff" in file_path:
                key = "acquisition_id_image"

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

            metadata_dict[key] = BIOMASSL2bFHProductMeasurement.MetadataCOG(
                metadata_read["tileID"].replace("'", "").replace("[", "").replace("]", "").split(", "),
                [metadata_read["basinID"].replace("'", "").replace("[", "").replace("]", "").split(", ")],
                metadata_read["COMPRESSION"],
                metadata_read["TIFFTAG_IMAGEDESCRIPTION"],
                metadata_read["TIFFTAG_SOFTWARE"],
                metadata_read["TIFFTAG_DATETIME"],
            )

            # Read band
            band = data_driver.GetRasterBand(1)
            temp = band.ReadAsArray()
            if key in ["acquisition_id_image", "bps_fnf"]:
                data_dict[key] = np.zeros(
                    (temp.shape[0], temp.shape[1], data_driver.RasterCount),
                    dtype=np.uint8,
                )
                for band_idx in range(data_driver.RasterCount):
                    data_dict[key][:, :, band_idx] = data_driver.GetRasterBand(band_idx + 1).ReadAsArray()

            else:
                data_dict[key] = temp

        # Eveltually close the driver:
        data_driver = None

        assert latitude_vec is not None
        assert longitude_vec is not None
        self.measurement = BIOMASSL2bFHProductMeasurement(
            latitude_vec,
            longitude_vec,
            data_dict,
            metadata_dict,
        )

    def _read_main_annotation_file(self):
        assert self.product_structure.main_annotation_file is not None
        main_annotation_path = Path(self.product_structure.main_annotation_file)
        main_annotation = read_main_annotation_l2b_fh(main_annotation_path)

        self.main_ads_product = main_annotation.product
        self.main_ads_raster_image = main_annotation.raster_image
        self.main_ads_input_information = main_annotation.input_information
        self.main_ads_processing_parameters = main_annotation.processing_parameters

    def read(self):
        bps_logger.info(f"Reading BIOMASS L2b FH product {self.product_path.name}")
        # # Read MPH file
        # bps_logger.info("..MPH file")
        # self._read_mph_file()
        assert self.product_structure.mph_file is not None
        self.product_doi = read_product_doi(Path(self.product_structure.mph_file))

        baseline_string = self.product_structure.product_path.name[-9:-7:1]
        baseline = 0 if baseline_string == "__" else int(baseline_string)

        # # Read STAC file
        # bps_logger.info("..STAC file")
        # self._read_stac_file()

        # Read measurement files
        bps_logger.debug("..measurement files")
        self._read_measurement_files()

        # Read annotation files
        bps_logger.debug("..main annotation file")
        self._read_main_annotation_file()

        bps_logger.debug("..done")

        return BIOMASSL2bFHProduct(
            self.measurement,
            self.main_ads_product,
            self.main_ads_raster_image,
            self.main_ads_input_information,
            self.main_ads_processing_parameters,
            baseline,
            self.product_doi,
        )


class InvalidBoolTagContent(RuntimeError):
    """Raised when input bool tag content is different from true or false"""


def str_to_bool(tag: str) -> bool:
    """Safe string to bool tag content conversion"""
    tag = tag.lower()
    if tag == "true":
        return True
    if tag == "false":
        return False
    raise InvalidBoolTagContent(tag)
