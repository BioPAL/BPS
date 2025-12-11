# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
from bps.common import bps_logger
from bps.common.io import translate_common
from bps.common.io.parsing import parse
from bps.transcoder.io import common_annotation_models_l2, main_annotation_models_l2b_fd
from bps.transcoder.sarproduct.biomass_l2bfdproduct import (
    BIOMASSL2bFDMainADSInputInformation,
    BIOMASSL2bFDMainADSproduct,
    BIOMASSL2bFDMainADSRasterImage,
    BIOMASSL2bFDProduct,
    BIOMASSL2bFDProductMeasurement,
    BIOMASSL2bFDProductStructure,
    BIOMASSL2bMainADSProcessingParametersFD,
)
from bps.transcoder.sarproduct.mph import read_product_doi
from bps.transcoder.utils.product_name import parse_l2bproduct_name
from osgeo import gdal
from pystac import Item

gdal.UseExceptions()


class BIOMASSL2bFDProductReader:
    def __init__(self, product_path: Path) -> BIOMASSL2bFDProduct:  # type: ignore
        product_folder_name = product_path.name
        product_type = product_folder_name[4:14]
        if product_type not in "FP_FD__L2B":
            bps_logger.error(
                "Not recognized input L2b Product from job order, file name not valid: {}".format(product_path)
            )

        self.product_path = product_path
        self.parsed_product_name = parse_l2bproduct_name(self.product_path.name)
        self.product: BIOMASSL2bFDProduct
        self.product_type = product_type
        self.product_structure = BIOMASSL2bFDProductStructure(self.product_path)

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
                self.product_structure.fd_file,
                self.product_structure.cfm_file,
                self.product_structure.probability_file,
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

            key = "fd"
            if "_i_cfm.tiff" in file_path:
                key = "cfm"
            if "_i_probability.tiff" in file_path:
                key = "probability_ofchange"
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

            if key == "heat_map":
                metadata_key_list = ["heat_map_contributing", "heat_map_agreeing"]
            else:
                metadata_key_list = [key]
            for key_md in metadata_key_list:
                metadata_dict[key_md] = BIOMASSL2bFDProductMeasurement.MetadataCOG(
                    metadata_read["tileID"].replace("'", "").replace("[", "").replace("]", "").split(", "),
                    [metadata_read["basinID"].replace("'", "").replace("[", "").replace("]", "").split(", ")],
                    metadata_read["COMPRESSION"],
                    metadata_read["TIFFTAG_IMAGEDESCRIPTION"],
                    metadata_read["TIFFTAG_SOFTWARE"],
                    metadata_read["TIFFTAG_DATETIME"],
                )

            # Read band
            if key == "acquisition_id_image":
                band = data_driver.GetRasterBand(1)
                temp = band.ReadAsArray()
                data_dict[key] = np.zeros(
                    (temp.shape[0], temp.shape[1], data_driver.RasterCount),
                    dtype=np.uint8,
                )
                for band_idx in range(data_driver.RasterCount):
                    data_dict[key][:, :, band_idx] = data_driver.GetRasterBand(band_idx + 1).ReadAsArray()
            elif key == "heat_map":
                band1 = data_driver.GetRasterBand(1)
                data_dict["heat_map_contributing"] = band1.ReadAsArray()
                band2 = data_driver.GetRasterBand(2)
                data_dict["heat_map_agreeing"] = band2.ReadAsArray()
            else:
                band = data_driver.GetRasterBand(1)
                temp = band.ReadAsArray()
                data_dict[key] = temp

        # Eveltually close the driver:
        data_driver = None

        assert latitude_vec is not None
        assert longitude_vec is not None
        self.measurement = BIOMASSL2bFDProductMeasurement(
            latitude_vec,
            longitude_vec,
            data_dict,
            metadata_dict,
        )

    def _read_main_annotation_file(self):
        # Read main annotation file
        assert self.product_structure.main_annotation_file is not None
        main_annotation_path = Path(self.product_structure.main_annotation_file)
        main_annotation_model: main_annotation_models_l2b_fd.MainAnnotation = parse(
            main_annotation_path.read_text(encoding="utf-8"),
            main_annotation_models_l2b_fd.MainAnnotation,
        )

        assert main_annotation_model is not None
        assert main_annotation_model.product is not None

        assert main_annotation_model.product.mission is not None
        assert main_annotation_model.product.tile_id is not None
        assert main_annotation_model.product.basin_id is not None
        assert main_annotation_model.product.product_type is not None
        assert main_annotation_model.product.start_time is not None
        assert main_annotation_model.product.stop_time is not None
        assert main_annotation_model.product.radar_carrier_frequency is not None
        assert main_annotation_model.product.mission_phase_id is not None
        assert main_annotation_model.product.sensor_mode is not None
        assert main_annotation_model.product.global_coverage_id is not None

        mission = main_annotation_model.product.mission.value
        tile_id_list = [main_annotation_model.product.tile_id]  # one element list
        basin_id_list = main_annotation_model.product.basin_id.id
        product_type = main_annotation_model.product.product_type.value
        start_time = translate_common.translate_datetime(main_annotation_model.product.start_time)
        stop_time = translate_common.translate_datetime(main_annotation_model.product.stop_time)
        radar_carrier_frequency = main_annotation_model.product.radar_carrier_frequency.value
        mission_phase_id = main_annotation_model.product.mission_phase_id.value
        sensor_mode = main_annotation_model.product.sensor_mode.value
        global_coverage_id = self.parsed_product_name.coverage

        baseline_string = str(Path(self.product_structure.main_annotation_file).parent.parent)[-9:-7:1]
        if baseline_string == "__":
            baseline = 0
        else:
            baseline = int(baseline_string)

        assert radar_carrier_frequency is not None
        assert global_coverage_id is not None
        self.main_ads_product = BIOMASSL2bFDMainADSproduct(
            mission,
            tile_id_list,  # one element list
            basin_id_list,
            product_type,
            start_time,
            stop_time,
            radar_carrier_frequency,
            mission_phase_id,
            sensor_mode,
            global_coverage_id,
            baseline,
        )

        assert main_annotation_model.raster_image is not None

        assert main_annotation_model.raster_image.first_latitude_value is not None
        assert main_annotation_model.raster_image.first_longitude_value is not None
        assert main_annotation_model.raster_image.latitude_spacing is not None
        assert main_annotation_model.raster_image.longitude_spacing is not None
        assert main_annotation_model.raster_image.number_of_samples is not None
        assert main_annotation_model.raster_image.number_of_lines is not None
        assert main_annotation_model.raster_image.projection is not None
        assert main_annotation_model.raster_image.datum is not None
        assert main_annotation_model.raster_image.pixel_representation is not None
        assert main_annotation_model.raster_image.pixel_type is not None
        assert main_annotation_model.raster_image.no_data_value is not None

        first_latitude_value = main_annotation_model.raster_image.first_latitude_value.value
        first_longitude_value = main_annotation_model.raster_image.first_longitude_value.value
        latitude_spacing = main_annotation_model.raster_image.latitude_spacing.value
        longitude_spacing = main_annotation_model.raster_image.longitude_spacing.value
        number_of_samples = main_annotation_model.raster_image.number_of_samples
        number_of_lines = main_annotation_model.raster_image.number_of_lines
        projection = main_annotation_model.raster_image.projection.value
        datum = main_annotation_model.raster_image.datum
        pixel_representation = main_annotation_model.raster_image.pixel_representation
        pixel_type = main_annotation_model.raster_image.pixel_type
        no_data_value = main_annotation_model.raster_image.no_data_value

        assert main_annotation_model.raster_image.footprint is not None
        assert first_latitude_value is not None
        assert first_longitude_value is not None
        assert latitude_spacing is not None
        assert longitude_spacing is not None
        assert number_of_samples is not None
        assert number_of_lines is not None

        footprint = main_annotation_model.raster_image.footprint.value.split(" ")
        footprint = [float(v) for v in footprint]

        self.main_ads_raster_image = BIOMASSL2bFDMainADSRasterImage(
            footprint,
            first_latitude_value,
            first_longitude_value,
            latitude_spacing,
            longitude_spacing,
            number_of_samples,
            number_of_lines,
            projection,
            datum,
            pixel_representation,
            pixel_type,
            no_data_value,
        )

        assert main_annotation_model.input_information is not None
        self.main_ads_input_information = BIOMASSL2bFDMainADSInputInformation(
            common_annotation_models_l2.InputInformationL2BL3ListType(
                main_annotation_model.input_information.l2a_inputs,
                len(main_annotation_model.input_information.l2a_inputs),
            )
        )

        assert main_annotation_model.processing_parameters is not None
        assert main_annotation_model.processing_parameters.minimum_l2a_coverage is not None

        processor_version = main_annotation_model.processing_parameters.processor_version
        product_generation_time = main_annotation_model.processing_parameters.product_generation_time
        compression_options = main_annotation_model.processing_parameters.compression_options
        minumum_l2a_coverage = main_annotation_model.processing_parameters.minimum_l2a_coverage

        assert processor_version is not None
        assert product_generation_time is not None
        assert compression_options is not None
        assert minumum_l2a_coverage is not None
        self.main_ads_processing_parameters = BIOMASSL2bMainADSProcessingParametersFD(
            processor_version,
            product_generation_time,  # type: ignore
            compression_options,
            minumum_l2a_coverage,
        )

    def read(self):
        bps_logger.info(f"Reading BIOMASS L2b FD product {self.product_path.name}")
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

        bps_logger.debug("..done")

        return BIOMASSL2bFDProduct(
            self.measurement,
            self.main_ads_product,
            self.main_ads_raster_image,
            self.main_ads_input_information,
            self.main_ads_processing_parameters,
            self.product_doi,
        )
