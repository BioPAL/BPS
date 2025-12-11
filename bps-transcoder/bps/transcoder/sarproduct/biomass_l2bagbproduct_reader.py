# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from bps.common import bps_logger
from bps.common.io import translate_common
from bps.common.io.parsing import parse
from bps.transcoder.io import (
    common_annotation_models_l2,
    main_annotation_models_l2b_agb,
)
from bps.transcoder.sarproduct.biomass_l2bagbproduct import (
    BIOMASSL2bAGBMainADSInputInformation,
    BIOMASSL2bAGBMainADSproduct,
    BIOMASSL2bAGBMainADSRasterImage,
    BIOMASSL2bAGBProduct,
    BIOMASSL2bAGBProductMeasurement,
    BIOMASSL2bAGBProductStructure,
    BIOMASSL2bMainADSProcessingParametersAGB,
    agbTileStatus,
)
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES, get_metadata
from bps.transcoder.utils.product_name import parse_l2bproduct_name
from osgeo import gdal
from pystac import Item

gdal.UseExceptions()


class BIOMASSL2bAGBProductReader:
    def __init__(self, product_path: Path) -> BIOMASSL2bAGBProduct:  # type: ignore
        product_folder_name = product_path.name
        product_type = product_folder_name[4:14]
        if product_type not in "FP_AGB_L2B":
            bps_logger.error(
                "Not recognized input L2b product from job order, file name not valid: {}".format(product_path)
            )

        self.product_path = product_path
        self.parsed_product_name = parse_l2bproduct_name(self.product_path.name)
        self.product: BIOMASSL2bAGBProduct
        self.product_type = product_type
        self.product_structure = BIOMASSL2bAGBProductStructure(self.product_path)

    def _read_mph_file(self):
        # To be integrated if needed.
        assert self.product_structure.mph_file is not None
        tree = ET.parse(self.product_structure.mph_file)
        root = tree.getroot()

        # phenomenon_time = root.find("om:phenomenonTime", MPH_NAMESPACES)
        # if phenomenon_time:
        #     time_period = phenomenon_time.find("gml:TimePeriod", MPH_NAMESPACES)
        #     self.product.phenomenon_time_start = PreciseDateTime.fromisoformat(
        #         time_period.find("gml:beginPosition", MPH_NAMESPACES).text
        #     )
        #     self.product.phenomenon_time_stop = PreciseDateTime.fromisoformat(
        #         time_period.find("gml:endPosition", MPH_NAMESPACES).text
        #     )

        # result_time = root.find("om:resultTime", MPH_NAMESPACES)
        # if result_time:
        #     time_instant = result_time.find("gml:TimeInstant", MPH_NAMESPACES)
        #     self.product.result_time = PreciseDateTime.fromisoformat(
        #         time_instant.find("gml:timePosition", MPH_NAMESPACES).text
        #     )

        # valid_time = root.find("om:validTime", MPH_NAMESPACES)
        # if valid_time:
        #     time_period = valid_time.find("gml:TimePeriod", MPH_NAMESPACES)
        #     self.product.valid_time_start = PreciseDateTime.fromisoformat(
        #         time_period.find("gml:beginPosition", MPH_NAMESPACES).text
        #     )
        #     self.product.valid_time_stop = PreciseDateTime.fromisoformat(
        #         time_period.find("gml:endPosition", MPH_NAMESPACES).text
        #     )

        metadata_property = root.find("eop:metaDataProperty", MPH_NAMESPACES)
        if metadata_property:
            earth_obs_md = metadata_property.find("bio:EarthObservationMetaData", MPH_NAMESPACES)
            self.agb_tile_status = agbTileStatus(earth_obs_md.find("bio:agbTileStatus", MPH_NAMESPACES).text)
            self.agb_tile_iteration = np.uint8(earth_obs_md.find("bio:agbTileIteration", MPH_NAMESPACES).text)

        self.product_doi = get_metadata(root).doi

    def _read_stac_file(self):
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
                self.product_structure.agb_file,
                self.product_structure.agb_standard_deviation_file,
                self.product_structure.heatmap_file,
                self.product_structure.acquisition_id_image_file,
                self.product_structure.bps_fnf_file,
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

            key = "agb"
            if "_i_agb_std_dev.tiff" in file_path:
                key = "agbstandard_deviation"
            if "_i_heatmap.tiff" in file_path:
                key = "heat_map"
            if "i_acquisition_id_image.tiff" in file_path:
                key = "acquisition_id_image"
            if "i_bps_fnf.tiff" in file_path:
                key = "bps_fnf"

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

            metadata_dict[key] = BIOMASSL2bAGBProductMeasurement.MetadataCOG(
                metadata_read["tileID"].replace("'", "").replace("[", "").replace("]", "").split(", "),
                [metadata_read["basinID"].replace("'", "").replace("[", "").replace("]", "").split(", ")],
                metadata_read["COMPRESSION"],
                metadata_read["TIFFTAG_IMAGEDESCRIPTION"],
                metadata_read["TIFFTAG_SOFTWARE"],
                metadata_read["TIFFTAG_DATETIME"],
            )

            # Read band
            if key == "heat_map":
                band1 = data_driver.GetRasterBand(1)
                data_dict["heat_map"] = band1.ReadAsArray()
                band2 = data_driver.GetRasterBand(2)
                data_dict["heat_map_ref_data"] = band2.ReadAsArray()
                band3 = data_driver.GetRasterBand(3)
                data_dict["heat_map_additional_ref_data"] = band3.ReadAsArray()
            else:
                # AGB, AGB std and acquisition_id_image
                band = data_driver.GetRasterBand(1)
                temp = band.ReadAsArray()
                data_dict[key] = temp

        # Eveltually close the driver:
        data_driver = None

        assert latitude_vec is not None
        assert longitude_vec is not None
        self.measurement = BIOMASSL2bAGBProductMeasurement(
            latitude_vec,
            longitude_vec,
            data_dict,
            metadata_dict,
        )

    def _read_main_annotation_file(self):
        # Read main annotation file
        assert self.product_structure.main_annotation_file is not None
        main_annotation_path = Path(self.product_structure.main_annotation_file)
        main_annotation_model: main_annotation_models_l2b_agb.MainAnnotation = parse(
            main_annotation_path.read_text(encoding="utf-8"),
            main_annotation_models_l2b_agb.MainAnnotation,
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

        contributing_tiles = main_annotation_model.product.contributing_tiles
        cal_ab_coverage_per_tile = main_annotation_model.product.cal_abcoverage_per_tile
        gncoverage_per_tile = main_annotation_model.product.gncoverage_per_tile

        assert radar_carrier_frequency is not None
        assert global_coverage_id is not None
        assert contributing_tiles is not None
        assert cal_ab_coverage_per_tile is not None
        assert gncoverage_per_tile is not None
        self.main_ads_product = BIOMASSL2bAGBMainADSproduct(
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
            contributing_tiles,
            cal_ab_coverage_per_tile,
            gncoverage_per_tile,
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
        assert main_annotation_model.processing_parameters is not None
        assert main_annotation_model.processing_parameters.use_constant_n is not None

        footprint = main_annotation_model.raster_image.footprint.value.split(" ")
        footprint = [float(v) for v in footprint]

        self.main_ads_raster_image = BIOMASSL2bAGBMainADSRasterImage(
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
        self.main_ads_input_information = BIOMASSL2bAGBMainADSInputInformation(
            common_annotation_models_l2.InputInformationL2BL3ListType(
                main_annotation_model.input_information.l2a_inputs,
                len(main_annotation_model.input_information.l2a_inputs),
            )
        )

        assert main_annotation_model.processing_parameters is not None
        assert main_annotation_model.processing_parameters.minimum_l2a_coverage is not None
        assert main_annotation_model.processing_parameters.forest_masking_flag is not None
        processor_version = main_annotation_model.processing_parameters.processor_version
        product_generation_time = main_annotation_model.processing_parameters.product_generation_time
        forest_masking_flag = str_to_bool(main_annotation_model.processing_parameters.forest_masking_flag)
        minumum_l2a_coverage = main_annotation_model.processing_parameters.minimum_l2a_coverage

        rejected_landcover_classes = main_annotation_model.processing_parameters.rejected_landcover_classes
        backscatter_limits = main_annotation_model.processing_parameters.backscatter_limits
        angle_limits = main_annotation_model.processing_parameters.angle_limits
        mean_agblimits = main_annotation_model.processing_parameters.mean_agblimits
        std_agblimits = main_annotation_model.processing_parameters.std_agblimits
        relative_agblimits = main_annotation_model.processing_parameters.relative_agblimits
        reference_selection = main_annotation_model.processing_parameters.reference_selection
        indexing_l = main_annotation_model.processing_parameters.indexing_l
        indexing_a = main_annotation_model.processing_parameters.indexing_a
        indexing_n = main_annotation_model.processing_parameters.indexing_n
        use_constant_n = str_to_bool(main_annotation_model.processing_parameters.use_constant_n)
        values_constant_n = main_annotation_model.processing_parameters.values_constant_n
        regression_solver = main_annotation_model.processing_parameters.regression_solver
        regression_matrix_subsampling_factor = (
            main_annotation_model.processing_parameters.regression_matrix_subsampling_factor
        )
        minimum_percentage_of_fillable_voids = (
            main_annotation_model.processing_parameters.minimum_percentage_of_fillable_voids
        )
        estimated_parameters = main_annotation_model.processing_parameters.estimated_parameters
        compression_options = main_annotation_model.processing_parameters.compression_options

        assert processor_version is not None
        assert product_generation_time is not None
        assert minumum_l2a_coverage is not None
        assert rejected_landcover_classes is not None
        assert backscatter_limits is not None
        assert angle_limits is not None
        assert mean_agblimits is not None
        assert std_agblimits is not None
        assert relative_agblimits is not None
        assert reference_selection is not None
        assert indexing_l is not None
        assert indexing_a is not None
        assert indexing_n is not None
        assert use_constant_n is not None
        assert values_constant_n is not None
        assert regression_solver is not None
        assert minimum_percentage_of_fillable_voids is not None
        assert regression_matrix_subsampling_factor is not None
        assert estimated_parameters is not None
        assert compression_options is not None

        self.main_ads_processing_parameters = BIOMASSL2bMainADSProcessingParametersAGB(
            processor_version,
            product_generation_time,  # type: ignore
            forest_masking_flag,
            minumum_l2a_coverage,
            rejected_landcover_classes,
            backscatter_limits,
            angle_limits,
            mean_agblimits,
            std_agblimits,
            relative_agblimits,
            reference_selection,
            indexing_l.value,
            indexing_a.value,
            indexing_n.value,
            use_constant_n,
            values_constant_n,
            regression_solver,
            minimum_percentage_of_fillable_voids,
            regression_matrix_subsampling_factor,
            estimated_parameters,
            compression_options,
        )

    def read(self):
        bps_logger.info(f"Reading BIOMASS L2b AGB product {self.product_path.name}")
        # # Read MPH file
        bps_logger.debug("..MPH file")
        self._read_mph_file()

        # # Read STAC file
        # bps_logger.debug("..STAC file")
        # self._read_stac_file()

        # Read measurement files
        bps_logger.debug("..measurement files")
        self._read_measurement_files()

        # Read annotation files
        bps_logger.debug("..main annotation file")
        self._read_main_annotation_file()

        bps_logger.debug("..done")

        return BIOMASSL2bAGBProduct(
            self.measurement,
            self.main_ads_product,
            self.main_ads_raster_image,
            self.main_ads_input_information,
            self.main_ads_processing_parameters,
            self.product_doi,
            self.agb_tile_iteration,
            self.agb_tile_status,
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
