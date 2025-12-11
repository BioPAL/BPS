# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
from bps.common import bps_logger
from bps.common.io import common_types, translate_common
from bps.common.io.parsing import parse
from bps.transcoder.io import (
    common_annotation_models_l2,
    main_annotation_models_l2a_fd,
    main_annotation_models_l2a_fh,
    main_annotation_models_l2a_gn,
    main_annotation_models_l2a_tfh,
)
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aLutAdsFD,
    BIOMASSL2aLutAdsFH,
    BIOMASSL2aLutAdsGN,
    BIOMASSL2aLutAdsTOMOFH,
    BIOMASSL2aMainADSInputInformation,
    BIOMASSL2aMainADSProcessingParametersFD,
    BIOMASSL2aMainADSProcessingParametersFH,
    BIOMASSL2aMainADSProcessingParametersGN,
    BIOMASSL2aMainADSProcessingParametersTOMOFH,
    BIOMASSL2aMainADSproduct,
    BIOMASSL2aMainADSRasterImage,
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
        # Read main annotation file
        assert self.product_structure.main_annotation_file is not None

        selected_reference_image = None  # only used in FD where more than two acquisitions have been used
        if self.product_type == "FP_FD__L2A":
            main_annotation_path = Path(self.product_structure.main_annotation_file)
            main_annotation_model: main_annotation_models_l2a_fd.MainAnnotation = parse(  # type: ignore
                main_annotation_path.read_text(encoding="utf-8"),
                main_annotation_models_l2a_fd.MainAnnotation,
            )
            selected_reference_image = main_annotation_model.product.selected_reference_image

        elif self.product_type == "FP_FH__L2A":
            main_annotation_path = Path(self.product_structure.main_annotation_file)
            main_annotation_model: main_annotation_models_l2a_fh.MainAnnotation = parse(  # type: ignore
                main_annotation_path.read_text(encoding="utf-8"),
                main_annotation_models_l2a_fh.MainAnnotation,
            )
        elif self.product_type == "FP_GN__L2A":
            main_annotation_path = Path(self.product_structure.main_annotation_file)
            main_annotation_model: main_annotation_models_l2a_gn.MainAnnotation = parse(
                main_annotation_path.read_text(encoding="utf-8"),
                main_annotation_models_l2a_gn.MainAnnotation,
            )
        elif self.product_type == "FP_TFH_L2A":
            main_annotation_path = Path(self.product_structure.main_annotation_file)
            main_annotation_model: main_annotation_models_l2a_tfh.MainAnnotation = parse(  # type: ignore
                main_annotation_path.read_text(encoding="utf-8"),
                main_annotation_models_l2a_tfh.MainAnnotation,
            )
        else:
            raise ValueError(f"Invalid product L2a: {self.product_type}")

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
        assert main_annotation_model.product.swath is not None
        assert main_annotation_model.product.major_cycle_id is not None
        assert main_annotation_model.product.absolute_orbit_number is not None
        assert main_annotation_model.product.relative_orbit_number is not None
        assert main_annotation_model.product.orbit_pass is not None
        assert main_annotation_model.product.data_take_id is not None
        assert main_annotation_model.product.frame is not None
        assert main_annotation_model.product.platform_heading is not None

        mission = main_annotation_model.product.mission.value
        tile_id_list = main_annotation_model.product.tile_id.id
        basin_id_list = main_annotation_model.product.basin_id.id
        product_type = main_annotation_model.product.product_type.value
        start_time = translate_common.translate_datetime(main_annotation_model.product.start_time)
        stop_time = translate_common.translate_datetime(main_annotation_model.product.stop_time)
        radar_carrier_frequency = main_annotation_model.product.radar_carrier_frequency.value
        mission_phase_id = main_annotation_model.product.mission_phase_id.value
        sensor_mode = main_annotation_model.product.sensor_mode.value
        global_coverage_id = self.parsed_product_name.coverage
        swath = main_annotation_model.product.swath.value
        major_cycle_id = self.parsed_product_name.major_cycle
        absolute_orbit_number = [tmp for tmp in main_annotation_model.product.absolute_orbit_number.val]
        relative_orbit_number = main_annotation_model.product.relative_orbit_number
        orbit_pass = main_annotation_model.product.orbit_pass.value
        datatake_id = [tmp for tmp in main_annotation_model.product.data_take_id.val]
        frame = self.parsed_product_name.frame_number
        platform_heading = main_annotation_model.product.platform_heading.value

        # retro compatibility
        if main_annotation_model.product.forest_coverage_percentage is None:
            forest_coverage_percentage = 100.0
        else:
            forest_coverage_percentage = main_annotation_model.product.forest_coverage_percentage

        assert self.product_structure.main_annotation_file is not None
        baseline_string = str(Path(self.product_structure.main_annotation_file).parent.parent)[-9:-7:1]
        if baseline_string == "__":
            baseline = 0
        else:
            baseline = int(baseline_string)

        assert radar_carrier_frequency is not None
        assert global_coverage_id is not None
        assert major_cycle_id is not None
        assert absolute_orbit_number is not None
        assert relative_orbit_number is not None
        assert datatake_id is not None
        assert frame is not None
        assert platform_heading is not None
        assert forest_coverage_percentage is not None

        self.main_ads_product = BIOMASSL2aMainADSproduct(
            mission,
            tile_id_list,
            basin_id_list,
            product_type,
            start_time,
            stop_time,
            radar_carrier_frequency,
            mission_phase_id,
            sensor_mode,
            global_coverage_id,
            swath,
            major_cycle_id,
            absolute_orbit_number,
            relative_orbit_number,
            orbit_pass,
            datatake_id,
            frame,
            platform_heading,
            baseline,
            forest_coverage_percentage,
            selected_reference_image,
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

        self.main_ads_raster_image = BIOMASSL2aMainADSRasterImage(
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
        assert main_annotation_model.input_information.product_type is not None
        assert main_annotation_model.input_information.polarisation_list is not None
        assert main_annotation_model.input_information.projection is not None
        assert main_annotation_model.input_information.footprint is not None
        assert main_annotation_model.input_information.acquisition_list is not None
        assert main_annotation_model.input_information.overall_products_quality_index is not None
        assert main_annotation_model.input_information.nominal_stack is not None

        # retro compatibility
        if main_annotation_model.input_information.vertical_wavenumbers is None:
            vertical_wavenumbers_info = common_annotation_models_l2.MinMaxTypeWithUnit(
                common_annotation_models_l2.FloatWithUnit(value=0.0, units=common_types.UomType.RAD_M),
                common_annotation_models_l2.FloatWithUnit(value=0.0, units=common_types.UomType.RAD_M),
            )
        else:
            vertical_wavenumbers_info = main_annotation_model.input_information.vertical_wavenumbers
        if main_annotation_model.input_information.height_of_ambiguity is None:
            height_of_ambiguity_info = common_annotation_models_l2.MinMaxTypeWithUnit(
                common_annotation_models_l2.FloatWithUnit(value=0.0, units=common_types.UomType.M),
                common_annotation_models_l2.FloatWithUnit(value=0.0, units=common_types.UomType.M),
            )
        else:
            height_of_ambiguity_info = main_annotation_model.input_information.height_of_ambiguity

        self.main_ads_input_information = BIOMASSL2aMainADSInputInformation(
            product_type=main_annotation_model.input_information.product_type.value,
            overall_products_quality_index=main_annotation_model.input_information.overall_products_quality_index,
            nominal_stack=main_annotation_model.input_information.nominal_stack,
            polarisation_list=main_annotation_model.input_information.polarisation_list,
            projection=main_annotation_model.input_information.projection.value,
            footprint=[float(string) for string in main_annotation_model.input_information.footprint.value.split()],
            vertical_wavenumbers=vertical_wavenumbers_info,
            height_of_ambiguity=height_of_ambiguity_info,
            acquisition_list=main_annotation_model.input_information.acquisition_list,
        )

        assert main_annotation_model.processing_parameters is not None
        assert main_annotation_model.processing_parameters.product_resolution is not None
        if not self.product_type == "FP_TFH_L2A":
            assert main_annotation_model.processing_parameters.upsampling_factor is not None

        processor_version = main_annotation_model.processing_parameters.processor_version
        product_generation_time = main_annotation_model.processing_parameters.product_generation_time
        general_configuration = main_annotation_model.processing_parameters.general_configuration_parameters

        compression_options = main_annotation_model.processing_parameters.compression_options

        product_resolution = main_annotation_model.processing_parameters.product_resolution.value

        if not self.product_type == "FP_TFH_L2A":
            upsampling_factor = main_annotation_model.processing_parameters.upsampling_factor
        assert processor_version is not None
        assert product_generation_time is not None
        assert general_configuration is not None
        assert compression_options is not None
        assert product_resolution is not None
        if not self.product_type == "FP_TFH_L2A":
            assert upsampling_factor is not None

        if self.product_type == "FP_FD__L2A":
            assert main_annotation_model.processing_parameters.emphasized_forest_height is not None
            assert main_annotation_model.processing_parameters.operational_mode is not None

            emphasized_forest_height = main_annotation_model.processing_parameters.emphasized_forest_height.value
            operational_mode = main_annotation_model.processing_parameters.operational_mode.value
            images_pair_selection = main_annotation_model.processing_parameters.images_pair_selection
            assert main_annotation_model.processing_parameters.disable_ground_cancellation_flag is not None
            disable_ground_cancellation_flag = str_to_bool(
                main_annotation_model.processing_parameters.disable_ground_cancellation_flag
            )
            significance_level = main_annotation_model.processing_parameters.significance_level
            numerical_determinant_limit = main_annotation_model.processing_parameters.numerical_determinant_limit

            assert emphasized_forest_height is not None
            assert disable_ground_cancellation_flag is not None

            self.main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersFD(
                processor_version,
                product_generation_time,
                general_configuration,
                compression_options,
                emphasized_forest_height,
                operational_mode,
                significance_level,
                product_resolution,
                numerical_determinant_limit,
                upsampling_factor,
                images_pair_selection,
                disable_ground_cancellation_flag,
            )

        elif self.product_type == "FP_FH__L2A":
            vertical_reflectivity_option = (
                main_annotation_model.processing_parameters.vertical_reflectivity_option.value
            )
            model_inversion = main_annotation_model.processing_parameters.model_inversion.value
            spectral_decorrelation_compensation_flag = str_to_bool(
                main_annotation_model.processing_parameters.spectral_decorrelation_compensation_flag
            )
            snr_decorrelation_compensation_flag = str_to_bool(
                main_annotation_model.processing_parameters.snrdecorrelation_compensation
            )
            if main_annotation_model.processing_parameters.correct_terrain_slopes_flag is None:
                bps_logger.warning(
                    "Missing 'correct_terrain_slopes_flag' in L2A FH product Main Annotation, this will become an error in future releases"
                )
                correct_terrain_slopes_flag = True
            else:
                correct_terrain_slopes_flag = str_to_bool(
                    main_annotation_model.processing_parameters.correct_terrain_slopes_flag
                )
            normalised_height_estimation_range = (
                main_annotation_model.processing_parameters.normalised_height_estimation_range
            )

            normalised_wavenumber_estimation_range = (
                main_annotation_model.processing_parameters.normalised_wavenumber_estimation_range
            )
            ground_to_volume_ratio_range = main_annotation_model.processing_parameters.ground_to_volume_ratio_range
            temporal_decorrelation_estimation_range = (
                main_annotation_model.processing_parameters.temporal_decorrelation_estimation_range
            )
            temporal_decorrelation_ground_to_volume_ratio = (
                main_annotation_model.processing_parameters.temporal_decorrelation_ground_to_volume_ratio
            )
            residual_decorrelation = main_annotation_model.processing_parameters.residual_decorrelation

            uncertainty_valid_values_limits = main_annotation_model.processing_parameters.uncertainty_validvalues_limits
            vertical_wavenumber_valid_values_limits = (
                main_annotation_model.processing_parameters.vertical_wavenumber_validvalues_limits
            )
            lower_height_limit = main_annotation_model.processing_parameters.lower_height_limit
            vertical_reflectivity_default_profile = (
                main_annotation_model.processing_parameters.vertical_reflectivity_default_profile
            )
            self.main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersFH(
                processor_version,
                product_generation_time,
                general_configuration,
                compression_options,
                vertical_reflectivity_option,
                model_inversion,
                spectral_decorrelation_compensation_flag,
                snr_decorrelation_compensation_flag,
                correct_terrain_slopes_flag,
                normalised_height_estimation_range,
                normalised_wavenumber_estimation_range,
                ground_to_volume_ratio_range,
                temporal_decorrelation_estimation_range,
                temporal_decorrelation_ground_to_volume_ratio,
                residual_decorrelation,
                product_resolution,
                uncertainty_valid_values_limits,
                vertical_wavenumber_valid_values_limits,
                lower_height_limit,
                upsampling_factor,
                vertical_reflectivity_default_profile,
            )

        elif self.product_type == "FP_GN__L2A":
            assert main_annotation_model.processing_parameters.emphasized_forest_height is not None
            assert main_annotation_model.processing_parameters.operational_mode is not None
            assert main_annotation_model.processing_parameters.compute_gnpower_flag is not None
            assert main_annotation_model.processing_parameters.radiometric_calibration_flag is not None
            assert main_annotation_model.processing_parameters.disable_ground_cancellation_flag is not None
            assert main_annotation_model.processing_parameters.compression_options is not None
            assert main_annotation_model.processing_parameters.compression_options.ads is not None
            assert main_annotation_model.processing_parameters.compression_options.ads.local_incidence_angle is not None
            least_significant_digit = main_annotation_model.processing_parameters.compression_options.ads.local_incidence_angle.least_significant_digit
            emphasized_forest_height = main_annotation_model.processing_parameters.emphasized_forest_height.value
            operational_mode = main_annotation_model.processing_parameters.operational_mode.value
            compute_gn_power_flag = str_to_bool(main_annotation_model.processing_parameters.compute_gnpower_flag)
            radiometric_calibration_flag = str_to_bool(
                main_annotation_model.processing_parameters.radiometric_calibration_flag
            )
            images_pair_selection = main_annotation_model.processing_parameters.images_pair_selection
            disable_ground_cancellation_flag = str_to_bool(
                main_annotation_model.processing_parameters.disable_ground_cancellation_flag
            )

            assert least_significant_digit is not None
            assert emphasized_forest_height is not None
            self.main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersGN(
                processor_version,
                product_generation_time,
                general_configuration,
                compression_options,
                least_significant_digit,
                emphasized_forest_height,
                operational_mode,
                compute_gn_power_flag,
                radiometric_calibration_flag,
                product_resolution,
                upsampling_factor,
                images_pair_selection,
                disable_ground_cancellation_flag,
            )

        elif self.product_type == "FP_TFH_L2A":
            assert main_annotation_model.processing_parameters.regularization_noise_factor is not None
            assert main_annotation_model.processing_parameters.power_threshold is not None
            assert main_annotation_model.processing_parameters.median_factor is not None
            assert main_annotation_model.processing_parameters.estimation_valid_values_limits is not None

            enable_super_resolution = str_to_bool(main_annotation_model.processing_parameters.enable_super_resolution)

            self.main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersTOMOFH(
                processor_version,
                product_generation_time,
                general_configuration,
                compression_options,
                enable_super_resolution,
                product_resolution,
                main_annotation_model.processing_parameters.regularization_noise_factor,
                main_annotation_model.processing_parameters.power_threshold,
                main_annotation_model.processing_parameters.median_factor,
                main_annotation_model.processing_parameters.estimation_valid_values_limits,
            )

    def _read_lut_annotation_file(self):
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
                if self.product_type == "FP_FD__L2A":
                    lut_fnf_metadata = BIOMASSL2aLutAdsFD.LutMetadata(
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
                if self.product_type == "FP_FH__L2A":
                    lut_fnf_metadata = BIOMASSL2aLutAdsFH.LutMetadata(
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
                if self.product_type == "FP_GN__L2A":
                    lut_fnf_metadata = BIOMASSL2aLutAdsFH.LutMetadata(
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
                if self.product_type == "FP_TFH_L2A":
                    lut_fnf_metadata = BIOMASSL2aLutAdsTOMOFH.LutMetadata(
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
                lut_acm_metadata = BIOMASSL2aLutAdsFD.LutMetadata(
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
                lut_number_of_averages_metadata = BIOMASSL2aLutAdsFD.LutMetadata(
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
            if group_name == "localIncidenceAngle":
                lut_local_incidence_angle_metadata = BIOMASSL2aLutAdsGN.LutMetadata(
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

        if self.product_type == "FP_FD__L2A":
            self.lut_ads = BIOMASSL2aLutAdsFD(
                lut_fnf,
                lut_acm,
                lut_number_of_averages,
                lut_fnf_metadata,  # type: ignore
                lut_acm_metadata,
                lut_number_of_averages_metadata,
            )

        elif self.product_type == "FP_FH__L2A":
            self.lut_ads = BIOMASSL2aLutAdsFH(
                lut_fnf,
                lut_fnf_metadata,  # type: ignore
            )

        elif self.product_type == "FP_GN__L2A":
            self.lut_ads = BIOMASSL2aLutAdsGN(
                lut_fnf,
                lut_local_incidence_angle,
                lut_fnf_metadata,  # type: ignore
                lut_local_incidence_angle_metadata,
            )

        elif self.product_type == "FP_TFH_L2A":
            self.lut_ads = BIOMASSL2aLutAdsFH(
                lut_fnf,
                lut_fnf_metadata,  # type: ignore
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
