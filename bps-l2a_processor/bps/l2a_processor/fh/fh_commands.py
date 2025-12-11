# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
FH commands
-----------
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import bps.l2a_processor
import numba as nb
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.fnf_utils import FnFMask
from bps.common.io import common_types
from bps.l2a_processor.core.aux_pp2_2a import AuxProcessingParametersL2A, MinMaxNumType, MinMaxType
from bps.l2a_processor.core.joborder_l2a import L2aJobOrder
from bps.l2a_processor.core.translate_job_order import L2A_OUTPUT_PRODUCT_FH
from bps.l2a_processor.fh import BPS_L2A_FH_PROCESSOR_NAME
from bps.l2a_processor.l2a_common_functionalities import (
    averaging_windows_sizes,
    build_filtering_sparse_matrices,
    check_lat_lon_orientation,
    fnf_annotation,
    geocoding,
    geocoding_update_dem_coordinates,
    get_dgg_sampling,
    mpmb_covariance_estimation,
    parallel_reinterpolate,
    refine_dgg_search_tiles,
)
from bps.transcoder.io import common_annotation_models_l2
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aLutAdsFH,
    BIOMASSL2aMainADSInputInformation,
    BIOMASSL2aMainADSProcessingParametersFH,
    BIOMASSL2aMainADSproduct,
    BIOMASSL2aMainADSRasterImage,
    BIOMASSL2aProductFH,
    BIOMASSL2aProductMeasurement,
    main_annotation_models_l2a_fh,
)
from bps.transcoder.sarproduct.biomass_l2aproduct_writer import (
    AVERAGING_FACTOR_QUICKLOOKS,
    COMPRESSION_EXIF_CODES_LERC_ZSTD,  # LERC, ZSTD
    DECIMATION_FACTOR_QUICKLOOKS,
    FLOAT_NODATA_VALUE,
    INT_NODATA_VALUE,
    BIOMASSL2aProductWriter,
)
from bps.transcoder.sarproduct.biomass_stackproduct import BIOMASSStackProduct
from bps.transcoder.sarproduct.l2_annotations import COORDINATE_REFERENCE_SYSTEM, ground_corner_points
from bps.transcoder.utils.dgg_utils import create_dgg_sampling_dict, dgg_search_tiles
from netCDF4 import Dataset

LIGHTSPEED = 299792458


class FH:
    """Forest height L2a Processor"""

    def __init__(
        self,
        job_order: L2aJobOrder,
        aux_pp2_2a: AuxProcessingParametersL2A,
        working_dir: Path,
        stack_products_list: list[BIOMASSStackProduct],
        scs_axes_dict: tuple[np.ndarray, np.ndarray, PreciseDateTime],
        scs_pol_list_calibrated: list[list[np.ndarray]],
        stack_lut_list: list[Dataset],
        stack_lut_axes_dict: dict,
        primary_image_index: int,
        acquisition_paths_selected_not_sorted: list[Path],
        mission_phase_id: str,
        fnf: FnFMask,
        latlon_coverage: list[float],
        forest_coverage_percentage: float,
        basin_id_list: list[str],
    ) -> None:
        """
        Parameters
        ----------
        job_order: L2aJobOrder
            content of the job order XML file
        aux_pp2_2a: AuxProcessingParametersL2A
            content of the AUX PP 2A XML file
        working_dir: Path
            working directory
        stack_products_list: List[BIOMASSStackProduct]
            list of all the acqisition products paths contained in the stack,
            two or three, as already selected by int_subsetting
        scs_axes_dict: Tuple[np.ndarray, np.ndarray, PreciseDateTime]
            input scs stack product data axes. retrived from str_image main annotation xml,
            the dictionary contains:
                self.scs_axes_dict["scs_axes_dict["scs_axis_sr_s"]"]: np.ndarray
                    slant range temporal axis, in seconds
                self.scs_axes_dict["scs_axes_dict["scs_axis_az_s"]"]: np.ndarray
                    azimuth temporal axis, in seconds
                self.scs_axes_dict["scs_axes_dict["scs_axis_az_mjd"]"]: PreciseDateTime
                    azimuth temporal axis, in MJD
        scs_pol_list_calibrated: List[List[np.ndarray]]
            Same data content from stack_products_list, but calibrated and reshaped:
            calibrated and reshaped scs data, if calibration is enabled;
            or just stack_products_list scs data reshaped if not.
                list with P=3 polarizations (in the order HH, XX, VV),
                each containing a list with M=2 or 3 acquisitions,
                each of dimensions [N_az x N_rg]
        stack_lut_list = List[Dataset]
            list of look up tables related to the stack product
            same cardinality of stack_products_list
        stack_lut_axes_dict: Dict
            Lut axes, one specific for each LUT
        primary_image_index: int
            it is a zero-based index of the primary image:
            the index is respect the order as found in the stack_products_list (NOT baselines-sorted)
        acquisition_paths_selected_not_sorted: List[Path]
            Path of each stack product acquisition, ordered as the stack_products_list
        mission_phase_id: str
            it can be INT, TOM, depending on the number of acquisitions found in input job order
        fnf: FnFMask,
            forest-non-forest mask object, containing fnf itself and axis definitions
            the map is a cut over footprint, of the whole FNF
        latlon_coverage: List[float]
            latitude/Longitude coverage, in degrees
            [lat_min, lat_max, lon_min, lon_max]
        forest_coverage_percentage: float
            forest coverage percentage
        basin_id_list: list[str]:
            List of Basin IDs
        """

        self.job_order = job_order
        self.aux_pp2_2a = aux_pp2_2a
        self.working_dir = working_dir
        self.stack_products_list = stack_products_list
        self.scs_axes_dict = scs_axes_dict
        self.scs_pol_list_calibrated = scs_pol_list_calibrated
        self.stack_lut_list = stack_lut_list
        self.mission_phase_id = mission_phase_id
        self.fnf = fnf
        self.latlon_coverage = latlon_coverage
        self.forest_coverage_percentage = forest_coverage_percentage
        self.basin_id_list = basin_id_list
        self.stack_lut_axes_dict = stack_lut_axes_dict
        self.primary_image_index = primary_image_index
        self.acquisition_paths_selected_not_sorted = acquisition_paths_selected_not_sorted
        self.output_baseline = 0  # set after from job order

    def _initialize_processing(self):
        """Initialize the FH L2a processing"""

        self.processing_start_time = datetime.now()
        self.start_time = PreciseDateTime.now()
        bps_logger.info("%s started", BPS_L2A_FH_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baselines is not None:
            for output_product, output_baseline in zip(self.job_order.output_products, self.job_order.output_baselines):
                if output_product == L2A_OUTPUT_PRODUCT_FH:
                    self.output_baseline = output_baseline

        self.product_type = L2A_OUTPUT_PRODUCT_FH

    def _core_processing(self):
        """Execute core FH L2a processing"""

        # vertical_wavenumbers:
        vertical_wavenumbers = [lut["waveNumbers"].astype(np.float64) for lut in self.stack_lut_list]

        # Coherence Estimation
        average_azimuth_velocity = np.mean(
            [
                np.linalg.norm(velocity)
                for velocity in self.stack_products_list[self.primary_image_index].general_sar_orbit[0].velocity_vector
            ]
        )

        bps_logger.info(
            f"Terrain slopes correction {'enabled' if self.aux_pp2_2a.fh.correct_terrain_slopes_flag else 'disabled'} in AUX PP2 2A"
        )

        (
            coherence_for_baseline_combinations,
            vertical_wavenumbers,  # Terrain corrected
            inc_angle_terrain_corrected_rad,
            terrain_slope_rad,
            resolution_rg,
            axis_az_subsampling_indexes,
            axis_rg_subsampling_indexes,
            _,
        ) = coherence_estimation(
            self.scs_pol_list_calibrated,
            self.scs_axes_dict["scs_axis_sr_s"],
            self.scs_axes_dict["scs_axis_az_s"],
            [
                [lut["denoisingHH"].astype(np.float64) for lut in self.stack_lut_list],
                [lut["denoisingXX"].astype(np.float64) for lut in self.stack_lut_list],
                [lut["denoisingVV"].astype(np.float64) for lut in self.stack_lut_list],
            ],
            [lut["sigmaNought"] for lut in self.stack_lut_list],
            np.deg2rad(self.stack_lut_list[self.primary_image_index]["terrainSlope"].astype(np.float32))
            * int(
                self.aux_pp2_2a.fh.correct_terrain_slopes_flag
            ),  # set to zero the slope if not to be used (from AUX-PP)
            np.deg2rad(self.stack_lut_list[self.primary_image_index]["incidenceAngle"].astype(np.float32)),
            vertical_wavenumbers,
            self.stack_lut_axes_dict["axis_primary_sr_s"],
            self.stack_lut_axes_dict["axis_primary_az_s"],
            self.stack_products_list[self.primary_image_index].sampling_constants_list[0].brg_hz,
            self.stack_products_list[self.primary_image_index].sampling_constants_list[0].baz_hz,
            average_azimuth_velocity,
            self.aux_pp2_2a.fh.snr_decorrelation_compensation_flag,
            self.aux_pp2_2a.fh.product_resolution,
            self.aux_pp2_2a.fh.upsampling_factor,
            self.aux_pp2_2a.fh.vertical_wavenumber_valid_values_limits,
        )

        forest_height, forest_height_quality, _ = forest_height_inversion(
            coherence_for_baseline_combinations,
            resolution_rg,
            vertical_wavenumbers,
            terrain_slope_rad,
            inc_angle_terrain_corrected_rad,
            self.aux_pp2_2a.fh.residual_decorrelation,
            self.aux_pp2_2a.fh.vertical_reflectivity_default_profile,
            self.aux_pp2_2a.fh.vertical_reflectivity_option.value,
            self.aux_pp2_2a.fh.model_inversion.value,
            self.aux_pp2_2a.fh.normalised_height_estimation_range,
            self.aux_pp2_2a.fh.normalised_wavenumber_estimation_range,
            self.aux_pp2_2a.fh.ground_to_volume_ratio_range,
            self.aux_pp2_2a.fh.temporal_decorrelation_estimation_range,
            self.aux_pp2_2a.fh.spectral_decorrelation_compensation_flag,
            self.aux_pp2_2a.fh.correct_terrain_slopes_flag,
            self.aux_pp2_2a.fh.temporal_decorrelation_ground_to_volume_ratio,
        )

        lut_dict = {}

        # Get the DGG sampling parameters, needed for the geocoding step:
        # Check the orientation of FNF lat lon axis and keep this convention in the L2a product.
        # Get the DGG sampling parameters, needed for the geocoding step:
        # Check the orientation of FNF lat lon axis and keep this convention in the L2a product.
        invert_latitude, invert_longitude = check_lat_lon_orientation(self.fnf.lat_axis, self.fnf.lon_axis)
        (
            dgg_latitude_axis_deg,
            dgg_longitude_axis_deg,
        ) = get_dgg_sampling(
            self.latlon_coverage,
            create_dgg_sampling_dict(L2A_OUTPUT_PRODUCT_FH),
            invert_latitude,
            invert_longitude,
        )

        # Geocoding step 1/2: update DEM coordinates
        (
            delaunay,
            dgg_latitude_mesh_rad,
            dgg_longitude_mesh_rad,
            dem_valid_values_mask,
        ) = geocoding_update_dem_coordinates(
            self.stack_lut_list[self.primary_image_index]["height"],
            self.stack_lut_list[self.primary_image_index]["latitude"],
            self.stack_lut_list[self.primary_image_index]["longitude"],
            self.stack_lut_axes_dict["axis_primary_az_s"],
            self.stack_lut_axes_dict["axis_primary_sr_s"],
            self.scs_axes_dict["scs_axis_az_s"][axis_az_subsampling_indexes],  # sub
            self.scs_axes_dict["scs_axis_sr_s"][axis_rg_subsampling_indexes],  # sub
            self.scs_axes_dict["scs_axis_az_mjd"][axis_az_subsampling_indexes],  # sub
            self.stack_products_list[self.primary_image_index].general_sar_orbit[0],  # SV
            self.aux_pp2_2a.fd.ground_cancellaton.emphasized_forest_height,
            np.deg2rad(dgg_latitude_axis_deg),
            np.deg2rad(dgg_longitude_axis_deg),
        )

        bps_logger.info("Geocoding forest height and quality:")
        # Pass both foresth height and quality in a list, in order to compute it faster in a single Delaunay call
        geocoded_list = geocoding(
            [forest_height, forest_height_quality],
            delaunay,
            dgg_latitude_mesh_rad,
            dgg_longitude_mesh_rad,
            dem_valid_values_mask,
            fill_value=FLOAT_NODATA_VALUE,
        )

        # Filling the output dictionary for GN
        # Casting back to float32 all the float data before saving
        # Geocoded list contains forest height at index 0 and quality at index 1
        # For consistency with other processors, put each MDS in a list (in GN there are three values in each list)
        processed_data_dict = {"fh": [], "quality": []}
        processed_data_dict["fh"].append(geocoded_list[0].astype(np.float32))
        processed_data_dict["quality"].append(geocoded_list[1].astype(np.float32))
        del geocoded_list  # free memory

        # Geocode a data mask of ones, to be used for the FNF mask masking over the data footprint
        # Pass it as a list, and get list [0] element
        footprint_mask_geocoded = geocoding(
            [np.ones(forest_height.shape, dtype=int)],
            delaunay,
            dgg_latitude_mesh_rad,
            dgg_longitude_mesh_rad,
            dem_valid_values_mask,
            fill_value=int(0),
        )[0].astype(bool)

        # Discard values
        bps_logger.info(
            f"    discarding estimations out of AUX PP2 2A uncertainty valid values limits [{self.aux_pp2_2a.fh.uncertainty_valid_values_limits.min}%, {self.aux_pp2_2a.fh.uncertainty_valid_values_limits.max}%]:"
        )

        float_no_data_values_mask = processed_data_dict["quality"][0] == FLOAT_NODATA_VALUE
        uncertainity_mask = np.logical_or(
            processed_data_dict["quality"][0] < self.aux_pp2_2a.fh.uncertainty_valid_values_limits.min,
            processed_data_dict["quality"][0] > self.aux_pp2_2a.fh.uncertainty_valid_values_limits.max,
        )
        num_nans_before_removal = np.sum(float_no_data_values_mask)

        processed_data_dict["fh"][0][uncertainity_mask] = FLOAT_NODATA_VALUE
        processed_data_dict["quality"][0][uncertainity_mask] = FLOAT_NODATA_VALUE

        num_uncertainity_removed_pixels = np.sum(uncertainity_mask) - num_nans_before_removal

        size_fh = processed_data_dict["fh"][0].size
        bps_logger.warning(
            f"        {num_uncertainity_removed_pixels / size_fh * 100:2.3f}% of pixels removed (pixels out of uncertainty valid values limits)"
        )

        bps_logger.info(
            f"    discarding estimations out of AUX PP2 2A lower height limit {self.aux_pp2_2a.fh.lower_height_limit} [m]:"
        )

        num_under_lower_height_limit = np.sum(
            processed_data_dict["fh"][0] < self.aux_pp2_2a.fh.lower_height_limit
        ) - np.sum(uncertainity_mask)

        lower_height_limit_mask = processed_data_dict["fh"][0] < self.aux_pp2_2a.fh.lower_height_limit

        processed_data_dict["fh"][0][lower_height_limit_mask] = FLOAT_NODATA_VALUE
        processed_data_dict["quality"][0][lower_height_limit_mask] = FLOAT_NODATA_VALUE

        bps_logger.warning(
            f"        {num_under_lower_height_limit / size_fh * 100:2.3f}% of pixels removed (pixels under lower height limit)"
        )

        # Footprint mask for quick looks transparency
        fh_for_footprint = np.copy(processed_data_dict["fh"][0].squeeze())
        fh_for_footprint[fh_for_footprint == FLOAT_NODATA_VALUE] = np.nan
        if AVERAGING_FACTOR_QUICKLOOKS > 1:
            fh_for_footprint = fh_for_footprint[::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS]

        footprint_mask_for_quicklooks = np.logical_not(np.isnan(fh_for_footprint))

        # FNF annotation step
        lut_dict["fnf"] = fnf_annotation(
            self.fnf,
            dgg_latitude_axis_deg,
            dgg_longitude_axis_deg,
            self.aux_pp2_2a.general.forest_mask_interpolation_threshold,
        )
        lut_dict["fnf"]["fnf"][np.logical_not(footprint_mask_geocoded)] = INT_NODATA_VALUE

        processed_data_dict["fh"][0][np.isnan(processed_data_dict["fh"][0])] = FLOAT_NODATA_VALUE
        processed_data_dict["fh"][0][np.isinf(processed_data_dict["fh"][0])] = FLOAT_NODATA_VALUE
        processed_data_dict["quality"][0][np.isnan(processed_data_dict["quality"][0])] = FLOAT_NODATA_VALUE
        processed_data_dict["quality"][0][np.isinf(processed_data_dict["quality"][0])] = FLOAT_NODATA_VALUE

        return (
            processed_data_dict,
            dgg_latitude_axis_deg.astype(np.float32),
            dgg_longitude_axis_deg.astype(np.float32),
            lut_dict,
            footprint_mask_for_quicklooks,
        )

    def _fill_product_for_writing(self, lut_dict):
        # Search the tiles falling in the STA bounding box
        (
            _,
            _,
            _,
            tiles_dict,
        ) = dgg_search_tiles(self.latlon_coverage, True)
        # Refine the above search, searching the tiles footprints falling in the STA footprint
        self.tile_id_list = refine_dgg_search_tiles(
            tiles_dict,
            np.array(self.stack_products_list[self.primary_image_index].footprint)[:, 0],
            np.array(self.stack_products_list[self.primary_image_index].footprint)[:, 1],
        )

        self.radar_carrier_frequency = 435000000.0

        lut_dict["fnf_metadata"] = BIOMASSL2aLutAdsFH.LutMetadata(
            first_sample=1,
            first_line=1,
            samples_interval=1,
            lines_interval=1,
            pixelType="8 bit Unsigned Integer",
            no_data_value=INT_NODATA_VALUE,
            projection=(common_annotation_models_l2.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG.value),
            coordinateReferenceSystem=COORDINATE_REFERENCE_SYSTEM,
            geodeticReferenceFrame=common_annotation_models_l2.GeodeticReferenceFrameType.WGS84.value,
        )

        # COG metadata
        # fill common fileds for FH and quality
        meta_data_temp = [
            BIOMASSL2aProductMeasurement.MetadataCOG(
                swath=common_annotation_models_l2.SwathType.S1.value,
                tile_id_list=self.tile_id_list,
                basin_id_list=self.basin_id_list,
                compression=COMPRESSION_EXIF_CODES_LERC_ZSTD,  #  [LERC, ZSTD]
                image_description="",
                software="",
                dateTime=self.start_time.isoformat(timespec="microseconds")[:-1],
            )
            for idx in range(2)
        ]

        # specific fileds for FH and FH Quality
        meta_data = {
            "fh": deepcopy(meta_data_temp[0]),
            "quality": deepcopy(meta_data_temp[1]),
        }

        meta_data["fh"].image_description = (
            f"BIOMASS L2a {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_HEIGHT_M.value
        )

        meta_data["quality"].image_description = (
            f"BIOMASS L2a {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_HEIGHT_QUALITY.value
        )

        return lut_dict, meta_data

    def _write_to_output(
        self,
        processed_data_dict,
        dgg_latitude_axis,
        dgg_longitude_axis,
        lut_dict,
        footprint_mask_for_quicklooks,
    ):
        """Write output FH L2a product"""

        # lat_min, lat_max, lon_min, lon_max]
        footprint = [
            self.stack_products_list[self.primary_image_index].footprint[0][0],
            self.stack_products_list[self.primary_image_index].footprint[0][1],
            self.stack_products_list[self.primary_image_index].footprint[1][0],
            self.stack_products_list[self.primary_image_index].footprint[1][1],
            self.stack_products_list[self.primary_image_index].footprint[2][0],
            self.stack_products_list[self.primary_image_index].footprint[2][1],
            self.stack_products_list[self.primary_image_index].footprint[3][0],
            self.stack_products_list[self.primary_image_index].footprint[3][1],
        ]

        lut_dict, metadata_dict = self._fill_product_for_writing(lut_dict)
        self.processing_stop_time = datetime.now()
        self.stop_time = PreciseDateTime.now()

        # Fill input objects for BIOMASSL2aProductFD initialization:
        # measurement,
        # main_ads_product,
        # main_ads_raster_image,
        # main_ads_input_information,
        # main_ads_processing_parameters,
        # lut_ads,

        # measurement object
        measurement = BIOMASSL2aProductMeasurement(
            dgg_latitude_axis,
            dgg_longitude_axis,
            processed_data_dict,
            metadata_dict,
        )

        # main_ads_product
        mission = common_annotation_models_l2.MissionType.BIOMASS.value
        sensor_mode = common_annotation_models_l2.SensorModeType.MEASUREMENT.value
        orbit_direction = (
            self.stack_products_list[self.primary_image_index].orbit_direction[0]
            + self.stack_products_list[self.primary_image_index].orbit_direction[1:].lower()
        )

        start_time_l1c = self.stack_products_list[0].start_time
        stop_time_l1c = self.stack_products_list[0].stop_time
        for l1c_product in self.stack_products_list:
            start_time_l1c = min(start_time_l1c, l1c_product.start_time)
            stop_time_l1c = max(stop_time_l1c, l1c_product.stop_time)
        main_ads_product = BIOMASSL2aMainADSproduct(
            mission,
            self.tile_id_list,
            self.basin_id_list,
            self.product_type,
            start_time_l1c,
            stop_time_l1c,
            self.radar_carrier_frequency,
            self.mission_phase_id,
            sensor_mode,
            self.stack_products_list[self.primary_image_index].global_coverage_id,
            self.stack_products_list[self.primary_image_index].swath_list[0],
            self.stack_products_list[self.primary_image_index].major_cycle_id,
            [product.orbit_number for product in self.stack_products_list],
            self.stack_products_list[self.primary_image_index].track_number,
            orbit_direction,
            [product.datatake_id for product in self.stack_products_list],
            self.stack_products_list[self.primary_image_index].frame_number,
            self.stack_products_list[self.primary_image_index].platform_heading,
            self.output_baseline,
            forest_coverage_percentage=self.forest_coverage_percentage,
        )

        # main_ads_raster_image
        projection = common_annotation_models_l2.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG.value
        coordinate_reference_system = COORDINATE_REFERENCE_SYSTEM
        geodetic_reference_frame = common_annotation_models_l2.GeodeticReferenceFrameType.WGS84.value
        datum = common_annotation_models_l2.DatumType(
            coordinate_reference_system,
            common_annotation_models_l2.GeodeticReferenceFrameType(geodetic_reference_frame),
        )
        pixel_representation_dict = {
            "fh": common_types.PixelRepresentationType.FOREST_HEIGHT_M,
            "quality": common_types.PixelRepresentationType.FOREST_HEIGHT_QUALITY,
        }
        pixel_representation = common_annotation_models_l2.PixelRepresentationChoiceType(
            fh=pixel_representation_dict["fh"],
            quality=pixel_representation_dict["quality"],
        )
        pixel_type = common_annotation_models_l2.PixelTypeChoiceType(
            float_pixel_type=main_annotation_models_l2a_fh.PixelTypeType("32 bit Float")
        )

        no_data_value = common_annotation_models_l2.NoDataValueChoiceType(float_no_data_value=FLOAT_NODATA_VALUE)
        main_ads_raster_image = BIOMASSL2aMainADSRasterImage(
            footprint,
            dgg_latitude_axis[0],
            dgg_longitude_axis[0],
            dgg_latitude_axis[1] - dgg_latitude_axis[0],
            dgg_longitude_axis[1] - dgg_longitude_axis[0],
            len(dgg_latitude_axis),
            len(dgg_longitude_axis),
            projection,
            datum,
            pixel_representation,
            pixel_type,
            no_data_value,
        )

        # main_ads_input_information
        polarisation_list = common_annotation_models_l2.PolarisationListType(
            polarisation=[
                common_annotation_models_l2.PolarisationType("HH"),
                common_annotation_models_l2.PolarisationType("VH"),
                common_annotation_models_l2.PolarisationType("VV"),
            ],
            count=int(3),
        )

        acquisition_list = []
        overall_product_quality_indices = []
        for idx, acq_folder_name in enumerate(self.acquisition_paths_selected_not_sorted):
            overall_product_quality_indices.append(
                self.stack_products_list[idx].stack_quality.overall_product_quality_index
            )
            if idx == self.primary_image_index:  # primary image index is ordered as the acquisitions (not sorted)
                reference_image = "true"
            else:
                reference_image = "false"

            sta_quality_parameters_list = []
            assert self.stack_products_list[0].stack_quality is not None
            assert self.stack_products_list[0].stack_quality.sta_quality_parameters_list is not None
            for param in self.stack_products_list[0].stack_quality.sta_quality_parameters_list:
                sta_quality_parameters_list.append(
                    common_annotation_models_l2.StaQualityParametersType(
                        invalid_l1a_data_samples=param.invalid_l1a_data_samples,
                        rfi_decorrelation=param.rfi_decorrelation,
                        rfi_decorrelation_threshold=param.rfi_decorrelation_threshold,
                        faraday_decorrelation=param.faraday_decorrelation,
                        faraday_decorrelation_threshold=param.faraday_decorrelation_threshold,
                        invalid_residual_shifts_fraction=param.invalid_residual_shifts_ratio,
                        residual_shifts_quality_threshold=param.residual_shifts_quality_threshold,
                        invalid_ground_phases_screen_estimates_fraction=param.invalid_skp_calibration_phase_screen_ratio,
                        ground_phases_screen_quality_threshold=param.skp_calibration_phase_screen_quality_threshold,
                        skp_decomposition_index=param.skp_decomposition_index,
                        polarisation=common_annotation_models_l2.PolarisationType(param.polarization),
                    )
                )

            sta_quality_parameters_list = common_annotation_models_l2.StaQualityParametersListType(
                sta_quality_parameters_list,
                len(self.stack_products_list[0].stack_quality.sta_quality_parameters_list),
            )

            sta_quality = common_annotation_models_l2.StaQualityType(
                self.stack_products_list[idx].stack_quality.overall_product_quality_index,
                sta_quality_parameters_list,
            )

            acquisition_list.append(
                common_annotation_models_l2.AcquisitionType(
                    folder_name=str(acq_folder_name.name),
                    sta_quality=sta_quality,
                    reference_image=reference_image,
                )
            )

        is_int_and_nominal = np.logical_and(
            self.mission_phase_id == "INT",
            len(self.acquisition_paths_selected_not_sorted) == 3,
        )
        is_tom_and_nominal = np.logical_and(
            self.mission_phase_id == "TOM",
            len(self.acquisition_paths_selected_not_sorted) == 7,
        )

        # Compute height of ambiguity from the average wavenumbers
        vertical_wavenumbers = [lut["waveNumbers"].astype(np.float64) for lut in self.stack_lut_list]
        average_wavenumbers = [np.nanmean(vw) for vw in vertical_wavenumbers]

        vertical_wavenumbers_info = common_annotation_models_l2.MinMaxTypeWithUnit(
            common_annotation_models_l2.FloatWithUnit(
                value=float(min(average_wavenumbers)),
                units=common_types.UomType.RAD_M,
            ),
            common_annotation_models_l2.FloatWithUnit(
                value=float(max(average_wavenumbers)),
                units=common_types.UomType.RAD_M,
            ),
        )

        wavenumber_spacings = np.abs(np.diff(average_wavenumbers))
        hoa_min = 2 * np.pi / min(wavenumber_spacings)
        hoa_max = 2 * np.pi / max(wavenumber_spacings)
        height_of_ambiguity_info = common_annotation_models_l2.MinMaxTypeWithUnit(
            common_annotation_models_l2.FloatWithUnit(value=float(hoa_min), units=common_types.UomType.M),
            common_annotation_models_l2.FloatWithUnit(value=float(hoa_max), units=common_types.UomType.M),
        )

        main_ads_input_information = BIOMASSL2aMainADSInputInformation(
            self.product_type,
            overall_products_quality_index=(0 if sum(overall_product_quality_indices) == 0 else 1),
            nominal_stack=str(np.logical_or(is_int_and_nominal, is_tom_and_nominal)).lower(),
            polarisation_list=polarisation_list,
            projection=projection,
            footprint=footprint,
            vertical_wavenumbers=vertical_wavenumbers_info,
            height_of_ambiguity=height_of_ambiguity_info,
            acquisition_list=common_annotation_models_l2.AcquisitionListType(
                acquisition=acquisition_list,
                count=len(acquisition_list),
            ),
        )

        # main_ads_processing_parameters
        general_configuration = common_annotation_models_l2.GeneralConfigurationParametersType(
            self.aux_pp2_2a.general.apply_calibration_screen.value,
            self.aux_pp2_2a.general.forest_coverage_threshold,
            self.aux_pp2_2a.general.forest_mask_interpolation_threshold,
            common_annotation_models_l2.SubsettingRuleType(self.aux_pp2_2a.general.subsetting_rule.value),
        )

        compression_options_fh = main_annotation_models_l2a_fh.CompressionOptionsL2A(
            main_annotation_models_l2a_fh.CompressionOptionsL2A.Mds(
                main_annotation_models_l2a_fh.CompressionOptionsL2A.Mds.Fh(
                    self.aux_pp2_2a.fh.compression_options.mds.fh.compression_factor,
                    self.aux_pp2_2a.fh.compression_options.mds.fh.max_z_error,
                ),
                main_annotation_models_l2a_fh.CompressionOptionsL2A.Mds.Quality(
                    self.aux_pp2_2a.fh.compression_options.mds.quality.compression_factor,
                    self.aux_pp2_2a.fh.compression_options.mds.quality.max_z_error,
                ),
            ),
            main_annotation_models_l2a_fh.CompressionOptionsL2A.Ads(
                main_annotation_models_l2a_fh.CompressionOptionsL2A.Ads.Fnf(
                    self.aux_pp2_2a.fh.compression_options.ads.fnf.compression_factor
                ),
            ),
            self.aux_pp2_2a.fh.compression_options.mds_block_size,
            self.aux_pp2_2a.fh.compression_options.ads_block_size,
        )

        normalised_height_estimation_range = main_annotation_models_l2a_fh.MinMaxType(
            min=self.aux_pp2_2a.fh.normalised_height_estimation_range.min,
            max=self.aux_pp2_2a.fh.normalised_height_estimation_range.max,
        )
        normalised_wavenumber_estimation_range = main_annotation_models_l2a_fh.MinMaxNumType(
            min=self.aux_pp2_2a.fh.normalised_wavenumber_estimation_range.min,
            max=self.aux_pp2_2a.fh.normalised_wavenumber_estimation_range.max,
            num=self.aux_pp2_2a.fh.normalised_wavenumber_estimation_range.num,
        )
        ground_to_volume_ratio_range = main_annotation_models_l2a_fh.MinMaxNumType(
            min=self.aux_pp2_2a.fh.ground_to_volume_ratio_range.min,
            max=self.aux_pp2_2a.fh.ground_to_volume_ratio_range.max,
            num=self.aux_pp2_2a.fh.ground_to_volume_ratio_range.num,
        )
        temporal_decorrelation_estimation_range = main_annotation_models_l2a_fh.MinMaxNumType(
            min=self.aux_pp2_2a.fh.temporal_decorrelation_estimation_range.min,
            max=self.aux_pp2_2a.fh.temporal_decorrelation_estimation_range.max,
            num=self.aux_pp2_2a.fh.temporal_decorrelation_estimation_range.num,
        )

        uncertainty_valid_values_limits = main_annotation_models_l2a_fh.MinMaxType(
            min=self.aux_pp2_2a.fh.uncertainty_valid_values_limits.min,
            max=self.aux_pp2_2a.fh.uncertainty_valid_values_limits.max,
        )
        vertical_wavenumber_valid_values_limits = main_annotation_models_l2a_fh.MinMaxType(
            min=self.aux_pp2_2a.fh.vertical_wavenumber_valid_values_limits.min,
            max=self.aux_pp2_2a.fh.vertical_wavenumber_valid_values_limits.max,
        )

        main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersFH(
            bps.l2a_processor.__version__,
            self.start_time,
            general_configuration,
            compression_options_fh,
            self.aux_pp2_2a.fh.vertical_reflectivity_option.value,
            self.aux_pp2_2a.fh.model_inversion.value,
            self.aux_pp2_2a.fh.spectral_decorrelation_compensation_flag,
            self.aux_pp2_2a.fh.snr_decorrelation_compensation_flag,
            self.aux_pp2_2a.fh.correct_terrain_slopes_flag,
            normalised_height_estimation_range,
            normalised_wavenumber_estimation_range,
            ground_to_volume_ratio_range,
            temporal_decorrelation_estimation_range,
            self.aux_pp2_2a.fh.temporal_decorrelation_ground_to_volume_ratio,
            self.aux_pp2_2a.fh.residual_decorrelation,
            self.aux_pp2_2a.fh.product_resolution,
            uncertainty_valid_values_limits,
            vertical_wavenumber_valid_values_limits,
            self.aux_pp2_2a.fh.lower_height_limit,
            self.aux_pp2_2a.fh.upsampling_factor,
            self.aux_pp2_2a.fh.vertical_reflectivity_default_profile,
        )

        # lut_ads
        lut_ads = BIOMASSL2aLutAdsFH(lut_dict["fnf"]["fnf"], lut_dict["fnf_metadata"])

        # Initialize FH Product
        product_to_write = BIOMASSL2aProductFH(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            lut_ads,
            product_doi=self.aux_pp2_2a.fh.l2aFHProductDOI,
        )

        # Write to file the FH Product
        write_obj = BIOMASSL2aProductWriter(
            product_to_write,
            self.product_path,
            bps.l2a_processor.BPS_L2A_PROCESSOR_NAME,
            bps.l2a_processor.__version__,
            [acq.name for acq in self.job_order.input_stack_acquisitions],
            ground_corner_points(dgg_latitude_axis, dgg_longitude_axis),
            self.job_order.aux_pp2_2a_path.name,
            self.job_order.fnf_directory.name,
            footprint_mask_for_quicklooks,
        )
        write_obj.write()

    def run_l2a_fh_processing(self):
        """Performs processing as described in job order.

        Parameters
        ----------

        Returns
        -------
        """

        self._initialize_processing()

        (
            processed_data_dict,
            dgg_latitude_axis,
            dgg_longitude_axis,
            lut_dict,
            footprint_mask_for_quicklooks,
        ) = self._core_processing()

        self._write_to_output(
            processed_data_dict,
            dgg_latitude_axis,
            dgg_longitude_axis,
            lut_dict,
            footprint_mask_for_quicklooks,
        )

        processing_stop_time = datetime.now()
        elapsed_time = processing_stop_time - self.processing_start_time
        bps_logger.info(
            "%s total processing time: %.3f s",
            BPS_L2A_FH_PROCESSOR_NAME,
            elapsed_time.total_seconds(),
        )


def coherence_estimation(
    scs_pol_list: list[np.ndarray],
    scs_axis_sr_s: list[np.ndarray],
    scs_axis_az_s: list[np.ndarray],
    denoising_pol_list: list[list[np.ndarray]],
    sigma_nought_list: list[np.ndarray],
    terrain_slope_rad: np.ndarray,
    incidence_angle_rad: np.ndarray,
    vertical_wavenumber_list: list[np.ndarray],
    lut_axis_sr_s: np.ndarray,
    lut_axis_az_s: np.ndarray,
    b_rg: float,
    b_az: float,
    average_az_velocity: float,
    snr_decorr_compensation_flag: bool,
    product_resolution: float,
    upsampling_factor: int,
    vertical_wavenumber_valid_values_limits: MinMaxType,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Coherence estimation

    Parameters
    ----------
    scs_pol_list: List[List[np.ndarray]]
        Stack of calibrated scs data
        list with P=3 polarizations (in the order HH, XP, VV),
        each containing a list with M=2 or 3 acquisitions,
        each of dimensions [N_az x N_rg]
        Note: it can contain also only 2 polarization, in this case, the list is still of P=3, but one element is None
    scs_axis_sr_s: np.ndarray
        slant range time axis of each scs data in scs_pol_list [s]
    scs_axis_az_s: np.ndarray
        azimuth time axis of each scs data in scs_pol_list[s]
    denoising_pol_list: List[List[np.ndarray]]
        Denoising LUT
        list with P=3 polarizations (in the order HH, XP, VV),
        each containing a list with M denoising matrices, each releated to an acquisition
        each of dimensions [N_az_lut x N_rg_lut]
    sigma_nought_list: List[np.ndarray]
        sigmaNought LUT
        list with M sigma nought, each releated to an acquisition
        each of dimensions [N_az_lut x N_rg_lut]
    terrain_slope_rad: np.ndarray
        Terrain slope in [rad], of reference acquisition
        of dimensions [N_az_lut x N_rg_lut]
    incidence_angle_rad: np.ndarray
        Incidence angle in [rad], of reference acquisition.
        Is the incidence angle on inflated ellipsoid,
        so to be corrected with slope in this code
        Dimensions [N_az_lut x N_rg_lut]
    vertical_wavenumber_list: List[np.ndarray]
        list of M vertical wavenumbers
        the one corresponding to primary image index should be all zeros by definition
        each of dimensions [N_az_lut x N_rg_lut]
    lut_axis_sr_s: np.ndarray
        Slant range axis valid for all the above luts
        (denoising, sigma nought, terrain slope, incidence angle and vertical wavenumbers)
        of dimensions [N_rg_lut,]
    lut_axis_az_s: np.ndarray
        Azimuth axis valid for all the above luts
        (denoising, sigma nought, terrain slope, incidence angle and vertical wavenumbers)
        of dimensions [N_az_lut,]
    b_rg: float
        L1c range bandwidth [Hz]
    b_az: float
        L1c azimuth bandwidth [Hz]
    average_az_velocity: float
        Average azimuth velocity [m/s]
    snr_decorr_compensation_flag: bool
        Flag to enable SNR decorrelation compensation
    product_resolution: float
        Value in [m] to be used as the resolution on ground range map
        and also to perform the coherence averaging in radar coordinates
    upsampling_factor: int
        Upsampling factor for covariance, in azimuth and range directions

    Returns
    -------
    coherence_for_baseline_combinations: np.ndarray
        Computed coherence for each baseline combination; details:
        from the whole Multi Baseline Multi Polarimetric coherence matrixm of dimensions
        [(PxM) x (PxM) x N_az x N_rg], this output contains only the sub polarimetric PxP matrices
        corresponding to the baselines combinations.
        Baselines combinations are C=3 in the nominal case of M=3: 01, 02, 12
        Baselines combination is C=1 in the contingency case of M=2: 01
        The output dimension is: [ P x P x C x N_az_coherence x N_rg_coherence]
    vertical_wavenumber_list: List[np.ndarray]
        M vertical Wavenumbers, terrain corrected and interpolated on the choerence output grid
        Each of dimensions [N_az_coherence x N_rg_coherence]
    terrain_corrected_incidence_angle_rad: np.ndarray
        Incidence angle[rad], terrain corrected with the slope application, and interpolated
        on the choerence output grid; dimensions [N_az_coherence x N_rg_coherence]
    terrain_slope_rad
        Terrain slope [rad] interpolated on the choerence output grid;
        dimensions [N_az_coherence x N_rg_coherence]
    resolution_rg: float
        Slant range resolution [m], computed from system bandwidth
    axis_az_subsampling_indexes: np.ndarray
        Azimuth indices to be applied to the input_data_list axis of dimensions [N_az x N_rg],
        to get the decimated mpmb_covariance sampling of dimensions [Naz_subsampled x Nrg_subsampled]
    axis_rg_subsampling_indexes: np.ndarray
        Range indices to be applied to the input_data_list axis of dimensions [N_az x N_rg],
        to get the decimated mpmb_covariance sampling of dimensions [Naz_subsampled x Nrg_subsampled]
    mbmp_coherence: np.ndarray
        coherence is an intermediate calculation, returned for UT testing
        same dimensions of the Multi Baseline Multi Polarimetric coherence matrix
        [(PxM) x (PxM) x N_az x N_rg]
    """

    start_time = datetime.now()
    bps_logger.info("Compute coherence estimation:")
    bps_logger.info(f"    using AUX PP2 2A product resolution: {product_resolution} [m]")
    bps_logger.info(f"    using AUX PP2 2A upsampling factor: {upsampling_factor}")

    num_pols = len(scs_pol_list)
    num_imms = len(scs_pol_list[0])

    # Compute averraging windows sizes, decimation factors and number of looks
    (averaging_window_size_az, averaging_window_size_rg, _) = averaging_windows_sizes(
        b_az,
        b_rg,
        1 / (scs_axis_az_s[1] - scs_axis_az_s[0]),
        1 / (scs_axis_sr_s[1] - scs_axis_sr_s[0]),
        product_resolution,
        average_az_velocity,
        np.nanmean(incidence_angle_rad),
    )

    resolution_rg = LIGHTSPEED / (2 * b_rg)

    decimation_factor_rg = np.ceil(averaging_window_size_rg / upsampling_factor).astype(np.uint8)
    decimation_factor_az = np.ceil(averaging_window_size_az / upsampling_factor).astype(np.uint8)
    bps_logger.info(f"    decimation factor used in range direction: {decimation_factor_rg}")
    bps_logger.info(f"    decimation factor used in azimuth direction: {decimation_factor_az}")

    # prepare sparse matriced for the MPMB covariance estimation
    (
        fa_normalized,
        axis_az_subsampling_indexes,
        fr_normalized_transposed,
        axis_rg_subsampling_indexes,
    ) = build_filtering_sparse_matrices(
        scs_pol_list[0][0].shape[0],  # azimuth shape
        scs_pol_list[0][0].shape[1],  # slant range shape
        averaging_window_size_rg,
        decimation_factor_rg,
        averaging_window_size_az,
        decimation_factor_az,
    )
    num_az_subsampled = axis_az_subsampling_indexes.size
    num_rg_subsampled = axis_rg_subsampling_indexes.size

    # Polarimetric-interferometric covariance estimation
    # note that Input to mpmb_covariance_estimation should be a list of pols containing a list of acq
    mpmb_covariance = mpmb_covariance_estimation(
        scs_pol_list,
        fa_normalized,
        num_az_subsampled,
        fr_normalized_transposed,
        num_rg_subsampled,
    )

    # Shuffle is to pass
    # from multi polarimetric - multi baseline, to multi baseline - multi polarimetric
    mbmp_covariance = _mpmb_shuffle(
        mpmb_covariance,
        mpmb_covariance.shape[2],
        mpmb_covariance.shape[3],
        num_pols,
        num_imms,
    )

    # Normalizing covariance matrix
    bps_logger.info("    normalization, covariance to correlation")
    mbmp_coherence, mbmp_coherence_diag = covariance_4d_to_correlation_4d(mbmp_covariance)

    # From current 9x9 coherence matrix,
    # get the three 3x3 entries down the diagonal,
    # one for each baseline combination, in the order 01,02,12
    coherence_for_baseline_combinations = extract_coherence_for_baseline_combinations(mbmp_coherence, num_imms)

    # Vertical wavenumbers terrain correction
    bps_logger.info("    vertical wavenumbers terrain correction")
    if not incidence_angle_rad.shape == vertical_wavenumber_list[0].shape:
        raise ValueError("    vertical wavenumbers and incidence angle LUTs are not on the same grid")

    for idx_acq, wavenumber_curr in enumerate(vertical_wavenumber_list):
        vertical_wavenumber_list[idx_acq] = (
            wavenumber_curr * np.sin(incidence_angle_rad) / np.sin(incidence_angle_rad - terrain_slope_rad)
        )

    if snr_decorr_compensation_flag:
        bps_logger.info("    applying SNR decorrelation compensation, as from AUX PP2 2A configuration")

        coherence_for_baseline_combinations = snr_decorrelation_compensation(
            coherence_for_baseline_combinations,
            mbmp_coherence_diag,
            scs_axis_az_s[axis_az_subsampling_indexes],
            scs_axis_sr_s[axis_rg_subsampling_indexes],
            denoising_pol_list,
            sigma_nought_list,
            lut_axis_az_s,
            lut_axis_sr_s,
        )
    else:
        bps_logger.info("    SNR decorrelation compensation disabled, as from AUX PP2 2A configuration")

    # Preliminary LUTs interpolation
    # Preliminary interpolation: wavenumbers
    bps_logger.info(
        f"    discarding vertical wavenumber values out of AUX PP2 2A limits [{vertical_wavenumber_valid_values_limits.min}, {vertical_wavenumber_valid_values_limits.max}] [rad/m]:"
    )  # this is done inside the for cycles above
    for idx_acq in range(len(vertical_wavenumber_list)):
        # Moved here this check, first it was in height estimation cycle

        num_removed_pixels = 0
        if not np.nanmean(np.abs(vertical_wavenumber_list[idx_acq])) < 0.00001:  # searching the reference
            # The reference acquisition having all zero wavenumbers should not be passed through here to avoid discarding all values
            num_removed_pixels = np.sum(
                np.logical_or(
                    np.abs(vertical_wavenumber_list[idx_acq]) < vertical_wavenumber_valid_values_limits.min,
                    np.abs(vertical_wavenumber_list[idx_acq]) > vertical_wavenumber_valid_values_limits.max,
                )
            )

            # Update vertical_wavenumber_list[idx_acq], setting to NaN all values < of the min (in abs)
            vertical_wavenumber_list[idx_acq] = np.where(
                np.abs(vertical_wavenumber_list[idx_acq]) < vertical_wavenumber_valid_values_limits.min,
                np.nan,
                vertical_wavenumber_list[idx_acq],
            )

            # Update vertical_wavenumber_list[idx_acq], setting to NaN all values > of the max (in abs)
            vertical_wavenumber_list[idx_acq] = np.where(
                np.abs(vertical_wavenumber_list[idx_acq]) > vertical_wavenumber_valid_values_limits.max,
                np.nan,
                vertical_wavenumber_list[idx_acq],
            )

        bps_logger.warning(
            f"        {num_removed_pixels / vertical_wavenumber_list[idx_acq].size * 100:2.3f}% of vertical wavenumbers pixels removed for acquisition {idx_acq + 1} of {len(vertical_wavenumber_list)}"
        )

    if not (
        len(axis_az_subsampling_indexes) == vertical_wavenumber_list[0].shape[0]
        and len(axis_rg_subsampling_indexes) == vertical_wavenumber_list[0].shape[1]
    ):
        bps_logger.info(
            "    preliminary interpolation to bring vertical wavenumbers onto the computed correlation grid"
        )
        vertical_wavenumber_list = parallel_reinterpolate(
            vertical_wavenumber_list,
            lut_axis_az_s,
            lut_axis_sr_s,
            scs_axis_az_s[axis_az_subsampling_indexes],
            scs_axis_sr_s[axis_rg_subsampling_indexes],
        )

    # Preliminary interpolation: reference incidence angle
    if not (
        len(axis_az_subsampling_indexes) == incidence_angle_rad.shape[0]
        and len(axis_rg_subsampling_indexes) == incidence_angle_rad.shape[1]
    ):
        bps_logger.info(
            "    preliminary interpolation to bring reference incidence angle onto the computed correlation grid"
        )
        incidence_angle_rad = parallel_reinterpolate(
            [incidence_angle_rad],  # function works with lists
            lut_axis_az_s,
            lut_axis_sr_s,
            scs_axis_az_s[axis_az_subsampling_indexes],
            scs_axis_sr_s[axis_rg_subsampling_indexes],
        )[0]  # function works with lists

    # Preliminary interpolation: reference terrain slope
    if not (
        len(axis_az_subsampling_indexes) == terrain_slope_rad.shape[0]
        and len(axis_rg_subsampling_indexes) == terrain_slope_rad.shape[1]
    ):
        bps_logger.info(
            "    preliminary interpolation to bring reference terrain slope onto the computed correlation grid"
        )
        terrain_slope_rad = parallel_reinterpolate(
            [terrain_slope_rad],  # function works with lists
            lut_axis_az_s,
            lut_axis_sr_s,
            scs_axis_az_s[axis_az_subsampling_indexes],
            scs_axis_sr_s[axis_rg_subsampling_indexes],
        )[0]  # function works with lists

    bps_logger.info(
        f"    final coherence dimensions: azimuth {coherence_for_baseline_combinations.shape[3]} pixels, range {coherence_for_baseline_combinations.shape[4]} pixels"
    )
    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Coherence estimation processing time: {elapsed_time:2.1f} s")
    return (
        coherence_for_baseline_combinations,
        vertical_wavenumber_list,
        incidence_angle_rad - terrain_slope_rad,
        terrain_slope_rad,
        resolution_rg,
        axis_az_subsampling_indexes,
        axis_rg_subsampling_indexes,
        mbmp_coherence,
    )


def covariance_4d_to_correlation_4d(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    It normalizes each element of the 4-D Multi-Polarimetric Multi-Baseline
    (or  Multi-Baseline Multi-Polarimetric)
    with respect to the corresponding diagonal terms.

    Parameters
    ----------
    covariance: np.ndarray
    [(PxM) x (PxM) x N_az x N_rg] covariance matrices
        P,number of polarizations
        M, number of acquisitions

    Returns
    -------
    correlation: np.ndarray
        [(PxM) x (PxM) x N_az x N_rg] normalized covariance matrices
    covariance_diagonal: np.ndarray
        N_az x N_rg x (PxM))] diagonal entries of input covariance
    """

    # Compute the mask to extract diagonal from matrix:
    _, p_x_m, num_az, num_rg = covariance.shape
    diag_mask = np.tile((np.eye(p_x_m).reshape(p_x_m, p_x_m, 1, 1)) > 0, [1, 1, num_az, num_rg])

    # Normalization factor computed from the covariance diagonal
    normalization_factor = covariance[diag_mask]
    normalization_factor = 1 / np.sqrt(normalization_factor + np.spacing(1))
    normalization_factor = normalization_factor.reshape((p_x_m, 1, num_az, num_rg)) * normalization_factor.reshape(
        (1, p_x_m, num_az, num_rg)
    )

    # Normalization of the covariance to get coherence
    coherence = normalization_factor * covariance

    # the function returns also the reshaped covariance diagonal, also moveaxis to get:
    # N_az x N_rg x (MxP))]
    #     It is from Multi Baseline - Multi Polarimetric coherence, so
    #     for each azimuth and range, the diagonal is ordered as (case of P=3, M=3)
    #     [M0_P0, M0_P1, M0_P2, M1_P0, M1_P1, M1_P2, M2_P0, M2_P1, M2_P2]
    covariance_diagonal = np.moveaxis(covariance[diag_mask].reshape((p_x_m, num_az, num_rg)), [1, 2, 0], [0, 1, 2])

    return coherence, covariance_diagonal


def _mpmb_shuffle(mpmb_coherence: np.ndarray, n_az: int, n_rg: int, num_pols: int, num_imms: int) -> np.ndarray:
    """
    Internal function to convert an input matrix
    from Multi Polarimetric - Multi Baseline,
        where the matrix is a P x P where each element is a sub matrix M x M
    to Multi Baseline - Multi polarimetric,
        where the matrix is a M x M where each element is a sub matrix P x P
    Note: diagonal is not shuffled in this function,
    it is shuffled after in covariance_4d_to_correlation_4d
    """
    mbmp_coherence = np.zeros((num_pols * num_imms, num_pols * num_imms, n_az, n_rg), dtype=np.complex64)
    r = mpmb_coherence.shape[0]
    II = np.eye(r)  # Identity matrix
    SS = II * 0
    for i in range(num_imms):
        SS[i * num_pols : num_pols + i * num_pols, :] = II[i:r:num_imms, :]

    for idx_az in range(n_az):
        for idx_rg in range(n_rg):
            mbmp_coherence[:, :, idx_az, idx_rg] = SS @ mpmb_coherence[:, :, idx_az, idx_rg] @ SS.T

    return mbmp_coherence


def extract_coherence_for_baseline_combinations(mbmp_coherence: np.ndarray, num_imms: int) -> np.ndarray:
    """
    From current (MxP)x(MxP) coherence matrix,
    get the three PxP entries down the diagonal,
    one for each baseline combination, in the order 01,02,12

    Parameters
    ----------
    mbmp_coherence:
        Multi Baseline - Multi Polarimetric
        whole coherence matrix of dimensions [(M x P)x(M x P) x N_az x N_rg]
        It is an MxM matrix, where each element is a PxP sub matrix
    num_imms: int
        Number of images in the input matrix (it is "M")

    Returns
    ------
    coherence_for_baseline_combinations
        [PxP x N x N_az x N_rg]
        Where
        P, number of polarizations
        N number of baselines combinations:
            the ordering of combinations is
            N = 0 -> Baselines 0-1
            N = 1 -> Baselines 0-2 (present only if num_imms = 3)
            N = 2 -> Baselines 1-2 (present only if num_imms = 3)
    """

    num_pols = int(mbmp_coherence.shape[0] / num_imms)
    num_baselines_combinations = int((num_imms * (num_imms - 1)) / 2)
    num_az_subsampling_indexes = mbmp_coherence.shape[2]
    num_rg_subsampling_indexes = mbmp_coherence.shape[3]

    coherence_for_baseline_combinations = np.zeros(
        (
            num_pols,
            num_pols,
            num_baselines_combinations,
            num_az_subsampling_indexes,
            num_rg_subsampling_indexes,
        ),
        dtype=np.complex64,
    )

    idx_baseline_combination = 0
    for idx_imm_n in range(num_imms - 1):
        for idx_imm_m in np.arange(idx_imm_n + 1, num_imms):
            coherence_for_baseline_combinations[
                :,
                :,
                idx_baseline_combination,
                :,
                :,
            ] = mbmp_coherence[
                idx_imm_n * num_pols : num_pols + idx_imm_n * num_pols,
                idx_imm_m * num_pols : num_pols + idx_imm_m * num_pols,
                :,
                :,
            ]
            idx_baseline_combination += 1

    return coherence_for_baseline_combinations


def snr_decorrelation_compensation(
    coherence_for_baseline_combinations: np.ndarray,
    mbmp_coherence_diag: np.ndarray,
    coherence_axis_az_s: np.ndarray,
    coherence_axis_sr_s: np.ndarray,
    denoising_pol_list: list[list[np.ndarray]],
    sigma_nought_list: list[np.ndarray],
    lut_axis_az_s: np.ndarray,
    lut_axis_sr_s: np.ndarray,
) -> np.ndarray:
    """
    SNR decorrelation compensation

    Parameters
    ----------
    coherence_for_baseline_combinations: np.ndarray
        [PxP x N x N_az x N_rg]
        Where
        P, number of polarizations
        N number of baselines combinations:
            the ordering of combinations is
            N = 0 -> Baselines 0-1
            N = 1 -> Baselines 0-2 (present only if num_imms = 3)
            N = 2 -> Baselines 1-2 (present only if num_imms = 3)
    mbmp_coherence_diag: np.ndarray,
        N_az x N_rg x (MxP))] diagonal entries of input covariance
        It is from Multi Baseline - Multi Polarimetric coherence, so
            for each azimuth and range, the diagonal is ordered as (case of P=3, M=3)
            [M0_P0, M0_P1, M0_P2, M1_P0, M1_P1, M1_P2, M2_P0, M2_P1, M2_P2]
    coherence_axis_az_s: np.ndarray,
        Azimuth axis of the input coherence [s] (subsampled axis of the input SCS data one)
    coherence_axis_sr_s: np.ndarray,
        Slant range axis of the input coherence [s] (subsampled axis of the input SCS data one)
    denoising_pol_list: List[List[np.ndarray]]:
        List denoising LUT for each P polarization,
        each containing a sub list with M elements, one for each acquisitions
    sigma_nought_list: List[np.ndarray],
        Sigma nought LUT list, M elements, one for each acquisition
    lut_axis_az_s: np.ndarray,
        Azimuth axis for each of the above Luts, denoising and sigma nought [s]
    lut_axis_sr_s: np.ndarray,
        Slant range axis for each of the above Luts, denoising and sigma nought [s]

    Returns
    -------
    coherence_for_baseline_combinations: np.ndarray
    SNR compensated coherence_for_baseline_combinations
    """

    num_pols = len(denoising_pol_list)
    num_imms = len(denoising_pol_list[0])
    num_az_coherence = coherence_for_baseline_combinations.shape[3]
    num_rg_coherence = coherence_for_baseline_combinations.shape[4]

    # Preliminary interpolation: noise luts
    if not (
        num_az_coherence == denoising_pol_list[0][0].shape[0] and num_rg_coherence == denoising_pol_list[0][0].shape[1]
    ):
        bps_logger.info("    preliminary interpolation to bring denoising luts onto the computed correlation grid")
        for idx_pol, denoising_acq_list in enumerate(denoising_pol_list):
            denoising_pol_list[idx_pol] = parallel_reinterpolate(
                denoising_acq_list,
                lut_axis_az_s,
                lut_axis_sr_s,
                coherence_axis_az_s,
                coherence_axis_sr_s,
            )

        bps_logger.info("    preliminary interpolation to bring sigma nought lut onto the computed correlation grid")
        sigma_nought_list = parallel_reinterpolate(
            sigma_nought_list,
            lut_axis_az_s,
            lut_axis_sr_s,
            coherence_axis_az_s,
            coherence_axis_sr_s,
        )

    # Compute SNR matrix, for all polarizations and all acquisitions
    snr = np.zeros(
        (
            num_pols,
            num_imms,
            num_az_coherence,
            num_rg_coherence,
        ),
        dtype=type(coherence_for_baseline_combinations[0, 0, 0, 0, 0]),
    )
    # mbmp_coherence_diag contains
    # [M0_P0, M0_P1, M0_P2, M1_P0, M1_P1, M1_P2, M2_P0, M2_P1, M2_P2]
    for idx_acq in range(num_imms):
        for idx_pol, denoising_acq_list in enumerate(denoising_pol_list):
            snr[idx_pol, idx_acq, :, :] = (
                mbmp_coherence_diag[:, :, idx_acq * num_imms + idx_pol] * sigma_nought_list[idx_acq]
                - denoising_acq_list[idx_acq]
            ) / denoising_acq_list[idx_acq]

    # Compute Gamma SNR, for each polarization and for each baseline combination
    # Than apply the Gamma SNR to the mbmp_coherence
    num_baselines_combinations = int((num_imms * (num_imms - 1)) / 2)  # result is 1 or 3

    # Considering this M x M matrix (mbmp):
    #      m=0 m=1 m=2
    #      -----------
    # m=0: |0   1   2|
    # m=1: |3   4   5|
    # m=2: !6   7   8|
    # (where each number corresponds to a P x P sub matrix)
    # combination [0,1] extracts entry "1" PxP sub matrix
    # combination [0,2] extracts entry "2" PxP sub matrix
    # combination [1,2] extracts entry "5" PxP sub matrix
    # Similar explanation in case of M=2
    baselines_combinations = [[0, 1], [0, 2], [1, 2]] if num_baselines_combinations == 3 else [[0, 1]]
    eps = 1.0e-010  # to avoid divide by zero
    for idx_pol_p in range(num_pols):
        for idx_pol_q in range(num_pols):
            baseline_combination_idx = 0
            for idx_bas_vec in baselines_combinations:
                idx_bas_n = idx_bas_vec[0]
                idx_bas_m = idx_bas_vec[1]
                gamma_snr = 1 / (
                    np.sqrt(
                        (1 + 1 / (snr[idx_pol_p, idx_bas_n, :, :] + eps))
                        * (1 + 1 / (snr[idx_pol_q, idx_bas_m, :, :] + eps))
                    )
                )
                # apply gamma_snr to each element out of the diagonal
                coherence_for_baseline_combinations[
                    idx_pol_p,
                    idx_pol_q,
                    baseline_combination_idx,
                    :,
                    :,
                ] = (
                    coherence_for_baseline_combinations[
                        idx_pol_p,
                        idx_pol_q,
                        baseline_combination_idx,
                        :,
                        :,
                    ]
                    / gamma_snr
                )
                baseline_combination_idx += 1

    return coherence_for_baseline_combinations


def forest_height_inversion(
    coherence_for_baseline_combinations: np.ndarray,
    resolution_rg: float,
    vertical_wavenumbers_terrain_corrected: list[np.ndarray],
    terrain_slope_rad: np.ndarray,
    inc_angle_terrain_corrected_rad: np.ndarray,
    residual_decorrelation: float,
    vertical_reflectivity_default_profile: np.ndarray,
    vertical_reflectivity_option: str,
    model_inversion: str,
    normalised_height_estimation_range: MinMaxType,
    normalised_wavenumber_estimation_range: MinMaxNumType,
    ground_to_volume_ratio_range: MinMaxNumType,
    temporal_decorrelation_estimation_range: MinMaxNumType,
    spectral_decorr_compensation_flag: bool,
    correct_terrain_slopes_flag: bool,
    temp_decorr_gv_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forest Height inversion

    Parameters
    ----------
    coherence_for_baseline_combinations: np.ndarray
        [PxP x N x N_az_coherence x N_rg_coherence]
        Where
        P, number of polarizations
        N number of baselines combinations:
            the ordering of combinations is
            N = 0 -> Baselines 0-1
            N = 1 -> Baselines 0-2 (present only if num_imms = 3)
            N = 2 -> Baselines 1-2 (present only if num_imms = 3)
    resolution_rg: float
        Slant range resolution [m], computed from system bandwidth
    vertical_wavenumbers_terrain_corrected: List[np.ndarray],
        vertical wavenumbers terrain corrected, one for each M acquisition
        already interpolated over the N_az_coherence x N_rg_coherence dimensions
    terrain_slope_rad: np.ndarray,
        Terrain slope [rad]
        already interpolated over the N_az_coherence x N_rg_coherence dimensions
    inc_angle_terrain_corrected_rad: np.ndarray,
        Incidence angle, terrain corrected with the slope [rad] and
        already interpolated over the N_az_coherence x N_rg_coherence dimensions
    residual_decorrelation: float
        Residual non-volumetric decorrelation to be used in error model computation
    vertical_reflectivity_default_profile: np.ndarray,
        Modeled tomographic volume-only reflectivity profile normalized to unit height
    vertical_reflectivity_option: str
        Not used in this software version.
        Specify which vertical reflectivity profile to use: default modelled profile
        or tomographic profile (out of BPS scope)
    model_inversion: str
        Model inversion algorithm to be used among single or dual baseline
    normalised_height_estimation_range: MinMaxType
        Range of normalized height values where the canopy height estimation process has to be performed
    normalised_wavenumber_estimation_range: MinMaxNumType,
        Range of normalized wavenumbers values where the canopy height estimation process has to be performed:
        Num to set sensitivity of 1 [m] for smaller k_z value of 0.05 [1/m]
    ground_to_volume_ratio_range: MinMaxNumType,
        Range of ground to volume ratio values to be used as valid ones [dB]
    temporal_decorrelation_estimation_range: MinMaxNumType,
        Range of temporal decorrelation values to be used as valid ones
        FH estimates lower this limit [m] are discarded and set to no data value
    spectral_decorr_compensation_flag: bool
        Flag to enable range spectral decorrelation
    correct_terrain_slopes_flag: bool
        Flag to enable terrain slopes correction
    temp_decorr_gv_ratio: float
        Ratio of temporal decorrelation between ground and volume (0.0 means no temporal decorrelation for ground,
        while 1.0 means ground and volume are equally impacted by temporal decorrelation)

    Returns
    -------
    forest_height: np.ndarray
        Estimated forest height [m], of dimensions [N_az_coherence, N_rg_coherence]
    forest_height_quality: np.ndarray
        Forest height quality, of dimensions [N_az_coherence, N_rg_coherence]
    """

    bps_logger.info("Compute Forest Height inversion:")
    start_time = datetime.now()

    # check model_inversion:
    if model_inversion == "dual" and len(vertical_wavenumbers_terrain_corrected) == 2:
        bps_logger.warning(
            "    AUX PP2 2A model inversion set to dual baseline, but input L1c contains only one baseline (two acquisitions)"
        )
        bps_logger.warning("    setting model inversion to single baseline")
        model_inversion = "single"

    else:
        bps_logger.info(f"    using AUX PP2 2A model inversion {model_inversion} baseline")

    bps_logger.info(f"    using AUX PP2 2A residual decorrelation: {residual_decorrelation}")
    bps_logger.info(f"    using AUX PP2 2A temporal decorrelation ground to volume ratio: {temp_decorr_gv_ratio}")
    # calculation of the LUT
    # vertical_reflectivity_option still TBD
    (
        lut_kh_mu_temp,
        lut_kh_axis,
        _,
        _,
        lut_kh_mu,
        lut_kh,
    ) = compute_lut_kh_mu_tempdec(
        vertical_reflectivity_default_profile,
        normalised_wavenumber_estimation_range,
        ground_to_volume_ratio_range,
        normalised_height_estimation_range,
        temporal_decorrelation_estimation_range,
        temp_decorr_gv_ratio,
    )

    # Biases is a vector containing all the possible quality indexes for each kh value
    sign_real = np.sign(1 - np.real(lut_kh_mu[:, 0]))
    sign_real[sign_real == 0] = 1
    sign_imag = np.sign(np.imag(lut_kh_mu[:, 0]))
    sign_imag[sign_imag == 0] = 1
    lut_kh_arctan = (
        sign_real
        * sign_imag
        * np.arctan2(
            np.abs(1 - np.real(lut_kh_mu[:, 0])),
            np.abs(np.imag(lut_kh_mu[:, 0])),
        )
    )

    biases = compute_biases(
        lut_kh_arctan,
        lut_kh_axis,
        lut_kh,
        residual_decorrelation,
    )

    # Initializations, than call the height estimation core (single or dual baseline)

    # compute wavenumber differences for each baseline combination
    # from here, each matrix is a matrix [num_imm, num_az, num_rg]
    num_imms = len(vertical_wavenumbers_terrain_corrected)
    if num_imms == 3:
        vertical_wavenumbers_stacked = np.stack(
            (
                vertical_wavenumbers_terrain_corrected[0] - vertical_wavenumbers_terrain_corrected[1],
                vertical_wavenumbers_terrain_corrected[0] - vertical_wavenumbers_terrain_corrected[2],
                vertical_wavenumbers_terrain_corrected[1] - vertical_wavenumbers_terrain_corrected[2],
            ),
            axis=0,
        )
    elif num_imms == 2:
        # Maintain the shape as NumBaselines x Naz x Nrg even if first dim is "1"
        vertical_wavenumbers_stacked = np.zeros(
            (
                1,
                vertical_wavenumbers_terrain_corrected[0].shape[0],
                vertical_wavenumbers_terrain_corrected[0].shape[1],
            ),
            dtype=np.float32,
        )
        vertical_wavenumbers_stacked[0, :, :] = (
            vertical_wavenumbers_terrain_corrected[0] - vertical_wavenumbers_terrain_corrected[1]
        )

    # no data values mask, to skip computations where not needed
    nan_mask = np.logical_or(
        np.any(np.isnan(vertical_wavenumbers_stacked), axis=0),
        np.any(np.isnan(coherence_for_baseline_combinations), axis=(0, 1, 2)),
    )

    # Polarimetric optimization (3.12)
    # from  [PxP x N x N_az_coherence x N_rg_coherence] to  [N x N_az_coherence x N_rg_coherence]
    optimized_coherence_mat = polarimetric_optimization(coherence_for_baseline_combinations)

    # Range decorrelation compensation (not for error model computation) (3.8)
    if spectral_decorr_compensation_flag:
        bps_logger.info("    applying spectral decorrelation compensation, as from AUX PP2 2A configuration")
        optimized_coherence_mat = range_decorrelation_compensation(
            optimized_coherence_mat,
            vertical_wavenumbers_stacked,
            inc_angle_terrain_corrected_rad,
            resolution_rg,
        )
    else:
        bps_logger.info("    spectral decorrelation compensation disabled, as from AUX PP2 2A configuration")

    # Remove values of coherences larger than 1
    optimized_coherence_abs = np.abs(optimized_coherence_mat)
    optimized_coherence_mat = np.where(
        optimized_coherence_abs > 1.0,
        optimized_coherence_mat / optimized_coherence_abs,
        optimized_coherence_mat,
    )

    if model_inversion == "dual":
        # Automatic baseline selection: select 2 baselines, min and max
        idx_sort = np.argsort(np.abs(vertical_wavenumbers_stacked), axis=0)
        vertical_wavenumber_min_mat = np.take_along_axis(
            vertical_wavenumbers_stacked, (idx_sort[0, :, :])[np.newaxis, :], axis=0
        )[0]
        vertical_wavenumber_max_mat = np.take_along_axis(
            vertical_wavenumbers_stacked, (idx_sort[2, :, :])[np.newaxis, :], axis=0
        )[0]
        coherence_kz_min_mat = np.take_along_axis(optimized_coherence_mat, (idx_sort[0, :, :])[np.newaxis, :], axis=0)[
            0
        ]
        coherence_kz_max_mat = np.take_along_axis(optimized_coherence_mat, (idx_sort[2, :, :])[np.newaxis, :], axis=0)[
            0
        ]

        # ratio between wavenumbers
        wavenumbers_ratio_mat = vertical_wavenumber_min_mat / vertical_wavenumber_max_mat
        # put no data values to nan here, to avoid carring the no data mask
        wavenumbers_ratio_mat[nan_mask] = np.nan

        # find minimum position for each range, azimuth pixel
        min_pos_mat = np.zeros(wavenumbers_ratio_mat.shape, dtype=np.uint8)
        find_minimums(
            min_pos_mat,
            lut_kh_mu_temp,
            wavenumbers_ratio_mat,
            coherence_kz_min_mat,
            coherence_kz_max_mat,
        )

        forest_height_quality = biases[min_pos_mat]

        kh_mat = lut_kh_axis[min_pos_mat]
        forest_height = kh_mat / np.abs(vertical_wavenumber_max_mat)
        forest_height_debug = None

    elif model_inversion == "single":
        height_mat = np.zeros(vertical_wavenumbers_stacked.shape, dtype=np.float32)
        quality_mat = np.zeros(vertical_wavenumbers_stacked.shape, dtype=np.float32)
        for idx_baseline in range(vertical_wavenumbers_stacked.shape[0]):
            coh = optimized_coherence_mat[idx_baseline, :, :]

            sign_real = np.sign(1 - np.real(coh))
            sign_real[sign_real == 0] = 1
            sign_imag = np.sign(np.imag(coh))
            sign_imag[sign_imag == 0] = 1
            arctangens_mat = sign_real * sign_imag * np.arctan2(np.abs(1 - np.real(coh)), np.abs(np.imag(coh)))

            (
                height_mat[idx_baseline, :, :],
                quality_mat[idx_baseline, :, :],
            ) = height_estimation_single_bas(
                vertical_wavenumbers_stacked[idx_baseline, :, :],
                arctangens_mat,
                lut_kh_arctan,
                lut_kh_axis,
                nan_mask,
                biases,
            )

        forest_height = np.nansum(1 / (0.01 + quality_mat) * height_mat, axis=0) / np.nansum(
            1 / (0.01 + quality_mat), axis=0
        )
        forest_height_quality = np.nansum(1 / (0.01 + quality_mat) * quality_mat, axis=0) / np.nansum(
            1 / (0.01 + quality_mat), axis=0
        )

        # forest_height_debug = height_mat[1, :, :] / np.cos(terrain_slope_rad)
        forest_height_debug = None

    if correct_terrain_slopes_flag:
        bps_logger.info("    applying terrain slopes correction, as from AUX PP2 2A configuration")
        forest_height = forest_height / np.cos(terrain_slope_rad)
    else:
        bps_logger.info("    terrain slopes correction disabled, as from AUX PP2 2A configuration")

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Forest height inversion processing time: {elapsed_time:2.1f} s")

    return forest_height, forest_height_quality, forest_height_debug


def compute_biases(
    lut_kh_arctan: np.ndarray,
    lut_kh_axis: np.ndarray,
    lut_kh: np.ndarray,
    residual_decorrelation: float,
) -> np.ndarray:
    """
    Calculate profile dependent paramteter biases vector
    It is the quality index, computed for each possible normalized wavenumber ("kh") value

    Parameters
    ----------
    lut_kh_arctan: np.ndarray,
        arctan of the look-up table of complex coherences for the normalized wavenumbers values ("kh")
        (its the first column of the whole "kh_mu_tempdecorr" matrix, see compute_lut_kh_mu_tempdec)
    lut_kh_axis: np.ndarray,
        Lut normalized wavenumbers axis
    lut_kh:
        Lut 1D where ground_to_volume_ratio_range = 0 and  temporal_decorrelation_estimation_range = 1
    residual_decorrelation: float
        Residual non-volumetric decorrelation to be used in error model computation

    Returns
    -------
    biases: np.ndarray
        vector of biases, in percentage (all possible modelled height quality indices),
        for each value of modelled normalized wavenumbers
    """

    # error model paramters calculation
    height_inverted = error_model_kzh(
        lut_kh_arctan,
        lut_kh_axis,
        lut_kh,
        residual_decorrelation,
    )

    # biases vector for every kz*H value in percenantage
    # lut_kh_axis is the real height
    biases = np.abs(height_inverted.astype(np.float64) - lut_kh_axis) / lut_kh_axis * 100
    # Biases is a percentage, max values greater than 100 are here saturated.
    biases[biases > 100.0] = 100.0

    return biases


def error_model_kzh(
    lut_kh_arctan: np.ndarray,
    lut_kh_axis: np.ndarray,
    lut_kh: np.ndarray,
    residual_decorrelation: float,
) -> np.ndarray:
    """
    Error model computation, from single baseline aproach

    Parameters
    ----------
    lut_kh_arctan: np.ndarray,
        arctan of the look-up table of complex coherences for the normalized wavenumbers values ("kh")
        (its the first column of the whole "kh_mu_tempdecorr" matrix, see compute_lut_kh_mu_tempdec)
    lut_kh_axis: np.ndarray,
        Lut normalized wavenumbers axis
    lut_kh:
        Lut 1D where ground_to_volume_ratio_range = 0 and  temporal_decorrelation_estimation_range = 1
    residual_decorrelation: float
        Residual non-volumetric decorrelation to be used in error model computation

    Returns
    -------
    height_inverted: np.ndarray
        vector of estimated heights over KzH values, assuming decorrelation
    """

    height_inverted = np.zeros(lut_kh_arctan.shape)
    for idx_kh, lut_kh_value in enumerate(lut_kh_axis):
        optimized_coherence = lut_kh[idx_kh] * residual_decorrelation

        sign_real = np.sign(1 - np.real(optimized_coherence)) if np.abs(1 - np.real(optimized_coherence)) > 0 else 1
        sign_imag = np.sign(np.imag(optimized_coherence)) if np.abs(np.imag(optimized_coherence)) > 0 else 1
        arctangens = (
            sign_real
            * sign_imag
            * np.arctan2(
                np.abs(1 - np.real(optimized_coherence)),
                np.abs(np.imag(optimized_coherence)),
            )
        )

        # calcluating result for the given decorrelation level
        (
            height_inverted[idx_kh],
            _,
        ) = _height_estimation_single_bas_core(
            arctangens,
            lut_kh_arctan,
            lut_kh_axis,
        )

    return height_inverted


def compute_lut_kh_mu_tempdec(
    profile: np.ndarray,
    normalised_wavenumber_estimation_range: MinMaxNumType,
    ground_to_volume_ratio_range: MinMaxNumType,
    normalised_height_estimation_range: MinMaxType,
    temporal_decorrelation_estimation_range: MinMaxNumType,
    temp_decorr_gv_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3 dimensions look-up table of complex coherences for each of the values of Kz*H and Mu for the given profile
    containing forward modelled complex coherence for the given profile and for each value of:
        normalized vertical wavenumber k_h=k_z*z,
        scalar temporal decorrelation Î³_TempV (ancillary parameter)
        ground-to-volume ratio m (ancillary parameter converted between decibels and a linear scale).


    Parameters
    ----------
    profile: np.ndarray
        volume-only tomographic reflectivity profile of "N" samples
    ground_to_volume_ratio_range: MinMaxNumType,
        Range of ground to volume ratio values to be used as valid ones [dB]
    normalised_wavenumber_estimation_range: MinMaxNumType,
        Range of normalized wavenumbers values where the canopy height estimation process has to be performed:
        Num to set sensitivity of 1 [m] for smaller k_z value of 0.05 [1/m]
    normalised_height_estimation_range: MinMaxType
        Range of normalized height values where the canopy height estimation process has to be performed
    temporal_decorrelation_estimation_range: MinMaxNumType,
        Range of temporal decorrelation values to be used as valid ones


    Returns
    -------
    lut_kh_mu_temp: np.ndarray
        Look.up table, of dimensions
        [ num_normalized_wavenumbars x num_temporal_decorrelation_estimations x
        num_ground_to_volume_ratio]
    lut_kh_axis: np.ndarray
        Lut normalized wavenumbers axis
    lut_mu_axis: np.ndarray
        Lut ground to volume ratio axis
    lut_temp_decorr_axis: np.ndarray
        Lut temporal decorrelation axis
    lut_kh_mu: np.ndarray
        lut_kh_mu
    lut_kh:
        Lut 1D where ground_to_volume_ratio_range = 0 and  temporal_decorrelation_estimation_range = 1
    """

    bps_logger.info("    LUT of complex coherences computation, using AUX PP2 2A:")
    bps_logger.info(f"        default vertical reflectivity profile, composed of #{len(profile)} samples")
    bps_logger.info(
        f"        normalised_height_estimation_range: [{normalised_height_estimation_range.min}, {normalised_height_estimation_range.max}]"
    )
    bps_logger.info(
        f"        normalised wavenumber estimation range: [{normalised_wavenumber_estimation_range.min}, {normalised_wavenumber_estimation_range.max}]"
    )
    bps_logger.info(
        f"        ground_to_volume_ratio_range: [{ground_to_volume_ratio_range.min}, {ground_to_volume_ratio_range.max},{ground_to_volume_ratio_range.num}]"
    )
    bps_logger.info(
        f"        temporal decorrelation estimation range: [{temporal_decorrelation_estimation_range.min}, {temporal_decorrelation_estimation_range.max}, {temporal_decorrelation_estimation_range.num}]"
    )
    # parameters for LUT
    lut_kh_axis = np.linspace(
        normalised_wavenumber_estimation_range.min,
        normalised_wavenumber_estimation_range.max,
        normalised_wavenumber_estimation_range.num,
        dtype=np.float32,
    )
    if lut_kh_axis[0] == 0.0:
        lut_kh_axis[0] = np.min([1.0e-06, lut_kh_axis[1] / 2])

    lut_mu_axis = np.power(
        10.0,
        np.linspace(
            ground_to_volume_ratio_range.min,
            ground_to_volume_ratio_range.max,
            ground_to_volume_ratio_range.num,
            dtype=np.float32,
        )
        / 10.0,
    )

    height_axis_n_samples = len(profile)
    lut_height_axis_norm = np.linspace(
        normalised_height_estimation_range.min,
        normalised_height_estimation_range.max,
        height_axis_n_samples,
        dtype=np.float32,
    )
    lut_temp_decorr_axis = np.linspace(
        temporal_decorrelation_estimation_range.min,
        temporal_decorrelation_estimation_range.max,
        temporal_decorrelation_estimation_range.num,
        dtype=np.float32,
    )

    lut_kh_mu_temp = np.zeros(
        (len(lut_kh_axis), len(lut_mu_axis), len(lut_temp_decorr_axis)),
        dtype=np.complex64,
    )

    for idx_kh, kh_value in enumerate(lut_kh_axis):
        # vector of complex exponentials for a given KzH
        exp_kz = np.exp(1j * kh_value * lut_height_axis_norm)
        # calculating volume only coherence
        gamma_v0 = (np.sum(exp_kz * profile)) / np.abs(np.sum(profile))

        for idx_mu, mu_value in enumerate(lut_mu_axis):
            for idx_decorr, temp_decorr_value in enumerate(lut_temp_decorr_axis):
                # volumetric coherence with ground component
                lut_kh_mu_temp[idx_kh, idx_mu, idx_decorr] = (
                    temp_decorr_value * gamma_v0 + mu_value - temp_decorr_gv_ratio * (1 - temp_decorr_value) * mu_value
                ) / (1 + mu_value)

    lut_kh_mu = np.zeros(
        (len(lut_kh_axis), len(lut_mu_axis)),
        dtype=np.complex64,
    )

    for idx_kh, kh_value in enumerate(lut_kh_axis):
        # vector of complex exponentials for a given KzH
        exp_kz = np.exp(1j * kh_value * lut_height_axis_norm)
        # calculating volume only coherence
        gamma_v0 = (np.sum(exp_kz * profile)) / np.abs(np.sum(profile))

        for idx_mu, mu_value in enumerate(lut_mu_axis):
            # volumetric coherence with ground component
            lut_kh_mu[idx_kh, idx_mu] = (gamma_v0 + mu_value) / (1 + mu_value)

    lut_kh = np.zeros(len(lut_kh_axis), dtype=np.complex64)

    for idx_kh, kh_value in enumerate(lut_kh_axis):
        # vector of complex exponentials for a given KzH
        exp_kz = np.exp(1j * kh_value * lut_height_axis_norm)
        # calculating volume only coherence
        gamma_v0 = (np.sum(exp_kz * profile)) / np.abs(np.sum(profile))
        lut_kh[idx_kh] = gamma_v0

    return (
        lut_kh_mu_temp,
        lut_kh_axis,
        lut_mu_axis,
        lut_temp_decorr_axis,
        lut_kh_mu,
        lut_kh,
    )


def height_estimation_single_bas(
    vertical_wavenumbers_curr_bas_mat: np.ndarray,
    arctangens_curr_bas_mat: np.ndarray,
    lut_kh_arctan: np.ndarray,
    lut_kh_axis: np.ndarray,
    nan_mask: np.ndarray,
    biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FH estimation from single baseline

    Parameters
    ----------
    vertical_wavenumbers_curr_bas_mat: np.ndarray
    arctangens_curr_bas_mat: np.ndarray
    lut_kh_arctan: np.ndarray
    lut_kh_axis: np.ndarray
    nan_mask: np.ndarray
    biases: np.ndarray

    Returns
    -------
    height_2d_mat: np.ndarray
    quality_2d_mat: np.ndarray
    """

    height_2d_mat, quality_2d_mat = _height_estimation_single_bas_core(
        arctangens_curr_bas_mat,
        lut_kh_arctan,
        lut_kh_axis,
        nan_mask,
        biases,
    )
    height_2d_mat = height_2d_mat / np.abs(vertical_wavenumbers_curr_bas_mat)

    return height_2d_mat, quality_2d_mat


def _height_estimation_single_bas_core(
    arctangens_mat: np.ndarray | float,
    lut_kh_arctan: np.ndarray,
    lut_kh_axis: np.ndarray,
    nan_mask: np.ndarray | None = None,
    biases: np.ndarray | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Core of the height estimation, single baseline

    Parameters
    ----------
    vertical_wavenumbers_curr_bas_mat: np.ndarray
    arctangens_curr_bas_mat: Union[np.ndarray, float]
        Matrix in case of height estimation algorithm
        Scalar in case of fuction call during error model computation
    lut_kh_arctan: np.ndarray
    lut_kh_axis: np.ndarray
    nan_mask: Optional[np.ndarray] = None,
        Needed only when arctangens_curr_bas_mat is np.ndarray
    biases: Optional[np.ndarray] = None,
        Computed biases only on cae of ehight estimation algorithm

    Returns
    -------
    height_2d_mat: Union[np.ndarray, float]
    quality_2d_mat: Union[np.ndarray, float]
    """

    if np.isscalar(arctangens_mat):
        idx_mat = np.nanargmin(np.abs(lut_kh_arctan - arctangens_mat))
    else:
        arctangens_mat[nan_mask] = 0.0
        idx_mat = np.nanargmin(np.abs(lut_kh_arctan[:, None, None] - arctangens_mat[None, :, :]), axis=0)

    height_2d_mat = lut_kh_axis[idx_mat]
    quality_2d_mat = (
        biases[idx_mat] if biases is not None else np.nan * np.zeros(arctangens_mat.shape, dtype=np.float32)
    )

    if not np.isscalar(arctangens_mat):
        height_2d_mat[nan_mask] = np.nan
        quality_2d_mat[nan_mask] = np.nan

    return height_2d_mat, quality_2d_mat


@nb.njit(nogil=True, cache=True, parallel=True)
def find_minimums(
    min_positions: np.ndarray,
    lut_kh_mu_temp: np.ndarray,
    w_ratio_mat: np.ndarray,
    coherence_kz_min_mat: np.ndarray,
    coherence_kz_max_mat: np.ndarray,
):
    """
    min_positions: np.ndarray,
        shape [num_az, num_rg]
    lut_kh_mu_temp: np.ndarray,
        shape [num_kh, num_mu. num_temp_decorr]
    w_ratio_mat: np.ndarray,
        shape [num_az, num_rg]
    coherence_kz_min_mat: np.ndarray,
        shape [num_az, num_rg]
    coherence_kz_max_mat: np.ndarray,
        shape [num_az, num_rg]
    """
    n_az = w_ratio_mat.shape[0]
    n_rg = w_ratio_mat.shape[1]

    n0 = lut_kh_mu_temp.shape[0]
    n1 = lut_kh_mu_temp.shape[1]

    for i_az in nb.prange(n_az):
        for i_rg in range(n_rg):
            if np.isnan(w_ratio_mat[i_az, i_rg]):
                continue

            min_val = 1.0e016
            min_pos = -1

            interp_lut = lut_kh_mu_temp[(w_ratio_mat[i_az, i_rg] * np.arange(n0)).astype(np.int32), :, :]

            for idx in range(n0):
                for idy in range(n1):
                    val = (
                        np.min(np.abs(lut_kh_mu_temp[idx, idy, :] - coherence_kz_max_mat[i_az, i_rg])) ** 2
                        + np.min(np.abs(interp_lut[idx, idy, :] - coherence_kz_min_mat[i_az, i_rg])) ** 2
                    )
                    if val < min_val:
                        min_pos = idx
                        min_val = val

            min_positions[i_az, i_rg] = min_pos

    return


def find_minimums_serial(
    min_positions: np.ndarray,
    lut_kh_mu_temp: np.ndarray,
    w_ratio_mat: np.ndarray,
    coherence_kz_min_mat: np.ndarray,
    coherence_kz_max_mat: np.ndarray,
):
    """
    min_positions: np.ndarray,
        shape [num_az, num_rg]
    lut_kh_mu_temp: np.ndarray,
        shape [num_kh, num_mu. num_temp_decorr]
    w_ratio_mat: np.ndarray,
        shape [num_az, num_rg]
    coherence_kz_min_mat: np.ndarray,
        shape [num_az, num_rg]
    coherence_kz_max_mat: np.ndarray,
        shape [num_az, num_rg]
    """
    n_az = w_ratio_mat.shape[0]
    n_rg = w_ratio_mat.shape[1]

    n0 = lut_kh_mu_temp.shape[0]
    n1 = lut_kh_mu_temp.shape[1]

    for i_az in nb.prange(n_az):
        for i_rg in range(n_rg):
            if np.isnan(w_ratio_mat[i_az, i_rg]):
                continue

            min_val = 1.0e016
            min_pos = -1

            interp_lut = lut_kh_mu_temp[(w_ratio_mat[i_az, i_rg] * np.arange(n0)).astype(np.int32), :, :]

            for idx in range(n0):
                for idy in range(n1):
                    val = (
                        np.min(np.abs(lut_kh_mu_temp[idx, idy, :] - coherence_kz_max_mat[i_az, i_rg])) ** 2
                        + np.min(np.abs(interp_lut[idx, idy, :] - coherence_kz_min_mat[i_az, i_rg])) ** 2
                    )
                    if val < min_val:
                        min_pos = idx
                        min_val = val

            min_positions[i_az, i_rg] = min_pos

    return


def polarimetric_optimization(
    coherence_for_baseline_combinations: np.ndarray,
) -> np.ndarray:
    """
    Polarimetric optimization

    Parameters
    ----------
    coherence_for_baseline_combinations: np.ndarray
    dimensions [P x P x N x num_az x num_rg] or [P x P x N]
        Where
        P, number of polarizations
        N number of baselines combinations:
            the ordering of combinations is
            N = 0 -> Baselines 0-1
            N = 1 -> Baselines 0-2 (present only if num_imms = 3)
            N = 2 -> Baselines 1-2 (present only if num_imms = 3)

    Returns
    -------
    optimized_coherence: np.ndarray
        dimensions [P x num_az x num_rg] or [P,]
    """

    return np.trace(coherence_for_baseline_combinations, axis1=0, axis2=1) / 3  # = optimized_coherence


def range_decorrelation_compensation(
    optimized_coherence_mat: np.ndarray,
    vertical_wavenumber_mat: np.ndarray,
    inc_angle_terrain_corrected_rad: float,
    resolution_rg: float,
) -> np.ndarray:
    """
    Range decorrelation compensation (not for error model computation) (3.8)

    Parameters
    ----------
    optimized_coherence_mat: np.ndarray
        dimensions [P x num_az x num_rg] or [P,]
    vertical_wavenumber_mat: np.ndarray
        vertical wavenumber, for all the baseline combinations
        dimensions [N x num_az x num_rg] or [N,]
    inc_angle_terrain_corrected_rad: float
        Incidence angle, terrain corrected with the slope [rad],
        for the current azimuth and range
    resolution_rg: float
        Slant range resolution [m]

    Returns
    -------
    optimized_coherence_deco: np.ndarray
        range decorrelation compensated coherence
    """

    rg_deco = 1 - np.abs(resolution_rg / (2 * np.pi / vertical_wavenumber_mat)) * np.cos(
        inc_angle_terrain_corrected_rad
    )

    return optimized_coherence_mat / rg_deco
