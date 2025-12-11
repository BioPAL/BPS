# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
TOMO FH commands
----------------
"""

from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import bps.l2a_processor
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.fnf_utils import FnFMask
from bps.common.io import common_types
from bps.l2a_processor.core.aux_pp2_2a import AuxProcessingParametersL2A
from bps.l2a_processor.core.joborder_l2a import L2aJobOrder
from bps.l2a_processor.core.translate_job_order import L2A_OUTPUT_PRODUCT_TFH
from bps.l2a_processor.l2a_common_functionalities import (
    check_lat_lon_orientation,
    fnf_annotation,
    geocoding,
    geocoding_update_dem_coordinates,
    get_dgg_sampling,
    parallel_reinterpolate,
    refine_dgg_search_tiles,
)
from bps.l2a_processor.tomo_fh import BPS_L2A_TFH_PROCESSOR_NAME
from bps.transcoder.io import common_annotation_models_l2
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aLutAdsTOMOFH,
    BIOMASSL2aMainADSInputInformation,
    BIOMASSL2aMainADSProcessingParametersTOMOFH,
    BIOMASSL2aMainADSproduct,
    BIOMASSL2aMainADSRasterImage,
    BIOMASSL2aProductMeasurement,
    BIOMASSL2aProductTOMOFH,
    main_annotation_models_l2a_tfh,
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
from scipy.signal import medfilt2d
from scipy.sparse import csr_matrix

LIGHTSPEED = 299792458


class TOMO_FH:
    """Tomo Forest height L2a Processor"""

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
        bps_logger.info("%s started", BPS_L2A_TFH_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baselines is not None:
            for output_product, output_baseline in zip(self.job_order.output_products, self.job_order.output_baselines):
                if output_product == L2A_OUTPUT_PRODUCT_TFH:
                    self.output_baseline = output_baseline

        self.product_type = L2A_OUTPUT_PRODUCT_TFH

    def _core_processing(self):
        """Execute core TOMO FH L2a processing"""

        scs_axis_sr_s = self.scs_axes_dict["scs_axis_sr_s"]
        scs_axis_az_s = self.scs_axes_dict["scs_axis_az_s"]
        lut_axis_sr_s = self.stack_lut_axes_dict["axis_primary_sr_s"]
        lut_axis_az_s = self.stack_lut_axes_dict["axis_primary_az_s"]

        # vertical_wavenumbers:
        # Prepare for BioPAL code, it also works transposed
        vertical_wavenumbers = {}
        for acq_idx, lut in enumerate(self.stack_lut_list):
            vertical_wavenumbers[acq_idx] = lut["waveNumbers"].astype(np.float64).T
        average_wavenumbers = [np.nanmean(vw) for vw in vertical_wavenumbers]

        if not (
            self.scs_pol_list_calibrated[0][0].shape[1] == vertical_wavenumbers[0].shape[0]
            and self.scs_pol_list_calibrated[0][0].shape[2] == vertical_wavenumbers[0].shape[1]
        ):
            bps_logger.info("    preliminary interpolation to bring vertical wavenumbers onto the L1c data grid")
            for acq_idx in vertical_wavenumbers.keys():
                vertical_wavenumbers[acq_idx] = parallel_reinterpolate(
                    [vertical_wavenumbers[acq_idx].T],  # function works with lists
                    lut_axis_az_s,
                    lut_axis_sr_s,
                    scs_axis_az_s,
                    scs_axis_sr_s,
                )[0].T  # function works with lists

        # Coherence Estimation
        average_azimuth_velocity = np.mean(
            [
                np.linalg.norm(velocity)
                for velocity in self.stack_products_list[self.primary_image_index].general_sar_orbit[0].velocity_vector
            ]
        )

        incidence_angle_rad = np.deg2rad(
            self.stack_lut_list[self.primary_image_index]["incidenceAngle"].astype(np.float32)
        )

        vertical_vector = np.arange(
            self.aux_pp2_2a.tfh.vertical_range.min.value,
            self.aux_pp2_2a.tfh.vertical_range.max.value + self.aux_pp2_2a.tfh.vertical_range.sampling,
            self.aux_pp2_2a.tfh.vertical_range.sampling,
        )

        conf_tomo_fh_est_obj = conf_tomo_fh_est(
            self.aux_pp2_2a.tfh.product_resolution,
            self.aux_pp2_2a.tfh.vertical_range,
            self.aux_pp2_2a.tfh.estimation_valid_values_limits,
            self.aux_pp2_2a.tfh.enable_super_resolution,
            self.aux_pp2_2a.tfh.regularization_noise_factor,
            self.aux_pp2_2a.tfh.power_threshold,
            self.aux_pp2_2a.tfh.median_factor,
        )

        # Prepare data_SLC for BioPAL code, it also works transposed
        pol_names = ["HH", "VH", "VV"]
        num_acquisitions = len(self.scs_pol_list_calibrated[0])

        data_SLC = {}
        for acq_idx in np.arange(num_acquisitions):
            data_SLC[acq_idx] = {}
            for pol_name in pol_names:
                data_SLC[acq_idx][pol_name] = None

        for scs_list_current_pol, pol_name in zip(self.scs_pol_list_calibrated, pol_names):
            for acq_idx, scs_acquisiton in enumerate(scs_list_current_pol):
                data_SLC[acq_idx][pol_name] = scs_acquisiton.T

        cov_est_window_size = self.aux_pp2_2a.tfh.product_resolution

        # Prepare pixel spacing in meters, for BioPAL code
        _, pixel_spacing_slant_rg_m, pixel_spacing_az_m = convert_rasterinfo_seconds_to_meters(
            self.stack_products_list[0].first_sample_sr_time,
            self.stack_products_list[0].rg_time_interval,
            self.stack_products_list[0].az_time_interval,
            average_azimuth_velocity,
        )

        # BIOPAL CODE HERE:
        (
            forest_height,
            _,
            rg_vec_subs,
            az_vec_subs,
            _,
            _,
            _,
        ) = BiomassForestHeightSKPD(
            data_SLC,
            cov_est_window_size,
            pixel_spacing_slant_rg_m,
            pixel_spacing_az_m,
            np.nanmean(incidence_angle_rad),
            self.stack_products_list[0].dataset_info[0].fc_hz,
            self.stack_products_list[0].sampling_constants_list[0].brg_hz,
            vertical_wavenumbers,
            vertical_vector,
            conf_tomo_fh_est_obj,
        )
        # BioPal works transposed
        forest_height = forest_height.T

        # This is from BPS:
        terrain_slope_rad = np.deg2rad(self.stack_lut_list[self.primary_image_index]["terrainSlope"].astype(np.float32))

        # Preliminary interpolation: reference terrain slope
        if not (len(az_vec_subs) == terrain_slope_rad.shape[0] and len(rg_vec_subs) == terrain_slope_rad.shape[1]):
            bps_logger.info(
                "    preliminary interpolation to bring reference terrain slope onto the computed correlation grid"
            )
            terrain_slope_rad = parallel_reinterpolate(
                [terrain_slope_rad],  # function works with lists
                lut_axis_az_s,
                lut_axis_sr_s,
                scs_axis_az_s[az_vec_subs],
                scs_axis_sr_s[rg_vec_subs],
            )[0]  # function works with lists

        # Preliminary interpolation: reference incidence angle
        if not (len(az_vec_subs) == incidence_angle_rad.shape[0] and len(rg_vec_subs) == incidence_angle_rad.shape[1]):
            bps_logger.info(
                "    preliminary interpolation to bring reference incidence angle onto the computed correlation grid"
            )
            incidence_angle_rad = parallel_reinterpolate(
                [incidence_angle_rad],  # function works with lists
                lut_axis_az_s,
                lut_axis_sr_s,
                scs_axis_az_s[az_vec_subs],
                scs_axis_sr_s[rg_vec_subs],
            )[0]  # function works with lists

        inc_angle_terrain_corrected_rad = incidence_angle_rad - terrain_slope_rad

        forest_height = forest_height * (1 - np.tan(terrain_slope_rad) / np.tan(inc_angle_terrain_corrected_rad))

        forest_height_quality = evaluate_estimation_quality_matrix(forest_height.shape)
        ################## BIOPAL CODE HERE (END) ##################

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
            create_dgg_sampling_dict(L2A_OUTPUT_PRODUCT_TFH),
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
            self.scs_axes_dict["scs_axis_az_s"][az_vec_subs],  # sub
            self.scs_axes_dict["scs_axis_sr_s"][rg_vec_subs],  # sub
            self.scs_axes_dict["scs_axis_az_mjd"][az_vec_subs],  # sub
            self.stack_products_list[self.primary_image_index].general_sar_orbit[0],  # SV
            self.aux_pp2_2a.fd.ground_cancellaton.emphasized_forest_height,
            np.deg2rad(dgg_latitude_axis_deg),
            np.deg2rad(dgg_longitude_axis_deg),
        )

        bps_logger.info("Geocoding tomo forest height and quality:")
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
            f"    discarding estimations out of AUX PP2 2A estimation valid values limits [m] [{self.aux_pp2_2a.tfh.estimation_valid_values_limits.min.value}, {self.aux_pp2_2a.tfh.estimation_valid_values_limits.max.value}]:"
        )

        float_no_data_values_mask = processed_data_dict["quality"][0] == FLOAT_NODATA_VALUE
        valid_values_mask = np.logical_or(
            processed_data_dict["fh"][0] < self.aux_pp2_2a.tfh.estimation_valid_values_limits.min.value,
            processed_data_dict["fh"][0] > self.aux_pp2_2a.tfh.estimation_valid_values_limits.max.value,
        )
        num_nans_before_removal = np.sum(float_no_data_values_mask)

        processed_data_dict["fh"][0][valid_values_mask] = FLOAT_NODATA_VALUE
        processed_data_dict["quality"][0][valid_values_mask] = FLOAT_NODATA_VALUE

        num_valid_values_removed_pixels = np.sum(valid_values_mask) - num_nans_before_removal

        size_tomo_fh = processed_data_dict["fh"][0].size
        bps_logger.warning(
            f"        {num_valid_values_removed_pixels / size_tomo_fh * 100:2.3f}% of pixels removed (pixels out of estimation valid values limits)"
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
            average_wavenumbers,
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

        lut_dict["fnf_metadata"] = BIOMASSL2aLutAdsTOMOFH.LutMetadata(
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
        # fill common fileds for TOMO FH and quality
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

        # specific fileds for TOMO FH and FH Quality
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
        average_wavenumbers,
        footprint_mask_for_quicklooks,
    ):
        """Write output TOMO FH L2a product"""

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
            float_pixel_type=main_annotation_models_l2a_tfh.PixelTypeType("32 bit Float")
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

        compression_options_tomo_fh = main_annotation_models_l2a_tfh.CompressionOptionsL2A(
            main_annotation_models_l2a_tfh.CompressionOptionsL2A.Mds(
                main_annotation_models_l2a_tfh.CompressionOptionsL2A.Mds.Tfh(
                    self.aux_pp2_2a.tfh.compression_options.mds.tfh.compression_factor,
                    self.aux_pp2_2a.tfh.compression_options.mds.tfh.max_z_error,
                ),
                main_annotation_models_l2a_tfh.CompressionOptionsL2A.Mds.Quality(
                    self.aux_pp2_2a.tfh.compression_options.mds.quality.compression_factor,
                    self.aux_pp2_2a.tfh.compression_options.mds.quality.max_z_error,
                ),
            ),
            main_annotation_models_l2a_tfh.CompressionOptionsL2A.Ads(
                main_annotation_models_l2a_tfh.CompressionOptionsL2A.Ads.Fnf(
                    self.aux_pp2_2a.tfh.compression_options.ads.fnf.compression_factor
                ),
            ),
            self.aux_pp2_2a.tfh.compression_options.mds_block_size,
            self.aux_pp2_2a.tfh.compression_options.ads_block_size,
        )

        main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersTOMOFH(
            bps.l2a_processor.__version__,
            self.start_time,
            general_configuration,
            compression_options_tomo_fh,
            self.aux_pp2_2a.tfh.enable_super_resolution,
            self.aux_pp2_2a.tfh.product_resolution,
            self.aux_pp2_2a.tfh.regularization_noise_factor,
            self.aux_pp2_2a.tfh.power_threshold,
            self.aux_pp2_2a.tfh.median_factor,
            self.aux_pp2_2a.tfh.estimation_valid_values_limits,
        )

        # lut_ads
        lut_ads = BIOMASSL2aLutAdsTOMOFH(lut_dict["fnf"]["fnf"], lut_dict["fnf_metadata"])

        # Initialize TOMO FH Product
        product_to_write = BIOMASSL2aProductTOMOFH(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            lut_ads,
            product_doi=self.aux_pp2_2a.tfh.l2aTOMOFHProductDOI,
        )

        # Write to file the TOMO FH Product
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

    def run_l2a_tomo_fh_processing(self):
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
            average_wavenumbers,
            footprint_mask_for_quicklooks,
        ) = self._core_processing()

        self._write_to_output(
            processed_data_dict,
            dgg_latitude_axis,
            dgg_longitude_axis,
            lut_dict,
            average_wavenumbers,
            footprint_mask_for_quicklooks,
        )

        processing_stop_time = datetime.now()
        elapsed_time = processing_stop_time - self.processing_start_time
        bps_logger.info(
            "%s total processing time: %.3f s",
            BPS_L2A_TFH_PROCESSOR_NAME,
            elapsed_time.total_seconds(),
        )


########## From BioPal "processing_TOMO.py" ##########
def BiomassForestHeightSKPD(
    data_stack,
    cov_est_window_size,
    pixel_spacing_slant_rg,
    pixel_spacing_az,
    incidence_angle_rad,
    carrier_frequency_hz,
    range_bandwidth_hz,
    kz_stack,
    vertical_vector,
    proc_conf,
):
    power_threshold = proc_conf.power_threshold

    # data_stack is a dictionary of two nested dictionaries composed as:
    # data_stack[ acquisition_name ][ polarization ]

    num_acq = len(data_stack)
    acq_names = list(data_stack.keys())
    first_acq_dict = data_stack[acq_names[0]]
    pol_names = list(first_acq_dict.keys())
    num_pols = len(pol_names)
    Nrg, Naz = first_acq_dict[pol_names[0]].shape
    Nz = np.size(vertical_vector)

    # Covariance estimation
    (
        MPMB_correlation,
        rg_vec_subs,
        az_vec_subs,
        subs_F_r,
        subs_F_a,
    ) = main_correlation_estimation_SR(
        data_stack,
        cov_est_window_size,
        pixel_spacing_slant_rg,
        pixel_spacing_az,
        incidence_angle_rad,
        carrier_frequency_hz,
        range_bandwidth_hz,
    )

    Nrg_subs = rg_vec_subs.size
    Naz_subs = az_vec_subs.size

    # Initialization of the SKPD routine
    class SKPD_kernel_opt_str:
        pass

    SKPD_kernel_opt_str.num_acq = num_acq
    SKPD_kernel_opt_str.num_pols = num_pols
    SKPD_kernel_opt_str.Nsubspaces = 2
    SKPD_kernel_opt_str.Nparam = 2
    SKPD_kernel_opt_str.error = np.zeros((Nrg_subs, Naz_subs, 4, 2))

    # Single polarimetric channel selector
    wpol = (np.kron(np.eye(num_pols), np.ones((1, num_acq)))) > 0

    tomo_cube = np.zeros((Nrg_subs, Naz_subs, Nz))
    Nrg_subs_string = str(Nrg_subs)
    for rg_sub_idx in np.arange(Nrg_subs):
        bps_logger.info("   Heigth step " + str(rg_sub_idx + 1) + " of " + Nrg_subs_string)
        for az_sub_idx in np.arange(Naz_subs):
            # Spectra estimation initialization
            class spectra:
                pass

            spectra.temp = np.zeros((Nz, 4))
            current_MPMB_correlation = MPMB_correlation[:, :, rg_sub_idx, az_sub_idx]

            # SKPD processing
            SKPD_kernel_out_str = SKPD_processing(current_MPMB_correlation, SKPD_kernel_opt_str)

            if np.any(SKPD_kernel_out_str.error):
                # Calibrating with respect to the linked phases of the
                # best polarimetric channel
                Rcoh_thin = Covariance2D2Correlation2D(
                    np.reshape(
                        current_MPMB_correlation[wpol[0, :][:, np.newaxis] * wpol[0, :][np.newaxis, :]],
                        (num_acq, num_acq),
                    )
                )
                Rcoh_fat = Covariance2D2Correlation2D(
                    np.reshape(
                        current_MPMB_correlation[wpol[1, :][:, np.newaxis] * wpol[1, :][np.newaxis, :]],
                        (num_acq, num_acq),
                    )
                )
                Rincoh_thin = Covariance2D2Correlation2D(
                    np.reshape(
                        current_MPMB_correlation[wpol[2, :][:, np.newaxis] * wpol[2, :][np.newaxis, :]],
                        (num_acq, num_acq),
                    )
                )
                Rincoh_fat = np.random.randn(num_acq, num_acq) + 1j * np.random.randn(num_acq, num_acq)

            else:
                # Scattering mechanisms
                Rcoh_thin = Covariance2D2Correlation2D(SKPD_kernel_out_str.Rcoh_thin)
                Rcoh_fat = Covariance2D2Correlation2D(SKPD_kernel_out_str.Rcoh_fat)
                Rincoh_thin = Covariance2D2Correlation2D(SKPD_kernel_out_str.Rincoh_thin)
                Rincoh_fat = Covariance2D2Correlation2D(SKPD_kernel_out_str.Rincoh_fat)

            # Steering matrix
            current_kz = np.zeros((num_acq, 1))
            for b_idx, stack_curr in enumerate(kz_stack.values()):
                current_kz[b_idx] = stack_curr[rg_vec_subs[rg_sub_idx], az_vec_subs[az_sub_idx]]

            A = np.exp(1j * current_kz * vertical_vector) / num_acq

            # Spectra estimation
            for m in np.arange(4):
                currR = (m == 0) * Rcoh_thin + (m == 1) * Rcoh_fat + (m == 2) * Rincoh_thin + (m == 3) * Rincoh_fat
                if proc_conf.enable_super_resolution:
                    # Capon
                    currR = currR + proc_conf.regularization_noise_factor * np.eye(currR.shape[0])
                    spectra.temp[:, m] = 1 / np.diag(np.abs(A.conj().transpose() @ np.linalg.inv(currR) @ A))
                else:
                    spectra.temp[:, m] = np.diag(np.abs(A.conj().transpose() @ currR @ A))

            # Volume mechanism recognized thanks to its higher elevation
            max_index = np.argmax(spectra.temp, axis=0)
            max_m = np.argmax(max_index)
            tomo_cube[rg_sub_idx, az_sub_idx, :] = spectra.temp[:, max_m]

    # Estimating canopy elevation by looking at the decay
    class opt_str:
        pass

    opt_str.z = vertical_vector
    opt_str.thr = power_threshold  # Power decay threshold With respect to the peak value
    out_str = UpperThresholdForestHeight(tomo_cube, opt_str)
    canopy_height = out_str.z
    power_peak = out_str.peak
    canopy_height = medfilt2d(canopy_height.astype("float64"), kernel_size=proc_conf.median_factor)

    return (
        canopy_height,
        power_peak,
        rg_vec_subs,
        az_vec_subs,
        subs_F_r,
        subs_F_a,
        tomo_cube,
    )


def UpperThresholdForestHeight(power_cube, opt_str):
    """It finds the last drop of the power spectrum from opt_str.thr times the
    peak of the spectrum. The corresponding elevation is returned.

    INPUT
         power_cube: [Nr x Nc x Nz] positive real valued cube of
                     backscattered power
         opt_str.
                 thr: threshold for the spectrum drop (between 0 and 1)
                 z: [Nz x 1] elevation axis
    OUTPUT
          out_str.
                  z: [Nr x Nc] canopy elevation map

    DEPENDENCIES
                FindStep3"""
    Nr, Na, Nz = power_cube.shape
    power_peak = np.max(power_cube, axis=2)
    greater_power_mask = power_cube > power_peak.reshape((Nr, Na, 1)) * opt_str.thr

    last_drop_linear_indices = FindStep3(greater_power_mask, "last")

    class out_str:
        pass

    out_str.z = opt_str.z[last_drop_linear_indices]
    out_str.peak = power_peak
    return out_str


########## From BioPal "utility_functions.py" ##########
def evaluate_estimation_quality_matrix(data_in_shape):
    # Placemark for the quality estimation to be defined

    bps_logger.warning(
        "The quality estimation of the product is still to be defined: zeros will be placed in the quality layer "
    )
    quality_matrix = np.zeros(data_in_shape)

    return quality_matrix


########## From BioPal "utility_statistics.py" ##########
def main_correlation_estimation_SR(
    data_stack,
    cov_est_window_size,
    pixel_spacing_slant_rg,
    pixel_spacing_az,
    incidence_angle_rad,
    carrier_frequency_hz,
    range_bandwidth_hz,
):
    (
        MPMB_covariance,
        rg_vec_subs,
        az_vec_subs,
        subs_F_r,
        subs_F_a,
    ) = main_covariance_estimation_SR(
        data_stack,
        cov_est_window_size,
        pixel_spacing_slant_rg,
        pixel_spacing_az,
        incidence_angle_rad,
        carrier_frequency_hz,
        range_bandwidth_hz,
    )

    # Normalizing covariance matrix
    MPMB_correlation, norm_diag = Covariance4D2Correlation4D(MPMB_covariance)

    return MPMB_correlation, rg_vec_subs, az_vec_subs, subs_F_r, subs_F_a


def main_covariance_estimation_SR(
    data_stack,
    cov_est_window_size,
    pixel_spacing_slant_rg,
    pixel_spacing_az,
    incidence_angle_rad,
    carrier_frequency_hz,
    range_bandwidth_hz,
):
    """Covariance estimation:  inputs are in Slant Range radar coordinates
    see also main_covariance_estimation_GR"""

    # compute the pixels spacing in ground range, from slant range:
    pixel_spacing_grd_x = pixel_spacing_slant_rg / np.sin(incidence_angle_rad)
    pixel_spacing_grd_y = pixel_spacing_az

    (
        MPMB_covariance,
        rg_vec_subs,
        az_vec_subs,
        subs_F_r,
        subs_F_a,
    ) = main_covariance_estimation_GR(
        data_stack,
        cov_est_window_size,
        pixel_spacing_grd_x,
        pixel_spacing_grd_y,
        carrier_frequency_hz,
        range_bandwidth_hz,
    )

    return MPMB_covariance, rg_vec_subs, az_vec_subs, subs_F_r, subs_F_a


def main_covariance_estimation_GR(
    data_stack,
    cov_est_window_size,
    pixel_spacing_grd_x,
    pixel_spacing_grd_y,
    carrier_frequency_hz,
    range_bandwidth_hz,
    slant_range_spacing_s=None,
):
    """Covariance estimation:  inputs are in Ground Range coordinates
    see also main_covariance_estimation_SR"""

    cov_est_opt_str = covariance_options_struct_filler(
        data_stack,
        cov_est_window_size,
        pixel_spacing_grd_x,
        pixel_spacing_grd_y,
        carrier_frequency_hz,
        range_bandwidth_hz,
        slant_range_spacing_s,
    )

    (
        MPMB_covariance,
        rg_vec_subs,
        az_vec_subs,
        subs_F_r,
        subs_F_a,
    ) = MPMBCovarianceEstimation(data_stack, cov_est_opt_str)

    return MPMB_covariance, rg_vec_subs, az_vec_subs, subs_F_r, subs_F_a


def MPMBCovarianceEstimation(D_in, opt_str):
    """MPMBCovarianceEstimation

    INPUTS:
        D_in: can be a:
                  dictionary with acquisition keys (Num_Baselines) each containing a dictionary with HH HV VH and VV
                  polarization keys containing arrays with dimensions [Nrg x Naz]
            or (in case of ground cancelled input) :
                  dictionary with HH HV VH and VV
                  polarization keys containing arrays with dimensions [Nrg x Naz]
        opt_str:
            NWmono_r: One-sided range average window [pixel]
            subs_F_r: Range subsampling factor
            NWmono_a: One-sided azimuth average window [pixel]
            subs_F_a: Azimuth subsampling factor
            polarimetric_mask: (optional) [Npol x Npol] logical matrix
                                          stating the polarization combination to
                                          be computed

    OUTPUTS:
        Cov_MPMB: Multi-polarimetric multi-baseline covariance matrix
        rg_out: range subsampled axis
        az_out: azimuth subsampled axis
    """

    first_level_key = next(iter(D_in.keys()))

    if first_level_key == "HH" or first_level_key == "HV" or first_level_key == "VH" or first_level_key == "VV":
        # if D_in contais ground cancelled data, there is not an acquisitions dict:,
        # need to encapsulate the input in an external dummy dictionary
        D = {"single_acq": D_in}
    else:
        D = D_in
    del D_in

    acq_names = list(D.keys())
    N_imm = len(acq_names)
    first_acq_dict = D[acq_names[0]]
    pol_names = list(first_acq_dict.keys())
    num_pols = len(pol_names)
    N_rg, N_az = first_acq_dict[pol_names[0]].shape
    cell_data = True
    double_data = first_acq_dict[pol_names[0]].dtype == "complex128"

    try:
        Mask = opt_str.polarimetric_mask
    except AttributeError:
        Mask = np.ones((num_pols, num_pols))

    # Range filter matrix
    class filtering_matrix_opt_str:
        pass

    filtering_matrix_opt_str.Nin = N_rg
    filtering_matrix_opt_str.NWmono = opt_str.NWmono_r
    filtering_matrix_opt_str.subs_F = opt_str.subs_F_r
    Fr, rg_out, Rnorm = build_filtering_matrix(filtering_matrix_opt_str)
    N_rg_out = rg_out.size
    Fr_normalized = Rnorm @ Fr

    # Azimuth filter matrix
    filtering_matrix_opt_str.Nin = N_az
    filtering_matrix_opt_str.NWmono = opt_str.NWmono_a
    filtering_matrix_opt_str.subs_F = opt_str.subs_F_a
    Fa, az_out, Anorm = build_filtering_matrix(filtering_matrix_opt_str)
    N_az_out = az_out.size
    Fa_normalized_transposed = (Anorm @ Fa).T

    # Init
    Cov_MPMB = np.zeros((num_pols * N_imm, num_pols * N_imm, N_rg_out, N_az_out), dtype=np.complex64)
    nan_mask = np.zeros((N_rg, N_az, N_imm), dtype=bool)

    bps_logger.info("    MPMB covariance estimation...")

    for pol_name_curr in pol_names:
        all_acq_data = np.zeros((N_rg, N_az, N_imm), dtype=complex)
        for acq_idx_curr, acq_curr in enumerate(acq_names):
            all_acq_data[:, :, acq_idx_curr] = D[acq_curr][pol_name_curr]
        nan_mask = nan_mask + np.isnan(all_acq_data)

    for acq_idx_curr, acq_curr in enumerate(acq_names):
        for pol_name_curr in pol_names:
            D[acq_curr][pol_name_curr][nan_mask[:, :, acq_idx_curr]] = 0

    for ch_i, pol_id_i in enumerate(pol_names):
        ind_i = np.arange(N_imm) + ch_i * N_imm
        for ch_j in np.arange(ch_i, num_pols):
            pol_id_j = pol_names[ch_j]
            if Mask[ch_i, ch_j] == 1:
                print("    .")
                ind_j = np.arange(N_imm) + ch_j * N_imm
                II = np.zeros((N_imm, N_imm, N_rg_out, N_az_out), dtype=np.complex64)
                for n_idx, acq_id_n in enumerate(acq_names):
                    if ch_i == ch_j:
                        m_min = n_idx  # ?
                    else:
                        m_min = 0

                    for m_idx in np.arange(m_min, N_imm):
                        acq_id_m = acq_names[m_idx]
                        if cell_data:
                            if double_data:
                                temp = D[acq_id_n][pol_id_i] * np.conjugate(D[acq_id_m][pol_id_j])
                            else:
                                temp = D[acq_id_n][pol_id_i].astype("complex128") * np.conjugate(
                                    D[acq_id_m][pol_id_j]
                                ).astype("complex128")
                        else:
                            if double_data:
                                temp = D[acq_id_n][pol_id_i] * np.conjugate(D[acq_id_m][pol_id_j])
                            else:
                                temp = D[acq_id_n][pol_id_i].astype("complex128") * np.conjugate(
                                    D[acq_id_m][pol_id_j]
                                ).astype("complex128")
                        temp = Fr_normalized @ temp
                        temp = temp @ Fa_normalized_transposed
                        II[n_idx, m_idx, :, :] = temp.astype("complex64")
                Cov_MPMB[ind_i[:, np.newaxis], ind_j[np.newaxis, :], :, :] = II

    # Symmetric part generation
    diag_mask = np.tile(
        (np.eye(num_pols * N_imm).reshape(num_pols * N_imm, num_pols * N_imm, 1, 1)) > 0,
        [1, 1, N_rg_out, N_az_out],
    )
    Cov_MPMB = Cov_MPMB + np.conjugate(np.moveaxis(Cov_MPMB, [1, 0, 2, 3], [0, 1, 2, 3]))
    Cov_MPMB[diag_mask] = Cov_MPMB[diag_mask] / 2

    bps_logger.info("    ...done.\n")

    return Cov_MPMB, rg_out, az_out, opt_str.subs_F_r, opt_str.subs_F_a


def build_filtering_matrix(opt_str):
    """It builds a sparse matrix to carry out one-dimensional moving average.
    It deals with regularly sampled data.

    INPUTS:
       opt_str:
            Nin:    number of rows of the matrix to be filtered
            NWmono: one-sided length of the average window
            subs_F: subsampling factor of the filtered signal

     OUTPUTS:
        F:     sparse filtering matrix
        xout:  axis of the filtered signal
        Rnorm: normalizing matrix
    """

    xin = np.arange(opt_str.Nin)
    Nxin = xin.size
    xout = np.arange(0, opt_str.Nin, opt_str.subs_F)
    Nxout = xout.size

    col = np.kron(xout, np.ones((1, 2 * opt_str.NWmono + 1))) + np.kron(
        np.ones((1, Nxout)), np.arange(-opt_str.NWmono, opt_str.NWmono + 1)
    )
    row = np.kron(np.arange(Nxout), np.ones((1, 2 * opt_str.NWmono + 1)))

    ok_mask = (col >= 0) * (col < Nxin)
    Nok_mask = np.sum(ok_mask)

    F = csr_matrix(
        (np.ones((1, Nok_mask)).flatten(), (row[ok_mask], col[ok_mask])),
        shape=(Nxout, Nxin),
    )

    Rnorm = csr_matrix((1 / np.sum(F.toarray(), axis=1), (np.arange(Nxout), np.arange(Nxout))))

    return F, xout, Rnorm


def covariance_options_struct_filler(
    data_stack,
    cov_est_window_size,
    pixel_spacing_grd_x,
    pixel_spacing_grd_y,
    carrier_frequency_hz,
    range_bandwidth_hz,
    slant_range_spacing_s=None,
):
    first_level_key = next(iter(data_stack.keys()))

    if first_level_key == "HH" or first_level_key == "HV" or first_level_key == "VH" or first_level_key == "VV":
        # if data_stack contais ground cancelled data, there is not an acquisitions dict
        pol_names = list(data_stack.keys())
        Num_of_pols = len(pol_names)

    else:
        acq_names = list(data_stack.keys())
        first_acq_dict = data_stack[acq_names[0]]
        pol_names = list(first_acq_dict.keys())
        Num_of_pols = len(pol_names)

    class cov_est_opt_str:
        pass

    cov_est_opt_str.NW_r = np.int64(cov_est_window_size / pixel_spacing_grd_x / 2) * 2 + 1
    cov_est_opt_str.NW_a = np.int64(cov_est_window_size / pixel_spacing_grd_y / 2) * 2 + 1

    cov_est_opt_str.NWmono_r = np.floor((cov_est_opt_str.NW_r - 1) / 2).astype("int64")
    cov_est_opt_str.NWmono_a = np.floor((cov_est_opt_str.NW_a - 1) / 2).astype("int64")

    cov_est_opt_str.subs_F_r = cov_est_opt_str.NWmono_r
    cov_est_opt_str.subs_F_a = cov_est_opt_str.NWmono_a

    cov_est_opt_str.polarimetric_mask = np.ones((Num_of_pols, Num_of_pols))

    cov_est_opt_str.wavelenght = LIGHTSPEED / carrier_frequency_hz
    cov_est_opt_str.B = range_bandwidth_hz
    cov_est_opt_str.f0 = carrier_frequency_hz
    cov_est_opt_str.dt = slant_range_spacing_s

    return cov_est_opt_str


def Covariance4D2Correlation4D(Cov_MPMB):
    """It normalizes each element of the 4-D Multi-Polarimetric Multi-Baseline
    with respect to the corresponding diagonal terms.

    INPUT
         Cov_MPMB: [Nimm*Npol x Nimm*Npol x Nr x Na] Covariance matrices

    OUTPUT
          Corr_MPMB: [Nimm*Npol x Nimm*Npol x Nr x Na] normalized covariance
                     matrices
          varargout{1}: [Nr x Na x Nimm*Npol] diagonal entries of Cov_MPMB"""
    temp, N, Nr, Na = Cov_MPMB.shape
    diag_mask = np.tile((np.eye(N).reshape(N, N, 1, 1)) > 0, [1, 1, Nr, Na])
    norm_util = Cov_MPMB[diag_mask]
    norm_diag = np.moveaxis(norm_util.reshape((N, Nr, Na)), [1, 2, 0], [0, 1, 2])
    norm_util = 1 / np.sqrt(norm_util + np.spacing(1))

    norm_util = norm_util.reshape((N, 1, Nr, Na)) * norm_util.reshape((1, N, Nr, Na))
    Corr_MPMB = norm_util * Cov_MPMB

    return Corr_MPMB, norm_diag


def Covariance2D2Correlation2D(Cov2D):
    """It normalizes each element of the covariance matrix with respect to the
    corresponding elements on the main diagonal.

    INPUT
         Cov2D: [N x N] covariance matrix

    OUTPUT
          Corr2D: [N x N] normalized covariance matrix"""
    temp = np.diag(1 / np.sqrt(np.diag(Cov2D)))
    Corr2D = temp @ Cov2D @ temp

    return Corr2D


def SKPD_processing(Cov_MPMB, opt_str):
    """SKPD_processing

    Kernel of the Sum of Kronecker Product Decomposition.

    INPUT
         Cov_MPMB: [N*Npol x N*Npol] covariance matrix
          opt_str.
                           N: number of images
                        Npol: number of polarizations
                  Nsubspaces: number of desired subspaces
                      Nparam: number of parameters sampling the "a" and "b"
                              range of physical admissibility

    OUTPUT
          out_str.
                  error: [1 x 4] logical vector.
                         1. Cov_MPMB has nan of inf values
                         2. Singular matrices have nan or inf values
                         3. Singular matrices are not full rank
                         4. Could not determine "a" or "b" ranges
                  lambda_SVD: singular values of the SKPD
                  Ro: tomographic part of the decomposition (normalized wrt
                      Ro{k}(1, 1))
                  Co: polarimetric part of the decomposition
                  ax: vector gathering the admissible values for parameter a
                  bx: vector gathering the admissible values for parameter b
                  Ra: corresponding tomographic covariance matrices (at the
                      edges of the possible range of values of "a")
                  Rb: corresponding tomographic covariance matrices (at the
                      edges of the possible range of values of "b")
                  Ca: corresponding polarimetric covariance matrices (at the
                      edges of the possible range of values of "a")
                  Cb: corresponding polarimetric covariance matrices (at the
                      edges of the possible range of values of "b")
                  full_rank_Ra: true if Ra is full rank
                  full_rank_Rb: true if Rb is full rank
                  full_rank_Ca: true if Ca is full rank
                  full_rank_Cb: true if Cb is full rank
                  Ra_coherence: tomographic coherence of Ra
                  Rb_coherence: tomographic coherence of Rb
                  Ca_coherence: tomographic coherence of Ca
                  Cb_coherence: tomographic coherence of Cb
                  Ra_more_coherent_than_Rb: [1 x 1] logical
                  ax_R_max_coh: [1 x 2] logical. True if the parameter edge
                                provides greater coherence
                  bx_R_max_coh: [1 x 2] logical. True if the parameter edge
                                provides greater coherence
                  R_most_coeherent: most coherent R
                  Rcoh_thin: Most coherent scattering mechanism, most
                             coherent among the range of admissibility
                  Rcoh_fat: Most coherent scattering mechanism, least
                             coherent among the range of admissibility
                  Rincoh_thin: Least coherent scattering mechanism, most
                             coherent among the range of admissibility
                  Rincoh_fat: Least coherent scattering mechanism, least
                             coherent among the range of admissibility

    DEPENDENCIES
                LowRankKronApproximation
                SemiDefinitePositivenessJointRange
                                JointDiagonalization"""

    class out_str:
        pass

    out_str.error = np.zeros((4, 1))

    if np.any(np.isnan(Cov_MPMB) | np.isinf(Cov_MPMB)):
        out_str.error[0] = 1
        return out_str

    class LRKA_str:
        pass

    LRKA_str.A = Cov_MPMB
    LRKA_str.rank = opt_str.Nsubspaces
    LRKA_str.Nr1 = opt_str.num_pols
    LRKA_str.Nc1 = opt_str.num_pols
    LRKA_str.Nr2 = opt_str.num_acq
    LRKA_str.Nc2 = opt_str.num_acq
    LRKA_out_str = LowRankKronApproximation(LRKA_str)
    Co = LRKA_out_str.B
    Ro = LRKA_out_str.C
    lambda_SVD = LRKA_out_str.lambda_SVD

    # Demanding power to polarimetric signature
    for k in np.arange(opt_str.Nsubspaces):
        temp = Ro[k][0, 0]
        Ro[k] = Ro[k] / temp
        Co[k] = Co[k] * temp * lambda_SVD[k]

    out_str.lambda_SVD = lambda_SVD[np.arange(opt_str.Nsubspaces)]
    out_str.Ro = Ro
    out_str.Co = Co

    if np.any(np.isnan(Ro)) or np.any(np.isnan(Co)) | np.any(np.isinf(Ro)) or np.any(np.isinf(Co)):
        out_str.error[1] = 1
        return out_str

    if (
        not (np.linalg.matrix_rank(Ro[0]) == opt_str.num_acq)
        or not (np.linalg.matrix_rank(Ro[1]) == opt_str.num_acq)
        or not (np.linalg.matrix_rank(Co[0]) == opt_str.num_pols)
        or not (np.linalg.matrix_rank(Co[1]) == opt_str.num_pols)
    ):
        out_str.error[2] = 1
        return out_str

    class SDP_opt_str:
        pass

    SDP_opt_str.Ro = Ro
    SDP_opt_str.Co = Co
    SDP_opt_str.N = opt_str.Nparam
    SDP_out_str = SemiDefinitePositivenessJointRange(SDP_opt_str)
    if SDP_out_str.error:
        out_str.error[3] = 1
        return out_str
    ax = SDP_out_str.a
    bx = SDP_out_str.b

    out_str.ax = ax
    out_str.bx = bx

    out_str.Ra = list([ax[0] * Ro[0] + (1 - ax[0]) * Ro[1], ax[-1] * Ro[0] + (1 - ax[-1]) * Ro[1]])
    out_str.Rb = list([bx[0] * Ro[0] + (1 - bx[0]) * Ro[1], bx[-1] * Ro[0] + (1 - bx[-1]) * Ro[1]])

    out_str.Ca = list([-(1 - ax[0]) * Co[0] + ax[0] * Co[1], -(1 - ax[-1]) * Co[0] + ax[-1] * Co[1]])
    out_str.Cb = list([(1 - bx[0]) * Co[0] - bx[0] * Co[1], (1 - bx[-1]) * Co[0] - bx[-1] * Co[1]])

    out_str.full_rank_Ra = [
        np.linalg.cond(out_str.Ra[0]) < np.linalg.cond(out_str.Ra[1]),
        False,
    ]
    out_str.full_rank_Ra[1] = not (out_str.full_rank_Ra[0])
    out_str.full_rank_Rb = [
        np.linalg.cond(out_str.Rb[0]) < np.linalg.cond(out_str.Rb[1]),
        False,
    ]
    out_str.full_rank_Rb[1] = not (out_str.full_rank_Rb[0])

    out_str.full_rank_Ca = [
        np.linalg.cond(out_str.Ca[0]) < np.linalg.cond(out_str.Ca[1]),
        False,
    ]
    out_str.full_rank_Ca[1] = not (out_str.full_rank_Ca[0])
    out_str.full_rank_Cb = [
        np.linalg.cond(out_str.Cb[0]) < np.linalg.cond(out_str.Cb[1]),
        False,
    ]
    out_str.full_rank_Cb[1] = not (out_str.full_rank_Cb[0])

    triu_mask_R = np.triu(np.ones((opt_str.num_acq, opt_str.num_acq)), 1) > 0
    Ntriu_mask_R = (opt_str.num_acq**2 - opt_str.num_acq) / 2
    triu_mask_C = np.triu(np.ones((opt_str.num_pols, opt_str.num_pols)), 1) > 0
    Ntriu_mask_C = (opt_str.num_pols**2 - opt_str.num_pols) / 2
    out_str.Ra_coherence = np.zeros((2, 1))
    out_str.Rb_coherence = np.zeros((2, 1))
    out_str.Ca_coherence = np.zeros((2, 1))
    out_str.Cb_coherence = np.zeros((2, 1))
    for k in np.arange(2):
        OO = np.diag(1 / np.sqrt(np.diag(out_str.Ra[k])))
        temp = OO @ out_str.Ra[k] @ OO
        out_str.Ra_coherence[k] = np.sum(np.abs(temp[triu_mask_R])) / Ntriu_mask_R
        OO = np.diag(1 / np.sqrt(np.diag(out_str.Rb[k])))
        temp = OO @ out_str.Rb[k] @ OO
        out_str.Rb_coherence[k] = np.sum(np.abs(temp[triu_mask_R])) / Ntriu_mask_R
        OO = np.diag(1 / np.sqrt(np.diag(out_str.Ca[k])))
        temp = OO @ out_str.Ca[k] @ OO
        out_str.Ca_coherence[k] = np.sum(np.abs(temp[triu_mask_C])) / Ntriu_mask_C
        OO = np.diag(1 / np.sqrt(np.diag(out_str.Cb[k])))
        temp = OO @ out_str.Cb[k] @ OO
        out_str.Cb_coherence[k] = np.sum(np.abs(temp[triu_mask_C])) / Ntriu_mask_C

    out_str.Ra_more_coherent_than_Rb = np.max(out_str.Ra_coherence) > np.max(out_str.Rb_coherence)
    out_str.ax_R_max_coh = [out_str.Ra_coherence[0] > out_str.Ra_coherence[1], False]
    out_str.ax_R_max_coh[1] = not (out_str.ax_R_max_coh[0])
    out_str.bx_R_max_coh = [out_str.Rb_coherence[0] > out_str.Rb_coherence[1], False]
    out_str.bx_R_max_coh[1] = not (out_str.bx_R_max_coh[0])

    out_str.R_most_coherent = (out_str.Ra_more_coherent_than_Rb) * (
        out_str.Ra[0] * out_str.ax_R_max_coh[0] + out_str.Ra[1] * out_str.ax_R_max_coh[1]
    ) + (1 - out_str.Ra_more_coherent_than_Rb) * (
        out_str.Rb[0] * out_str.bx_R_max_coh[0] + out_str.Rb[1] * out_str.bx_R_max_coh[1]
    )

    out_str.Rcoh_thin = (out_str.Ra_more_coherent_than_Rb) * (
        out_str.Ra[0] * out_str.ax_R_max_coh[0] + out_str.Ra[1] * out_str.ax_R_max_coh[1]
    ) + (1 - out_str.Ra_more_coherent_than_Rb) * (
        out_str.Rb[0] * out_str.bx_R_max_coh[0] + out_str.Rb[1] * out_str.bx_R_max_coh[1]
    )
    out_str.Rcoh_fat = (out_str.Ra_more_coherent_than_Rb) * (
        out_str.Ra[0] * out_str.ax_R_max_coh[1] + out_str.Ra[1] * out_str.ax_R_max_coh[0]
    ) + (1 - out_str.Ra_more_coherent_than_Rb) * (
        out_str.Rb[0] * out_str.bx_R_max_coh[1] + out_str.Rb[1] * out_str.bx_R_max_coh[0]
    )
    out_str.Rincoh_thin = (1 - out_str.Ra_more_coherent_than_Rb) * (
        out_str.Ra[0] * out_str.ax_R_max_coh[0] + out_str.Ra[1] * out_str.ax_R_max_coh[1]
    ) + (out_str.Ra_more_coherent_than_Rb) * (
        out_str.Rb[0] * out_str.bx_R_max_coh[0] + out_str.Rb[1] * out_str.bx_R_max_coh[1]
    )
    out_str.Rincoh_fat = (1 - out_str.Ra_more_coherent_than_Rb) * (
        out_str.Ra[0] * out_str.ax_R_max_coh[1] + out_str.Ra[1] * out_str.ax_R_max_coh[0]
    ) + (out_str.Ra_more_coherent_than_Rb) * (
        out_str.Rb[0] * out_str.bx_R_max_coh[1] + out_str.Rb[1] * out_str.bx_R_max_coh[0]
    )

    return out_str


def SemiDefinitePositivenessJointRange(opt_str):
    """Given the relationships:
                  Rg = a*Ro{1} + (1 - a)*Ro{2}
                  Rv = b*Ro{2} + (1 - b)*Ro{2}
                  Cg = (1 - b)*C{1} - b*C{2}
                  Cv = -(1 - a)*C{1} + a*C{2}
    this routine returns the range of values for parameters "a" and "b" such
    that matrices Rg, Rv, Cg, Cv are semi definite positive.

    INPUT
         opt_str.
                 Ro: [2 x 1] cell of [N x N] Hermitian matrices
                 Co: [2 x 1] cell of [Npol x Npol] Hermitian matrices
                  N: number of points sampling the admissible range of values

    OUTPUT
          out_str.
                  a: [1 x opt_str.N] vector
                  b: [1 x opt_str.N] vector
                  error: true if an error has occurred

    DEPENDENCIES
                JointDiagonalization"""

    class out_str:
        pass

    out_str.error = False

    # Semi positive definiteness of the interferometric covariance matrices
    U = JointDiagonalization(opt_str.Ro[0:2])
    d1 = np.real(np.diag(U.conj().transpose() @ opt_str.Ro[0] @ U))
    d2 = np.real(np.diag(U.conj().transpose() @ opt_str.Ro[1] @ U))

    dd = d1 - d2
    indm = dd < 0
    indp = dd > 0
    temp = -d2 / dd

    try:
        ab_lo = np.max(temp[indp])
    except ValueError:
        ab_lo = -np.inf
    try:
        ab_up = np.min(temp[indm])
    except ValueError:
        ab_up = np.inf
    # Semi positive definiteness of the polarimetric covariance matrices
    U = JointDiagonalization(opt_str.Co[0:2])
    d1 = np.real(np.diag(U.conj().transpose() @ opt_str.Co[0] @ U))
    d2 = np.real(np.diag(U.conj().transpose() @ opt_str.Co[1] @ U))

    dd = d1 + d2
    indm = dd < 0
    indp = dd > 0
    temp = d1 / dd

    try:
        a_lo = np.max(temp[indp])
    except ValueError:
        a_lo = -np.inf
    try:
        a_up = np.min(temp[indm])
    except ValueError:
        a_up = np.inf

    dd = -d1 - d2
    indm = dd < 0
    indp = dd > 0
    temp = -d1 / dd

    try:
        b_lo = np.max(temp[indp])
    except ValueError:
        b_lo = -np.inf
    try:
        b_up = np.min(temp[indm])
    except ValueError:
        b_up = np.inf

    if a_lo < b_up:
        out_str.error = True
    # Semi positive definiteness of both covariance matrices
    a_lo = np.max((a_lo, ab_lo))
    a_up = np.min((a_up, ab_up))
    b_lo = np.max((b_lo, ab_lo))
    b_up = np.min((b_up, ab_up))

    Da = a_up - a_lo
    Db = b_up - b_lo
    if Da < 0 or Db < 0:
        out_str.error = True

    out_str.a = np.linspace(a_lo, a_up, opt_str.N)
    out_str.b = np.linspace(b_lo, b_up, opt_str.N)

    return out_str


def JointDiagonalization(C):
    """It finds the matrix U that diagonalizes C{1} and C{2} at the same time
    (it means that U'*C{1}*U and U'*C{2}*U are diagonal)

    INPUT
         C: [2 x 1] cell of [N x N] Hermitian matrices

    OUTPUT
          U: diagonalizing matrix"""
    A, V = np.linalg.eig(C[1] @ np.linalg.inv(C[0]))
    U = np.linalg.inv(V.conj().transpose())

    return U


def LowRankKronApproximation(opt_str):
    """It returns the sum of Kronecker product decomposition of the matrix A
    just as the singular value decomposition. Matrix A is decomposed as:
    for k = 1:rank
        A = A + kron(B{k}, C{k});
    end

    INPUT
         opt_str.
                 A: [Nr x Nc] matrix to be decomposed and approximated
                 rank: rank of the approximation
                 Nr1: number of rows of the matrix B
                 Nc1: number of columns of the matrix B
                 Nr2: number of rows of the matrix C
                 Nr2: number of columns of the matrix C

    OUTPUT
          out_str.
                  lambda: singular values of the decomposition
                  B: [rank x 1] cell of [Nr1 x Nc1] matrices
                  C: [rank x 1] cell of [Nr2 x Nc2] matrices"""
    RA = np.zeros((opt_str.Nr1 * opt_str.Nc1, opt_str.Nr2 * opt_str.Nc2), dtype=np.complex128)
    for p in np.arange(opt_str.Nc1):
        Ap = np.zeros((opt_str.Nr1, opt_str.Nr2 * opt_str.Nc2), dtype=np.complex128)
        ind_p = np.arange(p * opt_str.Nc2, (p + 1) * opt_str.Nc2)
        for q in np.arange(opt_str.Nr1):
            ind_q = np.arange(q * opt_str.Nr2, (q + 1) * opt_str.Nr2)
            Aqp = opt_str.A[ind_q[:, np.newaxis], ind_p[np.newaxis, :]]
            Ap[q, :] = Aqp.transpose().flatten()
        ind = np.arange(p * opt_str.Nr1, (p + 1) * opt_str.Nr1)
        RA[ind, :] = Ap

    U, S, VH = np.linalg.svd(RA)

    class out_str:
        pass

    out_str.lambda_SVD = S

    out_str.B = list()
    out_str.C = list()
    for k in np.arange(opt_str.rank):
        out_str.B.append(np.reshape(U[:, k], (opt_str.Nr1, opt_str.Nc1), order="F"))
        out_str.C.append(np.reshape(np.transpose(VH)[:, k], (opt_str.Nr2, opt_str.Nc2), order="F"))  # warning !

    return out_str


conf_tomo_fh_est = namedtuple(
    "conf_tomo_fh_est",
    "product_resolution \
     vertical_range \
     estimation_valid_values_limits \
     enable_super_resolution \
     regularization_noise_factor \
     power_threshold \
     median_factor",
)


def convert_rasterinfo_seconds_to_meters(samples_start_s, samples_step_s, lines_step_s, sensor_velocity):
    lines_step_m = lines_step_s * sensor_velocity
    samples_start_m = samples_start_s * LIGHTSPEED / 2
    samples_step_m = samples_step_s * LIGHTSPEED / 2

    return samples_start_m, samples_step_m, lines_step_m


def FindStep3(II, varargin):
    """This routin finds the first or the last step in a logical 3D along the
    third direction. The first step is a rising one whereas the last is a
    drop.

    INPUT
         II: [Nr x Nc x N] logical array to be explored
         varargin{1}: (optional. Default: 'first') either 'first' or 'last'

    OUTPUT
          linear_indices: [Nr x Nc] matrix of linear indices where the step
                          has been detected"""
    Nr, Na, _ = II.shape
    II = II > 0

    if varargin == "first":
        rise_mask = 1 - (np.roll(II, (0, 0, 1), (0, 1, 2))) & II
        rise_mask[:, :, 0] = False
        rise_mask[:, :, -1] = False

        rise_mask_cumsum = np.cumsum(rise_mask, axis=2)

        linear_indices = np.sum(
            ((rise_mask_cumsum == 1) & rise_mask) * np.arange(II.shape[2]).reshape(1, 1, II.shape[2]),
            axis=2,
        )

    elif varargin == "last":
        drop_mask = 1 - (np.roll(II, (0, 0, -1), (0, 1, 2))) & II
        drop_mask[:, :, 0] = False
        drop_mask[:, :, -1] = False

        drop_mask_cumsum = np.cumsum(drop_mask, axis=2)

        linear_indices = np.sum(
            ((drop_mask_cumsum == drop_mask_cumsum[:, :, -1].reshape((Nr, Na, 1))) & drop_mask)
            * np.arange(II.shape[2]).reshape(1, 1, II.shape[2]),
            axis=2,
        )

    else:
        print("Unrecognized input, aborting.\n")
        linear_indices = []

    return linear_indices
