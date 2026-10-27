# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
GN commands
-----------
"""

from datetime import datetime
from pathlib import Path

import bps.l2a_processor
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.fnf_utils import FnFMask
from bps.common.io import common
from bps.l2a_processor.core.aux_pp2_2a import AuxProcessingParametersL2A
from bps.l2a_processor.core.joborder_l2a import L2aJobOrder
from bps.l2a_processor.core.translate_aux_pp2_2a import OperationalModeType
from bps.l2a_processor.core.translate_job_order import L2A_OUTPUT_PRODUCT_GN
from bps.l2a_processor.gn import BPS_L2A_GN_PROCESSOR_NAME
from bps.l2a_processor.ground_cancellation import ground_cancellation
from bps.l2a_processor.l2a_common_functionalities import (
    averaging_windows_sizes,
    build_filtering_sparse_matrices,
    check_lat_lon_orientation,
    fnf_annotation,
    geocoding,
    geocoding_update_dem_coordinates,
    get_dgg_sampling,
    parallel_reinterpolate,
    refine_dgg_search_tiles,
)
from bps.transcoder.io import common_annotation_l2l3, main_annotation_l2a_gn
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aLutAdsGN,
    BIOMASSL2aProductGN,
    BIOMASSL2aProductMeasurement,
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


class GN:
    """Ground Cancellation: AGB L2a Processor"""

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
                scs_axes_dict["scs_axes_dict["scs_axis_sr_s"]"]: np.ndarray
                    slant range temporal axis, in seconds
                scs_axes_dict["scs_axes_dict["scs_axis_az_s"]"]: np.ndarray
                    azimuth temporal axis, in seconds
                scs_axes_dict["scs_axes_dict["scs_axis_az_mjd"]"]: PreciseDateTime
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
        """Initialize the FD L2a processing"""

        self.processing_start_time = datetime.now()
        self.start_time = PreciseDateTime.now()
        bps_logger.info("%s started", BPS_L2A_GN_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baselines is not None:
            for output_product, output_baseline in zip(self.job_order.output_products, self.job_order.output_baselines):
                if output_product == L2A_OUTPUT_PRODUCT_GN:
                    self.output_baseline = output_baseline

        self.product_type = L2A_OUTPUT_PRODUCT_GN

    def _core_processing(self):
        """Execute core GN (AGB) L2a processing"""

        # Ground Cancellation
        vertical_wavenumbers = [lut["waveNumbers"].astype(np.float64) for lut in self.stack_lut_list]

        average_wavenumbers = [np.nanmean(vw) for vw in vertical_wavenumbers]

        ground_cancelled, idx_selected_reference = ground_cancellation(
            self.scs_pol_list_calibrated,
            self.scs_axes_dict["scs_axis_sr_s"],
            self.scs_axes_dict["scs_axis_az_s"],
            self.primary_image_index,
            vertical_wavenumbers,
            self.stack_lut_axes_dict["axis_primary_sr_s"],
            self.stack_lut_axes_dict["axis_primary_az_s"],
            self.aux_pp2_2a.agb.ground_cancellaton,
            self.acquisition_paths_selected_not_sorted,
            self.aux_pp2_2a.general.polarisation_combination_method,
        )
        acquisition_id_reference_image = (
            self.acquisition_paths_selected_not_sorted[idx_selected_reference].name
            if idx_selected_reference is not None
            else None
        )

        average_azimuth_velocity = np.mean(
            [
                np.linalg.norm(velocity)
                for velocity in self.stack_products_list[self.primary_image_index].general_sar_orbit[0].velocity_vector
            ]
        )

        (
            ground_canc_norm_multilook,
            local_inc_angle_rad_multilook,
            axis_az_subsampling_indexes,
            axis_rg_subsampling_indexes,
        ) = sigma_naught_normalisation(
            ground_cancelled,
            np.deg2rad(self.stack_lut_list[self.primary_image_index]["incidenceAngle"].astype(np.float32)),
            np.deg2rad(self.stack_lut_list[self.primary_image_index]["rangeTerrainSlope"].astype(np.float32)),
            self.stack_lut_axes_dict["axis_primary_az_s"],
            self.stack_lut_axes_dict["axis_primary_sr_s"],
            self.scs_axes_dict["scs_axis_az_s"],
            self.scs_axes_dict["scs_axis_sr_s"],
            self.stack_products_list[0].sampling_constants_list[0].baz_hz,
            self.stack_products_list[0].sampling_constants_list[0].brg_hz,
            average_azimuth_velocity,
            self.aux_pp2_2a.agb.product_resolution,
            self.aux_pp2_2a.agb.upsampling_factor,
            self.aux_pp2_2a.agb.ground_cancellaton.radiometric_calibration_flag,
        )

        bps_logger.info(
            "Interpolation of incidence angle, elevation angle and range, azimuth DEM Slopes over LUTs axis"
        )
        # height, latitude, longitude are already interpolated
        luts_interpolated_list = parallel_reinterpolate(
            [
                self.stack_lut_list[self.primary_image_index]["incidenceAngle"].astype(np.float32),
                self.stack_lut_list[self.primary_image_index]["elevationAngle"].astype(np.float32),
                self.stack_lut_list[self.primary_image_index]["rangeTerrainSlope"].astype(np.float32),
                self.stack_lut_list[self.primary_image_index]["azimuthTerrainSlope"].astype(np.float32),
            ],
            self.stack_lut_axes_dict["axis_primary_az_s"],
            self.stack_lut_axes_dict["axis_primary_sr_s"],
            self.scs_axes_dict["scs_axis_az_s"][axis_az_subsampling_indexes],  # sub,
            self.scs_axes_dict["scs_axis_sr_s"][axis_rg_subsampling_indexes],  # sub,
        )

        # Get the DGG sampling parameters, needed for the geocoding step:
        # Check the orientation of FNF lat lon axis and keep this convention in the L2a product.
        invert_latitude, invert_longitude = check_lat_lon_orientation(self.fnf.lat_axis, self.fnf.lon_axis)
        (
            dgg_latitude_axis_deg,
            dgg_longitude_axis_deg,
        ) = get_dgg_sampling(
            self.latlon_coverage,
            create_dgg_sampling_dict(L2A_OUTPUT_PRODUCT_GN),
            invert_latitude,
            invert_longitude,
        )

        # Geocoding step 1/2: update DEM coordinates
        (
            delaunay,
            dgg_latitude_mesh_rad,
            dgg_longitude_mesh_rad,
            dem_valid_values_mask,
            dem_for_lut,
            latitude_deg_for_lut,
            longitude_deg_for_lut,
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
            self.aux_pp2_2a.agb.ground_cancellaton.emphasized_forest_height,
            np.deg2rad(dgg_latitude_axis_deg),
            np.deg2rad(dgg_longitude_axis_deg),
        )

        bps_logger.info("   geocoding GN data and incidence angle")
        # Append incidence angle to the list, in order to compute it faster in a single Delaunay call
        ground_canc_norm_multilook = list(ground_canc_norm_multilook)
        ground_canc_norm_multilook.append(local_inc_angle_rad_multilook)
        geocoded_list = geocoding(
            ground_canc_norm_multilook,
            delaunay,
            dgg_latitude_mesh_rad,
            dgg_longitude_mesh_rad,
            dem_valid_values_mask,
            fill_value=FLOAT_NODATA_VALUE,
        )

        bps_logger.info(
            "Geocoding of DEM, latitude, longitude, incidence and elevation angles, range and azimuth DEM Slopes for LUTs annotation"
        )
        luts_geocoded_list = geocoding(
            [
                dem_for_lut,  # height
                latitude_deg_for_lut,  # latitude
                longitude_deg_for_lut,  # longitude
                luts_interpolated_list[0],  # incidenceAngle
                luts_interpolated_list[1],  # elevationAngle
                luts_interpolated_list[2],  # rangeTerrainSlope
                luts_interpolated_list[3],  # azimuthTerrainSlope
            ],
            delaunay,
            dgg_latitude_mesh_rad,
            dgg_longitude_mesh_rad,
            dem_valid_values_mask,
            fill_value=FLOAT_NODATA_VALUE,
        )
        del luts_interpolated_list

        lut_dict = {}
        for lut_name, lut_geocoded in zip(
            [
                "height",
                "latitude",
                "longitude",
                "incidenceAngle",
                "elevationAngle",
                "rangeTerrainSlope",
                "azimuthTerrainSlope",
            ],
            luts_geocoded_list,
        ):
            lut_dict[lut_name] = lut_geocoded.astype(np.float32)
            lut_dict[lut_name][~np.isfinite(lut_dict[lut_name])] = FLOAT_NODATA_VALUE
        del luts_geocoded_list

        # Filling the output dictionary for GN
        # Casting back to float32 all the float data before saving
        # Geocoded list contains 3 GN polarizations plus the incidence angle at idx == 3
        gn_multilooked_geocoded = {"gn": []}
        for idx, geocoded_list_entry in enumerate(geocoded_list):
            if idx == 3:
                # index of the incidence angle
                continue
            geocoded_list_entry[np.isnan(geocoded_list_entry)] = FLOAT_NODATA_VALUE
            gn_multilooked_geocoded["gn"].append(geocoded_list_entry.astype(np.float32))

        # Getting the incidence angle from the Geocoded list (last element, idx == 3), converting it to degrees
        local_inc_angle_deg_multilook_geocoded = np.rad2deg(
            geocoded_list[3].astype(type(local_inc_angle_rad_multilook[0, 0]))
        )
        # replacing no data values with FLOAT_NODATA_VALUE after conversion to degrees
        no_data_values_mask = np.logical_or(geocoded_list[3] == FLOAT_NODATA_VALUE, np.isnan(geocoded_list[3]))
        local_inc_angle_deg_multilook_geocoded[no_data_values_mask] = FLOAT_NODATA_VALUE
        del geocoded_list  # free memory

        # FNF annotation step
        lut_dict["fnf"] = fnf_annotation(
            self.fnf,
            dgg_latitude_axis_deg.astype(np.float32),
            dgg_longitude_axis_deg.astype(np.float32),
            self.aux_pp2_2a.general.forest_mask_interpolation_threshold,
        )
        lut_dict["local_incidence_angle"] = local_inc_angle_deg_multilook_geocoded.astype(np.float32)

        not_footprint_mask = gn_multilooked_geocoded["gn"][0] == FLOAT_NODATA_VALUE
        lut_dict["fnf"]["fnf"][not_footprint_mask] = INT_NODATA_VALUE

        lut_dict["local_incidence_angle"][np.isnan(lut_dict["local_incidence_angle"])] = FLOAT_NODATA_VALUE

        # Footprint mask for quick looks transparency
        gn_for_footprint = np.copy(gn_multilooked_geocoded["gn"][0])
        gn_for_footprint[gn_for_footprint == FLOAT_NODATA_VALUE] = np.nan
        if AVERAGING_FACTOR_QUICKLOOKS > 1:
            gn_for_footprint = gn_for_footprint[::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS]
        footprint_mask_for_quicklooks = np.logical_not(np.isnan(gn_for_footprint))

        return (
            gn_multilooked_geocoded,
            dgg_latitude_axis_deg.astype(np.float32),
            dgg_longitude_axis_deg.astype(np.float32),
            lut_dict,
            [wv.astype(np.float32) for wv in average_wavenumbers],
            idx_selected_reference,
            acquisition_id_reference_image,
            footprint_mask_for_quicklooks,
        )

    def _fill_product_for_writing(self, lut_dict):
        # fixed values (i.e. "mission=BIOMASS") are directly set in _write_to_output function
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

        lut_dict["fnf_metadata"] = BIOMASSL2aLutAdsGN.LutMetadata(
            first_sample=1,
            first_line=1,
            samples_interval=1,
            lines_interval=1,
            pixelType="8 bit Unsigned Integer",
            no_data_value=INT_NODATA_VALUE,
            projection=common.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG.value,
            coordinateReferenceSystem=COORDINATE_REFERENCE_SYSTEM,
            geodeticReferenceFrame=common.GeodeticReferenceFrameType.WGS84.value,
        )

        lut_dict["geometric_metadata"] = BIOMASSL2aLutAdsGN.LutMetadata(
            first_sample=1,
            first_line=1,
            samples_interval=1,
            lines_interval=1,
            pixelType="32 bit Float",
            no_data_value=FLOAT_NODATA_VALUE,
            projection=common.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG.value,
            coordinateReferenceSystem=COORDINATE_REFERENCE_SYSTEM,
            geodeticReferenceFrame=common.GeodeticReferenceFrameType.WGS84.value,
        )
        lut_dict["geometric_metadata"].least_significant_digit = np.uint8(
            self.aux_pp2_2a.agb.compression_options.ads.geometric_luts.least_significant_digit
        )

        # COG metadata, for GN
        metadata = {
            "gn": BIOMASSL2aProductMeasurement.MetadataCOG(
                swath=common.SwathType.S1.value,
                tile_id_list=self.tile_id_list,
                basin_id_list=self.basin_id_list,
                compression=COMPRESSION_EXIF_CODES_LERC_ZSTD,  #  [LERC, ZSTD]
                image_description="",
                software="",
                dateTime=self.start_time.isoformat(timespec="microseconds")[:-1],
            )
        }
        metadata["gn"].image_description = (
            f"BIOMASS L2a {self.product_type}"
            + ": "
            + common.PixelRepresentationType.GROUND_CANCELLED_BACKSCATTER.value
        )

        return lut_dict, metadata

    def _write_to_output(
        self,
        gn_multilooked_geocoded,
        dgg_latitude_axis,
        dgg_longitude_axis,
        lut_dict,
        average_wavenumbers,
        idx_selected_reference,
        acquisition_id_reference_image,
        footprint_mask_for_quicklooks,
    ):
        """Write output GN L2a product"""

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
            gn_multilooked_geocoded,
            metadata_dict,
        )

        # main_ads_product
        orbit_direction_str = self.stack_products_list[self.primary_image_index].orbit_direction
        if orbit_direction_str not in ["ASCENDING", "DESCENDING"]:
            raise ValueError(f"Invalid orbit direction: {orbit_direction_str}")
        orbit_direction = (
            common.OrbitPassType.ASCENDING if orbit_direction_str == "ASCENDING" else common.OrbitPassType.DESCENDING
        )

        start_time_l1c = self.stack_products_list[0].start_time
        stop_time_l1c = self.stack_products_list[0].stop_time
        for l1c_product in self.stack_products_list:
            start_time_l1c = min(start_time_l1c, l1c_product.start_time)
            stop_time_l1c = max(stop_time_l1c, l1c_product.stop_time)
        assert start_time_l1c is not None
        assert stop_time_l1c is not None

        primary_product = self.stack_products_list[self.primary_image_index]
        swath_name = primary_product.swath_list[0]
        swath = common.SwathType(swath_name)

        assert primary_product.global_coverage_id is not None
        assert primary_product.major_cycle_id is not None
        assert primary_product.track_number is not None
        assert primary_product.frame_number is not None
        assert primary_product.platform_heading is not None

        absolute_orbit_numbers = []
        datatake_ids = []
        for product in self.stack_products_list:
            assert product.orbit_number is not None
            assert product.datatake_id is not None
            absolute_orbit_numbers.append(product.orbit_number)
            datatake_ids.append(product.datatake_id)

        selected_reference_image = None
        if idx_selected_reference is not None:
            assert acquisition_id_reference_image is not None
            selected_reference_image = common_annotation_l2l3.SelectedReferenceImageType(
                value=idx_selected_reference, acquisition_id=acquisition_id_reference_image
            )

        main_ads_product = common_annotation_l2l3.BIOMASSL2aMainADSproduct(
            common.MissionType.BIOMASS,
            self.tile_id_list,
            self.basin_id_list,
            common.ProductType.from_l2_joborder_tag(self.product_type),
            start_time_l1c,
            stop_time_l1c,
            self.radar_carrier_frequency,
            common.MissionPhaseIdtype(self.mission_phase_id),
            common.SensorModeType.MEASUREMENT,
            primary_product.global_coverage_id,
            swath,
            primary_product.major_cycle_id,
            absolute_orbit_numbers,
            primary_product.track_number,
            orbit_direction,
            datatake_ids,
            primary_product.frame_number,
            primary_product.platform_heading,
            self.output_baseline,
            forest_coverage_percentage=self.forest_coverage_percentage,
            selected_reference_image=selected_reference_image,
        )

        # main_ads_raster_image
        projection = common.ProjectionType.LATITUDE_LONGITUDE_BASED_ON_DGG
        datum = common.DatumType(
            coordinate_reference_system=COORDINATE_REFERENCE_SYSTEM,
            geodetic_reference_frame=common.GeodeticReferenceFrameType.WGS84,
        )
        pixel_representation = common_annotation_l2l3.PixelRepresentationChoiceType(
            gn=common.PixelRepresentationType.GROUND_CANCELLED_BACKSCATTER
        )
        pixel_type = common_annotation_l2l3.PixelTypeChoiceType(
            float_pixel_type=common.PixelTypeType.VALUE_32_BIT_FLOAT
        )
        no_data_value = common_annotation_l2l3.NoDataValueChoiceType(float_no_data_value=FLOAT_NODATA_VALUE)
        main_ads_raster_image = common_annotation_l2l3.BIOMASSL2aMainADSRasterImage(
            footprint,
            float(dgg_latitude_axis[0]),
            float(dgg_longitude_axis[0]),
            float(dgg_latitude_axis[1] - dgg_latitude_axis[0]),
            float(dgg_longitude_axis[1] - dgg_longitude_axis[0]),
            len(dgg_latitude_axis),
            len(dgg_longitude_axis),
            projection,
            datum,
            pixel_representation,
            pixel_type,
            no_data_value,
        )

        # main_ads_input_information
        polarisation_list = [common.PolarisationType.HH, common.PolarisationType.VH, common.PolarisationType.VV]

        acquisition_list = []
        overall_product_quality_indices = []
        for idx, acq_folder_name in enumerate(self.acquisition_paths_selected_not_sorted):
            quality = self.stack_products_list[idx].stack_quality
            assert quality is not None
            overall_product_quality_indices.append(quality.overall_product_quality_index)
            reference_image = (
                idx == self.primary_image_index
            )  # primary image index is ordered as the acquisitions (not sorted)

            sta_quality_parameters_list = []
            assert self.stack_products_list[0].stack_quality is not None
            assert self.stack_products_list[0].stack_quality.sta_quality_parameters_list is not None
            for param in self.stack_products_list[0].stack_quality.sta_quality_parameters_list:
                sta_quality_parameters_list.append(
                    common_annotation_l2l3.StaQualityParametersType(
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
                        polarisation=param.polarization,
                    )
                )

            sta_quality = common_annotation_l2l3.StaQualityType(
                overall_product_quality_index=quality.overall_product_quality_index,
                sta_quality_parameters_list=sta_quality_parameters_list,
            )

            acquisition_list.append(
                common_annotation_l2l3.AcquisitionType(
                    folder_name=acq_folder_name.name,
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
        vertical_wavenumbers_info = common.MinMaxTypeWithUnit(
            min=float(min(average_wavenumbers)),
            max=float(max(average_wavenumbers)),
            units=common.UomType.RAD_M,
        )

        wavenumber_spacings = np.abs(np.diff(average_wavenumbers))
        hoa_max = 2 * np.pi / min(wavenumber_spacings)
        hoa_min = 2 * np.pi / max(wavenumber_spacings)
        height_of_ambiguity_info = common.MinMaxTypeWithUnit(
            min=float(hoa_min),
            max=float(hoa_max),
            units=common.UomType.M,
        )

        main_ads_input_information = common_annotation_l2l3.BIOMASSL2aMainADSInputInformation(
            common.ProductType.from_l2_joborder_tag(self.product_type),
            overall_products_quality_index=(0 if sum(overall_product_quality_indices) == 0 else 1),
            nominal_stack=str(np.logical_or(is_int_and_nominal, is_tom_and_nominal)).lower(),
            polarisation_list=polarisation_list,
            projection=projection,
            footprint=footprint,
            vertical_wavenumbers=vertical_wavenumbers_info,
            height_of_ambiguity=height_of_ambiguity_info,
            acquisition_list=acquisition_list,
        )

        # main_ads_processing_parameters
        general_configuration = common_annotation_l2l3.GeneralConfigurationParametersType(
            apply_calibration_screen=common_annotation_l2l3.CalibrationScreenType(
                self.aux_pp2_2a.general.apply_calibration_screen.value
            ),
            forest_coverage_threshold=self.aux_pp2_2a.general.forest_coverage_threshold,
            forest_mask_interpolation_threshold=self.aux_pp2_2a.general.forest_mask_interpolation_threshold,
            subsetting_rule=common_annotation_l2l3.SubsettingRuleType(self.aux_pp2_2a.general.subsetting_rule.value),
            polarisation_combination_method=self.aux_pp2_2a.general.polarisation_combination_method,
        )

        compression_options_gn = main_annotation_l2a_gn.CompressionOptionsL2AGN(
            mds=main_annotation_l2a_gn.CompressionOptionsL2AGN.Mds(
                gn=main_annotation_l2a_gn.CompressionFactorMaxZErrorType(
                    compression_factor=self.aux_pp2_2a.agb.compression_options.mds.gn.compression_factor,
                    max_z_error=self.aux_pp2_2a.agb.compression_options.mds.gn.max_z_error,
                ),
            ),
            ads=main_annotation_l2a_gn.CompressionOptionsL2AGN.Ads(
                fnf=main_annotation_l2a_gn.CompressionFactorType(
                    compression_factor=self.aux_pp2_2a.agb.compression_options.ads.fnf.compression_factor
                ),
                geometric_luts=main_annotation_l2a_gn.CompressionFactorLeastSignificantDigitType(
                    compression_factor=self.aux_pp2_2a.agb.compression_options.ads.geometric_luts.compression_factor,
                    least_significant_digit=self.aux_pp2_2a.agb.compression_options.ads.geometric_luts.least_significant_digit,
                ),
            ),
            mds_block_size=self.aux_pp2_2a.agb.compression_options.mds_block_size,
            ads_block_size=self.aux_pp2_2a.agb.compression_options.ads_block_size,
        )

        # imagesPairSelection: the AUX contains AuxppacquisitionListType.AcquisitionFolderName objects,
        # the annotation needs AcquisitionType: pick them from the stack acquisition list
        images_pair_selection = None
        if self.aux_pp2_2a.agb.ground_cancellaton.operational_mode == OperationalModeType.INSAR_PAIR:
            pair_from_aux_pp2 = {
                Path(acq.value).name: acq for acq in self.aux_pp2_2a.agb.ground_cancellaton.images_pair_selection
            }
            images_pair_selection = [
                common_annotation_l2l3.AcquisitionType(
                    folder_name=acq.folder_name,
                    sta_quality=acq.sta_quality,
                    reference_image=pair_from_aux_pp2[acq.folder_name].reference_image,
                    average_wavenumber=pair_from_aux_pp2[acq.folder_name].average_wavenumber,
                )
                for acq in acquisition_list
                if acq.folder_name in pair_from_aux_pp2
            ]
            if len(images_pair_selection) != 2:
                raise RuntimeError(
                    f"INSAR_PAIR: expected 2 acquisitions from AUX PP2 2A images pair selection "
                    f"{list(pair_from_aux_pp2)} in the stack, found {len(images_pair_selection)}"
                )

        main_ads_processing_parameters = main_annotation_l2a_gn.BIOMASSL2aMainADSProcessingParametersGN(
            processor_version=bps.l2a_processor.__version__,
            product_generation_time=self.start_time,
            general_configuration=general_configuration,
            compression_options=compression_options_gn,
            emphasized_forest_height=self.aux_pp2_2a.agb.ground_cancellaton.emphasized_forest_height,
            operational_mode=self.aux_pp2_2a.agb.ground_cancellaton.operational_mode.value,
            radiometric_calibration_flag=self.aux_pp2_2a.agb.ground_cancellaton.radiometric_calibration_flag,
            product_resolution=self.aux_pp2_2a.agb.product_resolution,
            upsampling_factor=self.aux_pp2_2a.agb.upsampling_factor,
            images_pair_selection=images_pair_selection,
            disable_ground_cancellation_flag=self.aux_pp2_2a.agb.ground_cancellaton.disable_ground_cancellation_flag,
        )

        # lut_ads
        lut_ads = BIOMASSL2aLutAdsGN(
            lut_fnf=lut_dict["fnf"]["fnf"],
            lut_local_incidence_angle=lut_dict["local_incidence_angle"],
            lut_latitude=lut_dict["latitude"],
            lut_longitude=lut_dict["longitude"],
            lut_dem=lut_dict["height"],
            lut_incidence_angle=lut_dict["incidenceAngle"],
            lut_elevation_angle=lut_dict["elevationAngle"],
            lut_range_dem_slope=lut_dict["rangeTerrainSlope"],
            lut_azimuth_dem_slope=lut_dict["azimuthTerrainSlope"],
            lut_fnf_metadata=lut_dict["fnf_metadata"],
            lut_geometric_metadata=lut_dict["geometric_metadata"],
        )

        # Initialize GN Product
        product_to_write = BIOMASSL2aProductGN(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            lut_ads,
            product_doi=self.aux_pp2_2a.agb.l2aAGBProductDOI,
        )

        # Write to file the GN Product
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

    def run_l2a_gn_processing(self):
        """Performs processing as described in job order.

        Parameters
        ----------

        Returns
        -------
        """

        self._initialize_processing()

        (
            gn_multilooked_geocoded,
            dgg_latitude_axis,
            dgg_longitude_axis,
            lut_dict,
            average_wavenumbers,
            idx_selected_reference,
            acquisition_id_reference_image,
            footprint_mask_for_quicklooks,
        ) = self._core_processing()

        self._write_to_output(
            gn_multilooked_geocoded,
            dgg_latitude_axis,
            dgg_longitude_axis,
            lut_dict,
            average_wavenumbers,
            idx_selected_reference,
            acquisition_id_reference_image,
            footprint_mask_for_quicklooks,
        )

        processing_stop_time = datetime.now()
        elapsed_time = processing_stop_time - self.processing_start_time
        bps_logger.info(
            "%s total processing time: %.3f s",
            BPS_L2A_GN_PROCESSOR_NAME,
            elapsed_time.total_seconds(),
        )


def sigma_naught_normalisation(
    ground_cancelled_squared_list: list[np.ndarray],
    incidence_angle_rad: np.ndarray,
    terrain_slope_rad: np.ndarray,
    lut_axis_az_s: np.ndarray,
    lut_axis_sr_s: np.ndarray,
    scs_axis_az_s: np.ndarray,
    scs_axis_sr_s: np.ndarray,
    b_az: float,
    b_rg: float,
    average_az_velocity: float,
    product_resolution: float,
    upsampling_factor: int,
    radiometric_calibration_flag: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sigma Naught Normalisation

    Parameters
    ----------
    ground_cancelled_squared_list: List[np.ndarray],
        Ground cancelled data (square value), it is a list of P polarizations,
        each of dimensions [N_az x N_rg]
    incidence_angle_rad: np.ndarray,
        Reference acquisition incidence angle [rad], dimensions [N_az_lut x N_rg_lut]
    terrain_slope_rad: np.ndarray,
        Rreference acquisition terrain slope [rad], dimensions [N_az_lut x N_rg_lut]
    b_az: float
        L1c azimuth bandwidth [Hz]
    b_rg: float
        L1c range bandwidth [Hz]
    average_az_velocity: float
        Average azimuth velocity [m/s]
    product_resolution: float
        Product resolution in [m]
    upsampling_factor: int
        Upsampling factor for covariance, in azimuth and range directions
    radiometric_calibration_flag: bool
       Flag to perform or disalble radiometric calibration
       If dlsabled, the input data is just multilooked/decimated

    Returns:
    ground_canc_norm_multilook:
        Ground cancelled data (square value), sigma naught normalized and multilooked
        dimensions [P, N_az_sub, N_rg_sub]
    local_inc_angle_rad_multilook,
        Locak incidence angle (computed as incidence angle minus terrain slope), multilooked
        dimensions [N_az_sub, N_rg_sub]
    axis_az_subsampling_indexes: int
        indices for the original azimuth axis, to obtain the decimated output mpmb_covariance data axis
    axis_rg_subsampling_indexes: int
        indices for the original slant range axis, to obtain the decimated output mpmb_covariance data axis
    """

    start_time = datetime.now()

    # Logging
    bps_logger.info("Sigma Naught Normalisation:")
    bps_logger.info(f"    using AUX PP2 2A product resolution: {product_resolution} [m]")
    bps_logger.info(f"    using AUX PP2 2A upsampling factor: {upsampling_factor}")

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

    decimation_factor_rg = np.ceil(averaging_window_size_rg / upsampling_factor).astype(np.uint8)
    decimation_factor_az = np.ceil(averaging_window_size_az / upsampling_factor).astype(np.uint8)
    bps_logger.info(f"    decimation factor used in range direction: {decimation_factor_rg}")
    bps_logger.info(f"    decimation factor used in azimuth direction: {decimation_factor_az}")

    bps_logger.info("    preliminary interpolation to bring incidence angle onto the same L1c grid")

    incidence_angle_rad = parallel_reinterpolate(
        [incidence_angle_rad],  # function works with lists
        lut_axis_az_s,
        lut_axis_sr_s,
        scs_axis_az_s,
        scs_axis_sr_s,
    )[0]  # function works with lists

    bps_logger.info("    preliminary interpolation to bring terrain slope onto the same L1c grid")
    terrain_slope_rad = parallel_reinterpolate(
        [terrain_slope_rad],  # function works with lists
        lut_axis_az_s,
        lut_axis_sr_s,
        scs_axis_az_s,
        scs_axis_sr_s,
    )[0]  # function works with lists

    # local incidence angle (wrt. ground surface normal)
    bps_logger.info("    computing local incidence angle (theta - slope)")
    local_incidence_angle_rad = np.sin(incidence_angle_rad - terrain_slope_rad)

    # prepare sparse matrices for the normalisation
    (
        fa_normalized,
        axis_az_subsampling_indexes,
        fr_normalized_transposed,
        axis_rg_subsampling_indexes,
    ) = build_filtering_sparse_matrices(
        ground_cancelled_squared_list[0].shape[0],  # azimuth shape
        ground_cancelled_squared_list[0].shape[1],  # slant range shape
        averaging_window_size_rg,
        decimation_factor_rg,
        averaging_window_size_az,
        decimation_factor_az,
    )
    num_az_subsampled = axis_az_subsampling_indexes.size
    num_rg_subsampled = axis_rg_subsampling_indexes.size

    if radiometric_calibration_flag:
        # local incidence angle (wrt. ground surface normal) radiometric correction
        bps_logger.info("    applying local incidence angle for radiometric calibration, as enabled in AUX PP2")
        ground_cancelled_squared_list = [
            local_incidence_angle_rad * ground_cancelled_squared
            for ground_cancelled_squared in ground_cancelled_squared_list
        ]
    else:
        bps_logger.info("    radiometric calibration using local incidence angle is disabled from AUX PP2")

    # Input to sigma_naught_normalisation_core should be a list of pols containing one acq
    bps_logger.info("    performing sigma naught normalisation and multilooking:")
    ground_canc_norm_multilook = sigma_naught_normalisation_core(
        ground_cancelled_squared_list,
        fa_normalized,
        num_az_subsampled,
        fr_normalized_transposed,
        num_rg_subsampled,
    ).astype(np.float64)

    bps_logger.info(
        f"    data shape after decimation: Azimuth {ground_canc_norm_multilook.shape[1]} samples, Slant-range {ground_canc_norm_multilook.shape[2]} samples"
    )
    bps_logger.info("    performing incidence angle multilooking")
    local_inc_angle_rad_multilook = (
        sigma_naught_normalisation_core(
            [local_incidence_angle_rad],
            fa_normalized,
            num_az_subsampled,
            fr_normalized_transposed,
            num_rg_subsampled,
        )
        .squeeze()
        .astype(np.float64)
    )

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Sigma Naught Normalisation processing time: {elapsed_time:2.1f} s")

    return (
        ground_canc_norm_multilook,
        local_inc_angle_rad_multilook,
        axis_az_subsampling_indexes,
        axis_rg_subsampling_indexes,
    )


def sigma_naught_normalisation_core(
    input_data_list: list[np.ndarray],
    fa_normalized: np.ndarray,
    num_az_subsampled: int,
    fr_normalized_transposed: np.ndarray,
    num_rg_subsampled: int,
) -> np.ndarray:
    """Sigma naught normalisation

    Parameters
    ----------
    input_data_list: List[np.ndarray]
        Is the squared ground cancelled data: I_GN^2
        It can be the one calibratedwith the local incidence angle: sin(local_incidence_angle) * I_GN^2
            (calibration performed outside and final function is passed here)
        It is a list of P=3 polarizations (in the order HH, XP, VV), with images [N_az x N_rg]
    fa_normalized: np.ndarray
        Azimuth normalized sparse matrix (see build_filtering_sparse_matrices)
    num_az_subsampled: int
        number of azimuth samples of the dacimated output mpmb_covariance data
    fr_normalized_transposed: np.ndarray
         Slant range normalized sparse matrix (see build_filtering_sparse_matrices)
    num_rg_subsampled: int
        number of slant range samples of the dacimated output mpmb_covariance data

    Returns
    -------
    ground_canc_norm_multilook: np.ndarray
        Sigma naught normalized ground cancelled data
        Dimensions [P x num_az_subsampled x num_rg_subsampled]
        Subsampled respect input_data_list [N_az x N_rg] dimensions.
    """

    # Inputs computation
    # Lista contenente solo 3 polarizzazioni
    num_pols = len(input_data_list)
    num_az_in, num_rg_in = input_data_list[0].shape

    # output initialization
    ground_canc_norm_multilook = np.zeros(
        (
            num_pols,
            num_az_subsampled,
            num_rg_subsampled,
        ),
        dtype=type(input_data_list[0][0, 0]),
    )

    # masking for invalid values
    nodata_values_mask = np.zeros((num_az_in, num_rg_in), dtype=bool)
    for image in input_data_list:
        nodata_values_mask = np.logical_or(nodata_values_mask, np.isnan(image))
    for image in input_data_list:
        image[nodata_values_mask] = 0

    for ch_p, pol_idx_p in enumerate(range(num_pols)):
        if num_pols > 1:
            bps_logger.info(f"        polarization {ch_p + 1} of {num_pols}")

        temp = fa_normalized @ input_data_list[pol_idx_p]
        ground_canc_norm_multilook[pol_idx_p, :, :] = temp @ fr_normalized_transposed

    return ground_canc_norm_multilook
