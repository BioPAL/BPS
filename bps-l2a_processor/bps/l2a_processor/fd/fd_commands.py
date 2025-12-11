# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
FD commands
-----------
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import bps.l2a_processor
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.fnf_utils import FnFMask
from bps.common.io import common_types
from bps.common.l2_joborder_tags import L2A_OUTPUT_PRODUCT_FD
from bps.l2a_processor.core.aux_pp2_2a import AuxProcessingParametersL2A
from bps.l2a_processor.core.joborder_l2a import L2aJobOrder
from bps.l2a_processor.fd import BPS_L2A_FD_PROCESSOR_NAME
from bps.l2a_processor.ground_cancellation import ground_cancellation
from bps.l2a_processor.l2a_common_functionalities import (
    averaging_windows_sizes,
    build_filtering_sparse_matrices,
    check_lat_lon_orientation,
    geocoding,
    geocoding_update_dem_coordinates,
    get_dgg_sampling,
    interpolate_fnf,
    mpmb_covariance_estimation,
    refine_dgg_search_tiles,
)
from bps.transcoder.io import common_annotation_models_l2
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aLutAdsFD,
    BIOMASSL2aMainADSInputInformation,
    BIOMASSL2aMainADSProcessingParametersFD,
    BIOMASSL2aMainADSproduct,
    BIOMASSL2aMainADSRasterImage,
    BIOMASSL2aProductFD,
    BIOMASSL2aProductMeasurement,
    main_annotation_models_l2a_fd,
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
from scipy.stats import chi2

LIGHTSPEED = 299792458


class FD:
    """Forest Disturbance L2a Processor"""

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
        l2a_fd_product: BIOMASSL2aProductFD | None,
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
                self.scs_axes_dict["scs_axis_sr_s"]: np.ndarray
                    slant range temporal axis, in seconds
                self.scs_axes_dict["scs_axis_az_s"]: np.ndarray
                    azimuth temporal axis, in seconds
                self.scs_axes_dict["scs_axis_az_mjd"]: PreciseDateTime
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
        l2a_fd_product: BIOMASSL2aProduct
            Optional previous cycle computation FD product
            If present, FD algorithm is the second cylce one
            If absent, FD algorithm is at first cycle and some MDS are not computed
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
        self.l2a_fd_product = l2a_fd_product
        self.stack_lut_axes_dict = stack_lut_axes_dict
        self.primary_image_index = primary_image_index
        self.acquisition_paths_selected_not_sorted = acquisition_paths_selected_not_sorted
        self.output_baseline = 0  # set after from job order

    def _initialize_processing(self):
        """Initialize the FD L2a processing"""

        self.processing_start_time = datetime.now()
        self.start_time = PreciseDateTime.now()
        bps_logger.info("%s started", BPS_L2A_FD_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baselines is not None:
            for output_product, output_baseline in zip(self.job_order.output_products, self.job_order.output_baselines):
                if output_product == L2A_OUTPUT_PRODUCT_FD:
                    self.output_baseline = output_baseline

        self.product_type = L2A_OUTPUT_PRODUCT_FD

    def _core_processing(self):
        """Execute core FD L2a processing"""

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
            self.aux_pp2_2a.fd.ground_cancellaton,
            self.acquisition_paths_selected_not_sorted,
        )
        acquisition_id_reference_image = (
            self.acquisition_paths_selected_not_sorted[idx_selected_reference].name
            if idx_selected_reference is not None
            else None
        )

        # Polarimetric Covariance Matrix Computation
        average_azimuth_velocity = np.mean(
            [
                np.linalg.norm(velocity)
                for velocity in self.stack_products_list[self.primary_image_index].general_sar_orbit[0].velocity_vector
            ]
        )
        (
            covariance_vec_9x,
            number_of_looks,
            axis_az_subsampling_indexes,
            axis_rg_subsampling_indexes,
        ) = polarimetric_covariance_matrix_computation(
            ground_cancelled,
            np.deg2rad(self.stack_lut_list[self.primary_image_index]["incidenceAngle"].astype(np.float32)),
            self.stack_products_list[self.primary_image_index].sampling_constants_list[0].baz_hz,
            self.stack_products_list[self.primary_image_index].sampling_constants_list[0].brg_hz,
            self.scs_axes_dict["scs_axis_az_s"],
            self.scs_axes_dict["scs_axis_sr_s"],
            average_azimuth_velocity,
            self.aux_pp2_2a.fd.product_resolution,
            self.aux_pp2_2a.fd.upsampling_factor,
        )

        # Get the DGG sampling parameters, needed for the geocoding step:
        # Check the orientation of FNF lat lon axis and keep this convention in the L2a product.
        invert_latitude, invert_longitude = check_lat_lon_orientation(self.fnf.lat_axis, self.fnf.lon_axis)
        (
            dgg_latitude_axis_deg,
            dgg_longitude_axis_deg,
        ) = get_dgg_sampling(
            self.latlon_coverage,
            create_dgg_sampling_dict(L2A_OUTPUT_PRODUCT_FD),
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

        bps_logger.info("Geocoding #9 polarimetric layers:")
        # initialize geocoded covariance
        cov_geocoded_vec_9x = np.ones(
            shape=(
                9,
                len(dgg_latitude_axis_deg),
                len(dgg_longitude_axis_deg),
            ),
            dtype=covariance_vec_9x.dtype,
        )

        # Convert covariance_vec_9x to list, in order to compute it faster in a single Delaunay call
        # Than convert it back to np.array after computation
        # Also cast the optput type
        cov_geocoded_vec_9x = np.array(
            geocoding(
                list(covariance_vec_9x),
                delaunay,
                dgg_latitude_mesh_rad,
                dgg_longitude_mesh_rad,
                dem_valid_values_mask,
                fill_value=np.nan,
            )
        ).astype(type(covariance_vec_9x[0, 0, 0]))

        # Forest Disturbance computation
        lut_dict = {}
        if self.l2a_fd_product:
            (
                fd,
                cfm,
                probability_ofchange,
                lut_dict["acm"],
                lut_dict["number_of_averages"],
                lut_dict["fnf"],
            ) = change_detection(
                cov_geocoded_vec_9x,
                dgg_latitude_axis_deg,
                dgg_longitude_axis_deg,
                number_of_looks,
                self.aux_pp2_2a.general.forest_mask_interpolation_threshold,
                self.aux_pp2_2a.fd.significance_level,
                self.fnf,
                len(self.scs_pol_list_calibrated),
                self.l2a_fd_product.lut_ads.lut_acm,
                self.l2a_fd_product.measurement.data_dict["cfm"],
                self.l2a_fd_product.lut_ads.lut_number_of_averages,
                self.aux_pp2_2a.fd.numerical_determinant_limit,
            )
        else:
            (
                fd,
                cfm,
                probability_ofchange,
                lut_dict["acm"],
                lut_dict["number_of_averages"],
                lut_dict["fnf"],
            ) = change_detection(
                cov_geocoded_vec_9x,
                dgg_latitude_axis_deg,
                dgg_longitude_axis_deg,
                number_of_looks,
                self.aux_pp2_2a.general.forest_mask_interpolation_threshold,
                self.aux_pp2_2a.fd.significance_level,
                self.fnf,
                len(self.scs_pol_list_calibrated),
                None,
                None,
                None,
                self.aux_pp2_2a.fd.numerical_determinant_limit,
            )

        # Casting back to float32 all the float data before saving

        # For consistency with other processors, put each MDS in a list (in GN there are three values in each list)
        processed_data_dict = {
            "fd": [fd],
            "cfm": [cfm],
            "probability_ofchange": [probability_ofchange.astype(np.float32)],
        }
        lut_dict["acm"] = lut_dict["acm"].astype(np.float32)

        # Footprint mask for quick looks transparency
        fd_for_footprint = cov_geocoded_vec_9x[0, :, :]
        fd_for_footprint[fd_for_footprint == FLOAT_NODATA_VALUE] = np.nan
        if AVERAGING_FACTOR_QUICKLOOKS > 1:
            fd_for_footprint = fd_for_footprint[::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS]
        footprint_mask_for_quicklooks = np.logical_not(np.isnan(fd_for_footprint))

        return (
            processed_data_dict,
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

        lut_metadata = BIOMASSL2aLutAdsFD.LutMetadata(
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
        lut_dict["fnf_metadata"] = deepcopy(lut_metadata)
        lut_dict["number_of_averages_metadata"] = deepcopy(lut_metadata)
        lut_metadata.least_significant_digit = self.aux_pp2_2a.fd.compression_options.ads.acm.least_significant_digit

        lut_metadata.pixelType = "32 bit Float"
        lut_metadata.no_data_value = FLOAT_NODATA_VALUE
        lut_dict["acm_metadata"] = deepcopy(lut_metadata)

        # COG metadata
        # fill common fileds for FD, CFM and Number of Averages
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
            for idx in range(3)
        ]

        # specific fileds for FD, CFM and Number of Averages
        meta_data = {
            "fd": deepcopy(meta_data_temp[0]),
            "probability_ofchange": deepcopy(meta_data_temp[1]),
            "cfm": deepcopy(meta_data_temp[2]),
        }
        meta_data["fd"].image_description = (
            f"BIOMASS L2a {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_DISTURBANCE.value
        )

        meta_data["fd"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data["probability_ofchange"].image_description = (
            f"BIOMASS L2a {self.product_type}" + ": " + common_types.PixelRepresentationType.PROBABILITY_OF_CHANGE.value
        )

        meta_data["cfm"].image_description = (
            f"BIOMASS L2a {self.product_type}" + ": " + common_types.PixelRepresentationType.COMPUTED_FOREST_MASK.value
        )
        meta_data["cfm"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        return lut_dict, meta_data

    def _write_to_output(
        self,
        processed_data_dict,
        dgg_latitude_axis,
        dgg_longitude_axis,
        lut_dict,
        average_wavenumbers,
        idx_selected_reference,
        acquisition_id_reference_image,
        footprint_mask_for_quicklooks,
    ):
        """Write output FD L2a product"""

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
            selected_reference_image=idx_selected_reference,
            acquisition_id_reference_image=acquisition_id_reference_image,
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
            "fd": common_types.PixelRepresentationType.FOREST_DISTURBANCE,
            "probability_ofchange": common_types.PixelRepresentationType.PROBABILITY_OF_CHANGE,
            "cfm": common_types.PixelRepresentationType.COMPUTED_FOREST_MASK,
        }
        pixel_representation = common_annotation_models_l2.PixelRepresentationChoiceType(
            fd=pixel_representation_dict["fd"],
            cfm=pixel_representation_dict["cfm"],
            probability_of_change=pixel_representation_dict["probability_ofchange"],
        )
        pixel_type = common_annotation_models_l2.PixelTypeChoiceType(
            float_pixel_type=main_annotation_models_l2a_fd.PixelTypeType("32 bit Float"),
            int_pixel_type=main_annotation_models_l2a_fd.PixelTypeType("8 bit Unsigned Integer"),
        )

        no_data_value = common_annotation_models_l2.NoDataValueChoiceType(
            float_no_data_value=FLOAT_NODATA_VALUE,
            int_no_data_value=INT_NODATA_VALUE,
        )
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
                    average_wavenumber=average_wavenumbers[idx],
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

        least_significant_digit_acm = self.aux_pp2_2a.fd.compression_options.ads.acm.least_significant_digit

        compression_options_fd = main_annotation_models_l2a_fd.CompressionOptionsL2A(
            main_annotation_models_l2a_fd.CompressionOptionsL2A.Mds(
                main_annotation_models_l2a_fd.CompressionOptionsL2A.Mds.Fd(
                    self.aux_pp2_2a.fd.compression_options.mds.fd.compression_factor
                ),
                main_annotation_models_l2a_fd.CompressionOptionsL2A.Mds.ProbabilityOfchange(
                    self.aux_pp2_2a.fd.compression_options.mds.probability_of_change.compression_factor,
                    self.aux_pp2_2a.fd.compression_options.mds.probability_of_change.max_z_error,
                ),
                main_annotation_models_l2a_fd.CompressionOptionsL2A.Mds.Cfm(
                    self.aux_pp2_2a.fd.compression_options.mds.cfm.compression_factor
                ),
            ),
            main_annotation_models_l2a_fd.CompressionOptionsL2A.Ads(
                main_annotation_models_l2a_fd.CompressionOptionsL2A.Ads.Fnf(
                    self.aux_pp2_2a.fd.compression_options.ads.fnf.compression_factor
                ),
                main_annotation_models_l2a_fd.CompressionOptionsL2A.Ads.Acm(
                    self.aux_pp2_2a.fd.compression_options.ads.acm.compression_factor,
                    least_significant_digit_acm,
                ),
                main_annotation_models_l2a_fd.CompressionOptionsL2A.Ads.NumberOfAverages(
                    self.aux_pp2_2a.fd.compression_options.ads.number_of_averages.compression_factor
                ),
            ),
            self.aux_pp2_2a.fd.compression_options.mds_block_size,
            self.aux_pp2_2a.fd.compression_options.ads_block_size,
        )

        main_ads_processing_parameters = BIOMASSL2aMainADSProcessingParametersFD(
            bps.l2a_processor.__version__,
            self.start_time,
            general_configuration,
            compression_options_fd,
            self.aux_pp2_2a.fd.ground_cancellaton.emphasized_forest_height,
            self.aux_pp2_2a.fd.ground_cancellaton.operational_mode.value,
            self.aux_pp2_2a.fd.significance_level,
            self.aux_pp2_2a.fd.product_resolution,
            self.aux_pp2_2a.fd.numerical_determinant_limit,
            self.aux_pp2_2a.fd.upsampling_factor,
            self.aux_pp2_2a.fd.ground_cancellaton.images_pair_selection,
            self.aux_pp2_2a.fd.ground_cancellaton.disable_ground_cancellation_flag,
        )

        # lut_ads
        lut_ads = BIOMASSL2aLutAdsFD(
            lut_dict["fnf"]["fnf"],
            lut_dict["acm"],
            lut_dict["number_of_averages"],
            lut_dict["fnf_metadata"],
            lut_dict["acm_metadata"],
            lut_dict["number_of_averages_metadata"],
        )

        # Initialize FD Product
        product_to_write = BIOMASSL2aProductFD(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            lut_ads,
            product_doi=self.aux_pp2_2a.fd.l2aFDProductDOI,
        )

        # Write to file the FD Product
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
            (self.job_order.input_l2a_fd_product.name if self.job_order.input_l2a_fd_product else None),
        )
        write_obj.write()

    def run_l2a_fd_processing(self):
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
            idx_selected_reference,
            acquisition_id_reference_image,
            footprint_mask_for_quicklooks,
        ) = self._core_processing()

        self._write_to_output(
            processed_data_dict,
            dgg_latitude_axis,
            dgg_longitude_axis,
            lut_dict,
            average_wavenumbers,
            idx_selected_reference,
            acquisition_id_reference_image,
            footprint_mask_for_quicklooks,
        )

        elapsed_time = self.processing_stop_time - self.processing_start_time
        bps_logger.info(
            "%s total processing time = %.3f s",
            BPS_L2A_FD_PROCESSOR_NAME,
            elapsed_time.total_seconds(),
        )


def polarimetric_covariance_matrix_computation(
    ground_cancelled_list: list[np.ndarray],
    incidence_angle_rad: np.ndarray,
    b_az: float,
    b_rg: float,
    scs_axis_az_s: list[np.ndarray],
    scs_axis_sr_s: list[np.ndarray],
    average_az_velocity: float,
    product_resolution: float,
    upsampling_factor: int,
) -> tuple[
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Polarimetric covariance matrix computation

    Parameters
    ----------
    ground_cancelled_list: List[np.ndarray]
        Ground cancelled complex image,
        ordered list of three polarizations, in the order HH, XP, VV
        each of dimensions [N_az x N_rg]
    incidence_angle_rad: np.ndarray
        Reference acquisition incidence angle [rad]
        dimensions [N_az_theta x N_rg_theta]
    b_az: float
        L1c azimuth bandwidth [Hz]
    b_rg: float
        L1c range bandwidth [Hz]
    scs_axis_az_s:  List[np.ndarray]
        azimuth time axis[s]
    scs_axis_sr_s:  List[np.ndarray]
        slant range time axis [s]
    average_az_velocity: float
        Average azimuth velocity [m/s]
    product_resolution: float
        Value in [m] to be used as the resolution on ground range map
        and also to perform the covariance averaging in radar coordinates
    upsampling_factor: int
        Upsampling factor for covariance, in azimuth and range directions

    Returns
    -------
    cov: np.ndarray
        Polarimetric covariance of the ground cancelled complex data of current stack
        dimensions [9 x N_az x N_rg]
        The output dimensions are reshaped from the original [N_pol x N_pol x N_az x N_rg]
        to reduce space occupancy, due to herimtian symmetry: see _covariance_matrix_mat2vec
    number_of_looks: int
        Number of looks
    axis_rg_subsampling_indexes: np.ndarray
        Range indices to be applied to the input_data_list axis of dimensions [N_az x N_rg],
        to get the decimated mpmb_covariance sampling of dimensions [Naz_subsampled x Nrg_subsampled]
    axis_az_subsampling_indexes: np.ndarray
        Azimuth indices to be applied to the input_data_list axis of dimensions [N_az x N_rg],
        to get the decimated mpmb_covariance sampling of dimensions [Naz_subsampled x Nrg_subsampled]
    """

    start_time = datetime.now()

    # Logging
    bps_logger.info("Compute polarimetric covariance matrix:")
    bps_logger.info(f"    using AUX PP2 2A product resolution: {product_resolution} [m]")
    bps_logger.info(f"    using AUX PP2 2A upsampling factor: {upsampling_factor}")

    # Compute averraging windows sizes, decimation factors and number of looks
    average_incidence_angle_rad = np.nanmean(incidence_angle_rad)
    (
        averaging_window_size_az,
        averaging_window_size_rg,
        number_of_looks,
    ) = averaging_windows_sizes(
        b_az,
        b_rg,
        1 / (scs_axis_az_s[1] - scs_axis_az_s[0]),
        1 / (scs_axis_sr_s[1] - scs_axis_sr_s[0]),
        product_resolution,
        average_az_velocity,
        average_incidence_angle_rad,
    )
    decimation_factor_rg = np.ceil(averaging_window_size_rg / upsampling_factor).astype(np.uint8)
    decimation_factor_az = np.ceil(averaging_window_size_az / upsampling_factor).astype(np.uint8)

    bps_logger.info(f"    decimation factor used in range direction: {decimation_factor_rg}")
    bps_logger.info(f"    decimation factor used in azimuth direction: {decimation_factor_az}")
    bps_logger.info(f"    computed number of looks value: {number_of_looks}")

    # prepare sparse matriced for the MPMB covariance estimation
    (
        fa_normalized,
        axis_az_subsampling_indexes,
        fr_normalized_transposed,
        axis_rg_subsampling_indexes,
    ) = build_filtering_sparse_matrices(
        ground_cancelled_list[0].shape[0],  # azimuth shape
        ground_cancelled_list[0].shape[1],  # slant range shape
        averaging_window_size_rg,
        decimation_factor_rg,
        averaging_window_size_az,
        decimation_factor_az,
    )
    num_az_subsampled = axis_az_subsampling_indexes.size
    num_rg_subsampled = axis_rg_subsampling_indexes.size

    # Input to mpmb_covariance_estimation should be a list of pols containing a list of acq
    cov = mpmb_covariance_estimation(
        [[ground_cancelled_acq] for ground_cancelled_acq in ground_cancelled_list],
        fa_normalized,
        num_az_subsampled,
        fr_normalized_transposed,
        num_rg_subsampled,
    )

    # covariance is reshaped from 3x3xNazxNrg to 6xNazxNrg
    cov = _covariance_matrix_mat2vec(cov)

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Polarimetric covariance matrix processing time: {elapsed_time:2.1f} s")
    return (
        cov,
        number_of_looks,
        axis_az_subsampling_indexes,
        axis_rg_subsampling_indexes,
    )


def _covariance_matrix_mat2vec(matrix_in: np.ndarray) -> np.ndarray:
    """Internal function to reshape an input matrix 3x3xNxM
    to an output matrix 9xNxM.

    Details
    This is used for the covariance matrix, which has herimtian symmetry, so:
        the elements under the diagonal are not saved in the output reshaped matrix.
        the elements over the diagonal are complex numbers and saved in abs + phase separated layers

    Elements in matrix in are stored in the optput one with following indexing convention:

    Those are the 3x3 matrix_in elements (for each N, M) indexes of extraction:
    [ 0  1  2
      /  4  5
      /  /  8 ]

    where:
        0, 4, 8 are real numbers
        1, 2, 5 are complex numbers

        0 is HH-HH
        1 is HH-XP
        2 is HH-VV
        4 is XP-XP
        5 is XP-VV
        8 is VV-VV

    Those are the 9 matrix_out elements (for each N, M)  indexes of insertion
    [ 0 abs(1) phase(1) abs(2) phase(2) 4 abs(5) phase(5) 8 ]
    phase are computed in radiants.

    Parameters:
    ----------
    matrix_in: np.ndarray
        Input matrix to be reshaped, losing the elements under the diagonal
        It can have a shape [3x3xNxM] or [3x3]

    Returns:
    -------
    matrix_out: np.ndarray
        Output reshaped matrix which misses the elements under the diagonal
        It can have a shape [9xNxM] or [9,]
    """
    #                    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    extraction_indexes = [0, 1, 1, 2, 2, 4, 5, 5, 8]

    # indexes of the complex numbers in the original hermitian matrix
    angle_indexes_in = [1, 2, 5]
    # indexes in the output vector where to put angle values
    angle_indexes_out = [2, 4, 7]

    shapes = matrix_in.shape
    if matrix_in.ndim == 4:
        # [3x3xNxM] case
        N = shapes[2]
        M = shapes[3]

        # first reshape and cast all to abs
        matrix_out = np.abs(matrix_in.reshape((9, N, M))[extraction_indexes])

        # than insert the angle values in the correct positions
        matrix_out[angle_indexes_out, :, :] = np.angle(matrix_in.reshape((9, N, M))[angle_indexes_in])

    else:
        # [3x3] case
        matrix_out = abs(matrix_in.reshape((9))[extraction_indexes]).astype(np.float64)
        matrix_out[angle_indexes_out] = np.angle(matrix_in.reshape((9))[angle_indexes_in]).astype(np.float64)

    return matrix_out


def _covariance_matrix_vec2mat(matrix_in):
    """Internal function to reshape an input matrix 9xNxM
    to an output matrix 3x3xNxM.

    Details
    This is the inverse function of _covariance_matrix_mat2vec,
    go there for further details.

    Parameters:
    ----------
    matrix_in: np.ndarray
        Input matrix to be reshaped back
        It can have a shape [9xNxM] or [9,]

    Returns:
    -------
    matrix_out: np.ndarray
        Output reshaped matrix
         It can have a shape [3x3xNxM] or [3x3]
    """

    # Input matrix values and indexing of each value:
    # [ 0      abs(1)  phase(1)  abs(2)  phase(2)  4      abs(5)  phase(5)  8    ]
    # [ idx0   idx1    idx2      idx3    idx4      idx5   idx6    idx7      idx8 ]

    # Values to be inserted in this order:
    #        [ 0         1     2
    #          cj(1)     4     5
    #          cj(2)  cj(5)    8 ]

    # how to index the input vector to obtain the result:
    #        [ idx0               idx1*exp(j idx2)    idx3*exp(j idx4)
    #          idx1*exp(-j idx2)      idx5            idx6*exp(j idx7)
    #          idx3*exp(-j idx4)  idx6*exp(-j idx7)               idx8 ]

    if matrix_in.ndim == 3:
        # [9xNxM] case
        N = matrix_in.shape[1]
        M = matrix_in.shape[2]

        matrix_out = np.zeros((3, 3, N, M)).astype(np.complex64)
        matrix_out[0, 0, :, :] = matrix_in[0, :, :]
        matrix_out[0, 1, :, :] = matrix_in[1, :, :] * np.exp(1j * matrix_in[2, :, :])
        matrix_out[0, 2, :, :] = matrix_in[3, :, :] * np.exp(1j * matrix_in[4, :, :])

        matrix_out[1, 0, :, :] = matrix_in[1, :, :] * np.exp(-1j * matrix_in[2, :, :])
        matrix_out[1, 1, :, :] = matrix_in[5, :, :]
        matrix_out[1, 2, :, :] = matrix_in[6, :, :] * np.exp(1j * matrix_in[7, :, :])

        matrix_out[2, 0, :, :] = matrix_in[3, :, :] * np.exp(-1j * matrix_in[4, :, :])
        matrix_out[2, 1, :, :] = matrix_in[6, :, :] * np.exp(-1j * matrix_in[7, :, :])
        matrix_out[2, 2, :, :] = matrix_in[8, :, :]

    elif matrix_in.ndim == 1:
        # [9,] case
        matrix_out = np.zeros((3, 3)).astype(np.complex64)
        matrix_out[0, 0] = matrix_in[0]
        matrix_out[0, 1] = matrix_in[1] * np.exp(1j * matrix_in[2])
        matrix_out[0, 2] = matrix_in[3] * np.exp(1j * matrix_in[4])
        matrix_out[1, 0] = matrix_in[1] * np.exp(-1j * matrix_in[2])
        matrix_out[1, 1] = matrix_in[5]
        matrix_out[1, 2] = matrix_in[6] * np.exp(1j * matrix_in[7])
        matrix_out[2, 0] = matrix_in[3] * np.exp(-1j * matrix_in[4])
        matrix_out[2, 1] = matrix_in[6] * np.exp(-1j * matrix_in[7])
        matrix_out[2, 2] = matrix_in[8]

    return matrix_out


def change_detection(
    covariance_curr_vec_9x: np.ndarray,
    dgg_latitude_axis_deg: np.ndarray,
    dgg_longitude_axis_deg: np.ndarray,
    number_of_looks: int,
    forest_mask_interpolation_threshold: float,
    significance_level_percent: float,
    fnf: FnFMask,
    num_pols: int | None = 3,
    previous_fd_acm: np.ndarray | None = None,
    previous_fd_cfm: np.ndarray | None = None,
    previous_fd_number_of_averages: np.ndarray | None = None,
    fd_determinant_limit: float | None = 1.0e-12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Change detection

    Parameters
    ----------
    covariance_curr_vec_9x: np.ndarray
        Polarimetric covariance of the ground cancelled complex data of current stack projected onto a geographic map
        dimensions [9 x N_lat x N_lon]
    dgg_latitude_axis_deg: np.ndarray
        Whole latitude vector [deg] in DGG sampling, covering the input footprint region
    dgg_longitude_axis_deg: np.ndarray
        Whole longitude vector [deg] in DGG sampling, covering the input footprint region
    number_of_looks: int
        Number of looks
    forest_mask_interpolation_threshold: float
        Threshold to fix rounding of pixels with decimal values originated
    significance_level_percent: float
        Significance [%] level to be applied in the change detection algorithm
    fnf: FnFMask,
        external forest-non-forest mask object with FNF itself and FNF lat lon axis
    lut_ads_fd_product: Optional[BIOMASSL2aLutAdsFD]
        Optional LUT ADS from FD L2a product of previous cycle:
            if absent, algorithm is at first cycle
            if present, algorithm is at second cycle
        Its a dict which contains:
        ACM: Average Covariance Matrix [N_pol x N_pol x N_lat x N_lon]
        CFM: Computed Forest Mask [N_pol x N_pol x N_lat x N_lon]
        number_of_averages: [N_lat x N_lon]
    num_pols: Optional[int]
        number of polarizations

    Returns
    -------
    fd_map: np.ndarray
        Boolean Forest Disturbance Map (integers 0,1), [N_lat x N_lon]
        If at first cycle, is a map filled with int no data values
    cfm: np.ndarray
        If at second cycle, is the Computed Forest Mask (updated) from input CFM
        If at first cycle, it is the input FNF, interpolated to fd_map axes
        [N_lat x N_lon]
    probability_of_change: np.ndarray
        Map of float numbers, probability of change, [N_lat x N_lon]
        If at first cycle, is a map filled with float no data values
    Y_ACM: np.ndarray
        Computed (at first cycle) or updated (at second cycle) Average Covariance Matrix
        None layers, each of shape [N_lat x N_lon]
    number_of_averages: np.ndarray
        Computed (at first cycle) or updated (at second cycle) number of averages, [N_lat x N_lon]
    lut_fnf_interp_dict:
        forest-non-forest mask here defined in the same latitude-longitude based on DGG projection used for L2a images
        and cropped to have a coverage containing the L2a product image boundaries
        dictionary with FNF itself and FNF lat lon axis
    """

    start_time = datetime.now()
    bps_logger.info("Compute change detection:")
    second_cycle_run = False
    if previous_fd_acm is not None and previous_fd_cfm is not None and previous_fd_number_of_averages is not None:
        second_cycle_run = True

    # Generate a mask for not valid pixels (where input covariance is NaN)
    pixel_is_not_valid = np.zeros((covariance_curr_vec_9x.shape[1], covariance_curr_vec_9x.shape[2])).astype(bool)
    for covariance_curr_vec_i in covariance_curr_vec_9x:
        pixel_is_not_valid = np.logical_or(pixel_is_not_valid, np.isnan(covariance_curr_vec_i))

    if second_cycle_run:
        bps_logger.info("    algorithm considering previous cycle computation (from FD product specified in job order)")
        bps_logger.info(f"    using AUX PP2 2A significance level: {significance_level_percent}%")

        # update the pixel_is_not_valid mask considering also previous ACM
        # (where input ACM is FLOAT_NODATA_VALUE)
        acm_is_not_valid = np.zeros((covariance_curr_vec_9x.shape[1], covariance_curr_vec_9x.shape[2])).astype(bool)
        for covariance_curr_vec_i in previous_fd_acm:
            acm_is_not_valid = np.logical_or(acm_is_not_valid, covariance_curr_vec_i == FLOAT_NODATA_VALUE)
        pixel_is_not_valid = np.logical_or(pixel_is_not_valid, acm_is_not_valid)

        # Convert FLOAT_NODATA_VALUE to NaN for the ACM, because it is used
        # during the computation
        previous_fd_acm[:, acm_is_not_valid] = np.nan

        # Initialize number of averages, setting to "1" the values over pixel_is_not_valid
        # because it is used during the computation
        # Note that INT_NODATA_VALUE is set after the computation
        number_of_averages = previous_fd_number_of_averages
        number_of_averages[pixel_is_not_valid] = 1  # INT_NODATA_VALUE is set after the FD computation

        # FD computation algorithm: matrix commputation without for cycles

        # First compute determinants
        det_x = det_mat_3x3_abs_angle(covariance_curr_vec_9x)  # this is det of "X"
        det_y = det_mat_3x3_abs_angle(previous_fd_acm)  # this is det of "Y"
        det_x_p_y = det_mat_3x3_sum_abs_angle(covariance_curr_vec_9x, previous_fd_acm)
        # numerical problems checked (as det_ are semi-definite positive matrices)
        det_x = np.where(det_x < fd_determinant_limit, fd_determinant_limit, det_x)
        det_y = np.where(det_y < fd_determinant_limit, fd_determinant_limit, det_y)
        det_x_p_y = np.where(det_x_p_y < fd_determinant_limit, fd_determinant_limit, det_x_p_y)
        # Compute the rho and omega2
        rho, omega2 = funs_rho_omega_wishart_SU(number_of_averages + 1, number_of_looks, num_pols)

        # Algorithm core, compute statistics
        test_statistic = (
            -2
            * rho
            * fun_ln_R_wishart_SU(
                det_y,
                det_x,
                det_x_p_y,
                number_of_averages + 1,
                number_of_looks,
                num_pols,
            )
        )
        test_statistic_is_not_positive = test_statistic < 0

        # compute the probability of no change
        probability_no_change = 1 - (
            chi2.cdf(test_statistic, num_pols**2) * (1 - omega2) + omega2 * chi2.cdf(test_statistic, num_pols**2 + 4)
        )

        # update the no valid mask, adding new no valid values coming from computation
        pixel_is_not_valid = np.logical_or(pixel_is_not_valid, np.isnan(probability_no_change))

        # compute all the optputs, from the probability of no change
        probability_of_change = (1 - probability_no_change).astype(np.float64)
        probability_of_change[test_statistic_is_not_positive] = 0
        probability_of_change[pixel_is_not_valid] = FLOAT_NODATA_VALUE

        fd_map = np.where(probability_no_change < significance_level_percent / 100, 1, 0).astype(np.uint8)
        fd_map[test_statistic_is_not_positive] = 0

        number_of_averages = np.where(
            probability_no_change < significance_level_percent / 100,
            1,
            number_of_averages + 1,
        ).astype(np.uint8)
        number_of_averages[pixel_is_not_valid] = INT_NODATA_VALUE

        acm_updated = np.where(
            probability_no_change < significance_level_percent / 100,
            covariance_curr_vec_9x,  # "X"
            previous_fd_acm + covariance_curr_vec_9x,  # "Y+X"
        ).astype(np.float64)
        # ACM is saved il LUT with the FLOAT_NODATA_VALUE
        acm_updated[np.repeat(pixel_is_not_valid[np.newaxis, :, :], 9, axis=0)] = FLOAT_NODATA_VALUE

    else:
        bps_logger.info("    algorithm is at first cycle (previous FD product not specified in job order)")
        bps_logger.info(
            f"    AUX PP2 2A significance level '{significance_level_percent}'% value ignored at first cycle"
        )
        # First cycle, no computations done, initializing outputs

        # Initialize FD to all no data values
        fd_map = (
            np.ones(
                (covariance_curr_vec_9x.shape[1], covariance_curr_vec_9x.shape[2]),
                dtype=np.uint8,
            )
            * INT_NODATA_VALUE
        )

        # Initialize probability of change to all no data values
        probability_of_change = (
            np.ones(
                (covariance_curr_vec_9x.shape[1], covariance_curr_vec_9x.shape[2]),
                dtype=np.float32,
            )
            * FLOAT_NODATA_VALUE
        )

        # Initialize number of averages, all ones, and no data values where input is not valid
        number_of_averages = np.ones((covariance_curr_vec_9x.shape[1], covariance_curr_vec_9x.shape[2])).astype(
            np.uint8
        )
        number_of_averages[pixel_is_not_valid] = INT_NODATA_VALUE

        # Initialize ACM: in this case it is the one in input, no update
        acm_updated = covariance_curr_vec_9x
        # ACM is saved il LUT with the FLOAT_NODATA_VALUE
        acm_updated[np.repeat(pixel_is_not_valid[np.newaxis, :, :], 9, axis=0)] = FLOAT_NODATA_VALUE

    # the FNF needs to be extracted from the whole one, over the Product latitude longitude vectors
    lut_fnf_interp_dict = interpolate_fnf(
        fnf.mask,
        fnf.lat_axis,
        fnf.lon_axis,
        forest_mask_interpolation_threshold,
        dgg_latitude_axis_deg,
        dgg_longitude_axis_deg,
    )

    if second_cycle_run:
        not_footprint_mask = acm_updated[0, :, :] == FLOAT_NODATA_VALUE
    else:
        not_footprint_mask = np.isnan(acm_updated[0, :, :])

    lut_fnf_interp_dict["fnf"][not_footprint_mask] = INT_NODATA_VALUE

    if second_cycle_run:
        bps_logger.info("    update ACM from previous cycle")
        bps_logger.info("    update CFM from previous cycle")

        cfm_no_valid_mask = previous_fd_cfm == INT_NODATA_VALUE
        cfm = np.logical_and(1 - fd_map, previous_fd_cfm).astype(np.uint8)

        cfm[cfm_no_valid_mask] = INT_NODATA_VALUE
        fd_map[pixel_is_not_valid] = INT_NODATA_VALUE
    else:
        bps_logger.info("    first cycle: setting CFM to external FNF")

        cfm = lut_fnf_interp_dict["fnf"]

    stop_time = datetime.now()
    elapsed_time = (stop_time - start_time).total_seconds()
    bps_logger.info(f"Change detection processing time: {elapsed_time:2.1f} s")
    return (
        fd_map,
        cfm,
        probability_of_change,
        acm_updated,
        number_of_averages,
        lut_fnf_interp_dict,
    )


def fun_ln_R_wishart_SU(
    det_y: np.ndarray,
    det_x: np.ndarray,
    det_x_p_y: np.ndarray,
    number_of_averages: np.ndarray,
    number_of_looks: int,
    num_pols: int | None = 3,
) -> float:
    """
    Compute the statistic "R" at current global cycle “N+1”
    The statistic output is a logaritm (ln) value.

    Parameters
    ----------
    det_y: np.ndarray
        determinant of the "Y" input ACM matrix, one value for each pixel
        Shape is [num_lat x num_lon]
    det_x: np.ndarray
        determinant of the "X" input CM matrix, one value for each pixel
        Shape is [num_lat x num_lon]
        Shape is [num_lat x num_lon]
    det_x_p_y: np.ndarray
        determinant of the sum "X" + "Y" matrices, one value for each pixel
        Shape is [num_lat x num_lon]
    number_of_averages: np.ndarray
        for each pixel from the number of avarages matrix,
        representing the number of times Y Average Covariange Matrix has been averaged
        (averages of previous "N" cycles)
        Shape is [num_lat x num_lon]
    number_of_looks. int
        number of looks
    num_pols: Optional[int]
         number of polarizations. Default nominal value is 3 (HH, XP, VV)

    Returns
    -------
    ln_R: np.ndarray
        natural logarithm of the "R" statistic, for each pixel
        Shape is [num_lat x num_lon]
    """

    ln_R_mat = np.real(
        number_of_looks
        * (
            num_pols
            * (
                (number_of_averages + 1) * np.log(number_of_averages + 1)
                - number_of_averages * np.log(number_of_averages)
            )
            + number_of_averages * np.log(det_y)
            + np.log(det_x)
            - (number_of_averages + 1) * np.log(det_x_p_y)
        )
    )

    return ln_R_mat


def funs_rho_omega_wishart_SU(
    number_of_averages: np.ndarray, number_of_looks: int, num_pols: int | None = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the variables "rho" and "omega" at current global cycle “N+1”

    Parameters
    ----------
    number_of_averages: np.ndarray
        for each pixel from the number of avarages matrix,
        representing the number of times Y Average Covariange Matrix has been averaged
        (averages of previous "N" cycles)
        Shape is [num_lat x num_lon]
    number_of_looks: int
         number of looks
    num_pols: Optional[int]
         number of polarizations. Default nominal value is 3 (HH, XP, VV)

    Returns
    -------
    rho: np.ndarray
        rho for each pixel, [n_lat x n_lon]
    omega2: np.ndarray
        omega2 for each pixel, [n_lat x n_lon]
    """

    rho = 1 - (2 * num_pols**2 - 1) * (1 + 1 / (number_of_averages * (number_of_averages + 1))) / (
        6 * num_pols * number_of_looks
    )

    omega2 = -(num_pols**2) * (1 - 1 / rho) ** 2 / 4 + num_pols**2 * (num_pols**2 - 1) * (
        1 + (2 * number_of_averages + 1) / ((number_of_averages + 1) ** 2 * number_of_averages**2)
    ) / (24 * number_of_looks**2 * rho**2)

    return rho, omega2


def det_mat_3x3_real_imag(x: np.ndarray) -> np.ndarray:
    """
    Compute determinant for each input matrix pixel
    In case x contains values in real, image format

    Input matrix values and indexing of each value:
    [ 0      real(1) imag(1)   real(2) imag(2)   4      real(5) imag(5)   8    ]
    [ idx0   idx1    idx2      idx3    idx4      idx5   idx6    idx7      idx8 ]

    Values to be inserted in this order:
           [ 0         1     2
             cj(1)     4     5
             cj(2)  cj(5)    8 ]

    how to index the input vector to obtain the result:
           [ idx0               idx1+j*idx2        idx3+j*idx4
             idx1-j*idx2        idx5               idx6+j*idx7
             idx3-j*idx4        idx6-j*idx7        idx8        ]

    Parameters
    ----------
    x: np.ndarray
        input covariance with shape [9 x num_lat x num_lon]
        x contains values in real, image format

    Returns
    -------
    d: np.ndarray
        determinant for each of the [num_lat x num_lon] pixel
        Shape is [num_lat x num_lon]
    """

    d = (
        (x[0, ...] * x[5, ...] * x[8, ...])
        + ((x[1, ...] + 1j * x[2, ...]) * (x[6, ...] + 1j * x[7, ...]) * (x[3, ...] - 1j * x[4, ...]))
        + ((x[1, ...] - 1j * x[2, ...]) * (x[6, ...] - 1j * x[7, ...]) * (x[3, ...] + 1j * x[4, ...]))
        - ((x[3, ...] ** 2 + x[4, ...] ** 2) * x[5, ...])
        - ((x[1, ...] ** 2 + x[2, ...] ** 2) * x[8, ...])
        - ((x[6, ...] ** 2 + x[7, ...] ** 2) * x[0, ...])
    )

    return np.real(d)


def det_mat_3x3_abs_angle(x: np.ndarray) -> np.ndarray:
    """
    Compute determinant for each input matrix pixel
    In case x contains values in abs, angle format

    Input matrix values and indexing of each value:
    [ 0      abs(1)  phase(1)  abs(2)  phase(2)  4      abs(5)  phase(5)  8    ]
    [ idx0   idx1    idx2      idx3    idx4      idx5   idx6    idx7      idx8 ]

    Values to be inserted in this order:
           [ 0         1     2
             cj(1)     4     5
             cj(2)  cj(5)    8 ]

    how to index the input vector to obtain the result:
           [ idx0               idx1*exp(j idx2)    idx3*exp(j idx4)
             idx1*exp(-j idx2)      idx5            idx6*exp(j idx7)
             idx3*exp(-j idx4)  idx6*exp(-j idx7)               idx8 ]

    Parameters
    ----------
    x: np.ndarray
        input covariance with shape [9 x num_lat x num_lon]
        x contains values in abs, angle format

    Returns
    -------
    d: np.ndarray
        determinant for each of the [num_lat x num_lon] pixel
        Shape is [num_lat x num_lon]
    """

    d = (
        (x[0, ...] * x[5, ...] * x[8, ...])
        + (
            (x[1, ...] * x[6, ...] * x[3, ...])
            * (np.exp(1j * (x[2, ...] - x[4, ...] + x[7, ...])) + np.exp(-1j * (x[2, ...] - x[4, ...] + x[7, ...])))
        )
        - (x[5, ...] * x[3, ...] * x[3, ...])
        - (x[1, ...] * x[1, ...] * x[8, ...])
        - (x[0, ...] * x[6, ...] * x[6, ...])
    )

    return np.real(d)


def det_mat_3x3_sum_abs_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute determinant for the sum of each input matrix pixel
    In case x and y contains values in abs, angle format

    Input matrix values and indexing of each value:
    [ 0      abs(1)  phase(1)  abs(2)  phase(2)  4      abs(5)  phase(5)  8    ]
    [ idx0   idx1    idx2      idx3    idx4      idx5   idx6    idx7      idx8 ]

    Values to be inserted in this order:
           [ 0         1     2
             cj(1)     4     5
             cj(2)  cj(5)    8 ]

    how to index the input vector to obtain the result:
           [ idx0               idx1*exp(j idx2)    idx3*exp(j idx4)
             idx1*exp(-j idx2)      idx5            idx6*exp(j idx7)
             idx3*exp(-j idx4)  idx6*exp(-j idx7)               idx8 ]

    Parameters
    ----------
    x: np.ndarray
        input covariance with shape [9 x num_lat x num_lon]
        x contains values in abs, angle format
    y: np.ndarray
        input covariance with shape [9 x num_lat x num_lon]
        y contains values in abs, angle format

    Returns
    -------
    d: np.ndarray
        determinant for each of the [num_lat x num_lon] pixel
        Shape is [num_lat x num_lon]
    """

    # First build summed matrix

    XpY = np.zeros_like(x)

    XpY[0, ...] = x[0, ...] + y[0, ...]

    temp = x[1, ...] * np.exp(1j * x[2, ...]) + y[1, ...] * np.exp(1j * y[2, ...])
    XpY[1, ...] = np.real(temp)
    XpY[2, ...] = np.imag(temp)

    temp = x[3, ...] * np.exp(1j * x[4, ...]) + y[3, ...] * np.exp(1j * y[4, ...])
    XpY[3, ...] = np.real(temp)
    XpY[4, ...] = np.imag(temp)

    XpY[5, ...] = x[5, ...] + y[5, ...]

    temp = x[6, ...] * np.exp(1j * x[7, ...]) + y[6, ...] * np.exp(1j * y[7, ...])
    XpY[6, ...] = np.real(temp)
    XpY[7, ...] = np.imag(temp)

    XpY[8, ...] = x[8, ...] + y[8, ...]

    return det_mat_3x3_real_imag(XpY)
