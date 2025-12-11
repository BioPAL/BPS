# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2B FD commands
---------------
"""

from datetime import datetime
from pathlib import Path

import bps.l2b_fd_processor
import numba as nb
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common_types
from bps.l2b_fd_processor.core.aux_pp2_2b_fd import AuxProcessingParametersL2BFD
from bps.l2b_fd_processor.core.joborder_l2b_fd import L2bFDJobOrder
from bps.l2b_fd_processor.core.translate_job_order import L2B_OUTPUT_PRODUCT_FD
from bps.l2b_fd_processor.fd import BPS_L2B_FD_PROCESSOR_NAME
from bps.l2b_fd_processor.l2b_common_functionalities import (
    compute_l2a_contributing_heat_map,
    dgg_tiling,
)
from bps.transcoder.io import common_annotation_models_l2, main_annotation_models_l2b_fd
from bps.transcoder.sarproduct.biomass_l2aproduct import BIOMASSL2aProductFD
from bps.transcoder.sarproduct.biomass_l2bfdproduct import (
    BIOMASSL2bFDMainADSInputInformation,
    BIOMASSL2bFDMainADSproduct,
    BIOMASSL2bFDMainADSRasterImage,
    BIOMASSL2bFDProduct,
    BIOMASSL2bFDProductMeasurement,
    BIOMASSL2bMainADSProcessingParametersFD,
)
from bps.transcoder.sarproduct.biomass_l2bfdproduct_writer import (
    AVERAGING_FACTOR_QUICKLOOKS,
    COMPRESSION_EXIF_CODES_LERC_ZSTD,  # LERC, ZSTD
    DECIMATION_FACTOR_QUICKLOOKS,
    FLOAT_NODATA_VALUE,
    INT_NODATA_VALUE,
    BIOMASSL2bFDProductWriter,
)
from bps.transcoder.sarproduct.l2_annotations import COORDINATE_REFERENCE_SYSTEM, ground_corner_points


class FDL2B:
    """Forest Disturbance L2b Processor"""

    def __init__(
        self,
        job_order: L2bFDJobOrder,
        aux_pp2_fd: AuxProcessingParametersL2BFD,
        working_dir: Path,
        l2a_fd_products_list: list[BIOMASSL2aProductFD],
    ) -> None:
        """
        Parameters
        ----------
        job_order  = L2bFDJobOrder
            content of the job order XML file
        aux_pp2_fd  = AuxProcessingParametersL2BFD
            content of the AUX PP2 FD XML file
        working_dir  = Path
            working directory
        l2a_fd_products_list  = List[BIOMASSL2aProductFD]
            list of all the L2a FD products paths
        """

        self.job_order = job_order
        self.aux_pp2_fd = aux_pp2_fd
        self.working_dir = working_dir
        self.l2a_fd_products_list = l2a_fd_products_list
        self.output_baseline = 0

    def _get_l2a_inputs_information(self):
        # Precompute here the info to construct main_ads_input_information_l2b:

        mission_phase_id = None
        global_coverage_id = None
        l2a_inputs_list = []
        basin_id_list = []
        for idx, l2a_product in enumerate(self.l2a_fd_products_list):
            basin_id_list = basin_id_list + l2a_product.main_ads_product.basin_id_list

            footprint = main_annotation_models_l2b_fd.FloatArrayWithUnits(
                value=l2a_product.main_ads_input_information.footprint,
                count=8,
                units=common_types.UomType.DEG,
            )

            l1_inputs = common_annotation_models_l2.InputInformationL2AType(
                product_type=common_annotation_models_l2.ProductType(
                    l2a_product.main_ads_input_information.product_type
                ),
                overall_products_quality_index=l2a_product.main_ads_input_information.overall_products_quality_index,
                nominal_stack=str(l2a_product.main_ads_input_information.nominal_stack).lower(),
                polarisation_list=l2a_product.main_ads_input_information.polarisation_list,
                projection=common_annotation_models_l2.ProjectionType(
                    l2a_product.main_ads_input_information.projection
                ),
                footprint=footprint,
                vertical_wavenumbers=l2a_product.main_ads_input_information.vertical_wavenumbers,
                height_of_ambiguity=l2a_product.main_ads_input_information.height_of_ambiguity,
                acquisition_list=l2a_product.main_ads_input_information.acquisition_list,
            )

            l2a_inputs_list.append(
                common_annotation_models_l2.InputInformationL2BL3ListType.L2AInputs(
                    l2a_product_folder_name=str(self.job_order.input_l2a_products[idx]),
                    l2a_product_date=l2a_product.main_ads_product.start_time.isoformat(timespec="microseconds"),
                    l1_inputs=l1_inputs,
                    significance_level=l2a_product.main_ads_processing_parameters.significance_level,
                )
            )

            if idx == 0:
                mission_phase_id = l2a_product.main_ads_product.mission_phase_id
                global_coverage_id = l2a_product.main_ads_product.global_coverage_id
            else:
                if not mission_phase_id == l2a_product.main_ads_product.mission_phase_id:
                    raise ValueError(
                        "Input L2a products should be of the same phase: found some INT and some TOM phase L2a products."
                    )

        return (
            l2a_inputs_list,
            mission_phase_id,
            global_coverage_id,
            list(np.unique(basin_id_list)),
        )

    def _initialize_processing(self):
        """Initialize the FD L2b processing"""

        self.processing_start_time = datetime.now()
        self.start_time = PreciseDateTime.now()
        bps_logger.info("%s started", BPS_L2B_FD_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baseline is not None:
            if self.job_order.output_product == L2B_OUTPUT_PRODUCT_FD:
                self.output_baseline = self.job_order.output_baseline

        self.product_type = L2B_OUTPUT_PRODUCT_FD

    def _core_processing(self):
        """Execute core FD L2b processing"""

        bps_logger.info("L2a products fusion into L2b")
        bps_logger.info(
            f"        checking with AUX PP2 minimum L2a products coverage of {self.aux_pp2_fd.minimumL2acoverage}%"
        )
        # Fusion: dgg tiling > temporal sorting > derivative > fd steps count > consistency check

        # dgg tiling: put each L2A into L2B Tile:
        # note: input data can have INT or FLOAT no data values,
        # this function converts them to np.nan
        (
            skip_fd_computation,
            data_3d_mat_dict,
            dgg_tile_latitude_axis,
            dgg_tile_longitude_axis,
            dgg_tile_footprint,
        ) = dgg_tiling(
            self.l2a_fd_products_list,
            self.aux_pp2_fd.minimumL2acoverage,
            self.job_order.processing_parameters.tile_id,
            L2B_OUTPUT_PRODUCT_FD,
        )

        if not skip_fd_computation:
            bps_logger.info("   Fusion algorithm:")

            # initializing output l2b dictionary
            l2b_data_final_d = {
                "dgg_latitude_axis": dgg_tile_latitude_axis.astype(np.float32),
                "dgg_longitude_axis": dgg_tile_longitude_axis.astype(np.float32),
            }

            bps_logger.info("       compute aggregated FD, probability and heat maps")
            # compute all the aggregated matrices
            l2b_data_final_d["heat_map"] = {}
            (
                l2b_data_final_d["fd"],
                l2b_data_final_d["probability_ofchange"],
                l2b_data_final_d["heat_map"]["heat_map_contributing"],
                l2b_data_final_d["heat_map"]["heat_map_agreeing"],
            ) = find_aggregarted_fd(data_3d_mat_dict["fd"], data_3d_mat_dict["probability_ofchange"])
            l2b_data_final_d["probability_ofchange"] = l2b_data_final_d["probability_ofchange"].astype(np.float32)

            # update CFM
            if np.sum(np.isnan(data_3d_mat_dict["fd"])) == data_3d_mat_dict["fd"].size:
                # If input FD is all no data values, it is surely an input L2B FD generated at first cycle
                bps_logger.info("       First Cycle: setting CFM to external FNF")
                l2b_data_final_d["cfm"] = data_3d_mat_dict["cfm"][:, :, 0]
                l2b_data_final_d["cfm"][np.isnan(l2b_data_final_d["cfm"])] = INT_NODATA_VALUE
                l2b_data_final_d["cfm"] = l2b_data_final_d["cfm"].astype(np.uint8)
            else:
                bps_logger.info("       Update CFM")
                l2b_data_final_d["cfm"] = update_cfm(data_3d_mat_dict["cfm"], l2b_data_final_d["fd"])

            # Compute additional heat map:
            l2b_data_final_d["acquisition_id_image"] = compute_l2a_contributing_heat_map(data_3d_mat_dict, "fd")

            # Construct main_ads_input_information_l2b, for the L2b FD writer:
            (
                l2a_inputs_list,
                mission_phase_id,
                global_coverage_id,
                self.basin_id_list,
            ) = self._get_l2a_inputs_information()

            main_ads_input_information_l2b = BIOMASSL2bFDMainADSInputInformation(
                common_annotation_models_l2.InputInformationL2BL3ListType(
                    l2a_inputs=l2a_inputs_list,
                    count=len(l2a_inputs_list),
                ),
            )

        else:
            l2b_data_final_d = None
            mission_phase_id = None
            global_coverage_id = None
            dgg_tile_footprint = None
            main_ads_input_information_l2b = None
            bps_logger.info("L2b FD processor: nothing to compute, exiting.")

        start_time_l2a = self.l2a_fd_products_list[0].main_ads_product.start_time
        stop_time_l2a = self.l2a_fd_products_list[0].main_ads_product.stop_time
        for l2a_product in self.l2a_fd_products_list:
            start_time_l2a = min(start_time_l2a, l2a_product.main_ads_product.start_time)
            stop_time_l2a = max(stop_time_l2a, l2a_product.main_ads_product.stop_time)

        # Footprint mask for quick looks transparency
        footprint_masks_for_quicklooks = {}
        if AVERAGING_FACTOR_QUICKLOOKS > 1:
            fd_for_footprint = data_3d_mat_dict["fd"].squeeze()[
                ::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS
            ]
        else:
            fd_for_footprint = data_3d_mat_dict["dh"].squeeze()
        footprint_masks_for_quicklooks["data_mask"] = np.logical_not(np.isnan(fd_for_footprint))

        # Footprint mask for quick looks transparency
        cfm_no_data_value_mask = data_3d_mat_dict["cfm"].squeeze() == INT_NODATA_VALUE
        cfm_for_footprint = data_3d_mat_dict["cfm"].squeeze().astype(np.float32)
        cfm_for_footprint[cfm_no_data_value_mask] = np.nan
        if AVERAGING_FACTOR_QUICKLOOKS > 1:
            cfm_for_footprint = cfm_for_footprint[::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS]
        footprint_masks_for_quicklooks["cfm_mask"] = np.logical_not(np.isnan(cfm_for_footprint))

        return (
            l2b_data_final_d,
            self.job_order.processing_parameters.tile_id,
            mission_phase_id,
            global_coverage_id,
            dgg_tile_footprint,
            main_ads_input_information_l2b,
            skip_fd_computation,
            start_time_l2a,
            stop_time_l2a,
            footprint_masks_for_quicklooks,
        )

    def _fill_product_for_writing(self, tile_id):
        # TEMP = Fill needed fields with dummy values:
        # fixed values (i.e. "mission=BIOMASS") are directly set in _write_to_output function

        meta_data_d = {}  # one metadata for each tile, and one for each product (fd, cfm, probability, heat_map)

        # fill common fileds for FD, CFM and Number of Averages
        meta_data_temp = [
            BIOMASSL2bFDProductMeasurement.MetadataCOG(
                tile_id_list=[tile_id],
                basin_id_list=self.basin_id_list,
                compression=COMPRESSION_EXIF_CODES_LERC_ZSTD,  #  [LERC, ZSTD]
                image_description="",
                software="",
                dateTime=self.start_time.isoformat(timespec="microseconds")[:-1],
            )
            for idx in range(5)
        ]

        # specific fileds
        meta_data_d = {
            "fd": meta_data_temp[0],
            "probability_ofchange": meta_data_temp[1],
            "cfm": meta_data_temp[2],
            "heat_map": meta_data_temp[3],
            "acquisition_id_image": meta_data_temp[4],
        }
        meta_data_d["fd"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_DISTURBANCE.value
        )
        meta_data_d["fd"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data_d["probability_ofchange"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.PROBABILITY_OF_CHANGE.value
        )
        meta_data_d["cfm"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.COMPUTED_FOREST_MASK.value
        )
        meta_data_d["cfm"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data_d["heat_map"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.HEAT_MAP.value
        )
        meta_data_d["heat_map"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data_d["acquisition_id_image"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + "acquisition id image"
        )
        meta_data_d["acquisition_id_image"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        return meta_data_d

    def _write_to_output(
        self,
        tiled_data_d,
        tile_id,
        mission_phase_id,
        global_coverage_id,
        footprint,
        main_ads_input_information,
        start_time_l2a,
        stop_time_l2a,
        footprint_masks_for_quicklooks,
    ):
        """Write output FD L2b products"""

        radar_carrier_frequency = 435000000.0

        dgg_tile_latitude_axis = tiled_data_d["dgg_latitude_axis"]
        dgg_tile_longitude_axis = tiled_data_d["dgg_longitude_axis"]

        metadata_d = self._fill_product_for_writing(tile_id)

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
        measurement = BIOMASSL2bFDProductMeasurement(
            dgg_tile_latitude_axis,
            dgg_tile_longitude_axis,
            tiled_data_d,
            metadata_d,
        )

        # main_ads_product
        mission = common_annotation_models_l2.MissionType.BIOMASS.value
        sensor_mode = common_annotation_models_l2.SensorModeType.MEASUREMENT.value
        main_ads_product = BIOMASSL2bFDMainADSproduct(
            mission,
            [tile_id],
            self.basin_id_list,
            self.product_type,
            self.start_time,
            self.stop_time,
            radar_carrier_frequency,
            mission_phase_id,
            sensor_mode,
            global_coverage_id,
            self.output_baseline,
        )

        # main_ads_raster_image
        projection = common_annotation_models_l2.ProjectionType.DGG.value
        coordinate_reference_system = COORDINATE_REFERENCE_SYSTEM
        geodetic_reference_frame = common_annotation_models_l2.GeodeticReferenceFrameType.WGS84.value
        datum = common_annotation_models_l2.DatumType(
            coordinate_reference_system,
            common_annotation_models_l2.GeodeticReferenceFrameType(geodetic_reference_frame),
        )
        pixel_representation_d = {
            "fd": common_types.PixelRepresentationType.FOREST_DISTURBANCE,
            "probability_of_change": common_types.PixelRepresentationType.PROBABILITY_OF_CHANGE,
            "cfm": common_types.PixelRepresentationType.COMPUTED_FOREST_MASK,
            "heat_map": [common_types.PixelRepresentationType.HEAT_MAP],
            "acquisition_id_image": common_types.PixelRepresentationType.ACQUISITION_ID_IMAGE,
        }
        pixel_representation = common_annotation_models_l2.PixelRepresentationChoiceType(
            fd=pixel_representation_d["fd"],
            cfm=pixel_representation_d["cfm"],
            probability_of_change=pixel_representation_d["probability_of_change"],
            fd_heat_map=pixel_representation_d["heat_map"],
            acquisition_id_image=pixel_representation_d["acquisition_id_image"],
        )
        pixel_type = common_annotation_models_l2.PixelTypeChoiceType(
            float_pixel_type=main_annotation_models_l2b_fd.PixelTypeType("32 bit Float"),
            int_pixel_type=main_annotation_models_l2b_fd.PixelTypeType("8 bit Unsigned Integer"),
        )

        no_data_value = common_annotation_models_l2.NoDataValueChoiceType(
            float_no_data_value=FLOAT_NODATA_VALUE, int_no_data_value=INT_NODATA_VALUE
        )
        main_ads_raster_image = BIOMASSL2bFDMainADSRasterImage(
            footprint,
            dgg_tile_latitude_axis[0],
            dgg_tile_longitude_axis[0],
            dgg_tile_latitude_axis[1] - dgg_tile_latitude_axis[0],
            dgg_tile_longitude_axis[1] - dgg_tile_longitude_axis[0],
            len(dgg_tile_latitude_axis),
            len(dgg_tile_longitude_axis),
            projection,
            datum,
            pixel_representation,
            pixel_type,
            no_data_value,
        )

        # main_ads_processing_parameters
        compression_options_fd = main_annotation_models_l2b_fd.CompressionOptionsL2B(
            main_annotation_models_l2b_fd.CompressionOptionsL2B.Mds(
                main_annotation_models_l2b_fd.CompressionOptionsL2B.Mds.Fd(
                    self.aux_pp2_fd.compression_options.mds.fd.compression_factor
                ),
                main_annotation_models_l2b_fd.CompressionOptionsL2B.Mds.ProbabilityOfChange(
                    self.aux_pp2_fd.compression_options.mds.probability_of_change.compression_factor,
                    self.aux_pp2_fd.compression_options.mds.probability_of_change.max_z_error,
                ),
                main_annotation_models_l2b_fd.CompressionOptionsL2B.Mds.Cfm(
                    self.aux_pp2_fd.compression_options.mds.cfm.compression_factor
                ),
                main_annotation_models_l2b_fd.CompressionOptionsL2B.Mds.HeatMap(
                    self.aux_pp2_fd.compression_options.mds.heatmap.compression_factor
                ),
                main_annotation_models_l2b_fd.CompressionOptionsL2B.Mds.AcquisitionIdImage(
                    self.aux_pp2_fd.compression_options.mds.acquisition_id_image.compression_factor
                ),
            ),
            self.aux_pp2_fd.compression_options.mds_block_size,
        )

        main_ads_processing_parameters = BIOMASSL2bMainADSProcessingParametersFD(
            bps.l2b_fd_processor.__version__,
            self.start_time,
            compression_options_fd,
            minumum_l2a_coverage=self.aux_pp2_fd.minimumL2acoverage,
        )

        # Initialize FD Product
        product_to_write = BIOMASSL2bFDProduct(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            self.aux_pp2_fd.l2bFDProductDOI,
        )

        # Write to file the FD Product
        write_obj = BIOMASSL2bFDProductWriter(
            product_to_write,
            self.product_path,
            bps.l2b_fd_processor.BPS_L2B_FD_PROCESSOR_NAME,
            bps.l2b_fd_processor.__version__,
            footprint,
            [acq.name for acq in self.job_order.input_l2a_products],
            ground_corner_points(dgg_tile_latitude_axis, dgg_tile_longitude_axis),
            self.job_order.aux_pp2_fd_path.name,
            start_time_l2a,
            stop_time_l2a,
            footprint_masks_for_quicklooks,
        )
        write_obj.write()

    def run_l2b_fd_processing(self):
        """Performs processing as described in job order.

        Parameters
        ----------

        Returns
        -------
        """

        self._initialize_processing()

        (
            l2b_data_final_d,
            tile_id,
            mission_phase_id,
            global_coverage_id,
            footprint,
            main_ads_input_information,
            skip_fd_computation,
            start_time_l2a,
            stop_time_l2a,
            footprint_masks_for_quicklooks,
        ) = self._core_processing()

        if not skip_fd_computation:
            assert l2b_data_final_d is not None

            self._write_to_output(
                l2b_data_final_d,
                tile_id,
                mission_phase_id,
                global_coverage_id,
                footprint,
                main_ads_input_information,
                start_time_l2a,
                stop_time_l2a,
                footprint_masks_for_quicklooks,
            )


def update_cfm(cfm_mat_input: np.ndarray, fd_aggregated: np.ndarray) -> np.ndarray:
    """Uptate CFM

    Parameters
    ----------
    cfm_mat_input: np.ndarray
        Input CFM for each l2a, is a 3D matrix of shape [num_lat_dgg_tile, num_lon_dgg_tile, num_l2a_products]
        Note: it is casted to float32 data type
    fd_aggregated: np.ndarray
        FD aggregated from the input FD, of shape [num_lat_dgg_tile, num_lon_dgg_tile]
        This is used to check where the fd aggregated agrees with the input fd_mat_sorted
        Note: it is uint8 data type

    Returns
    -------
    cfm_aggregated: np.ndarray
        Updated CFM, aggregation of the input ones
        shape [num_lat_dgg_tile, num_lon_dgg_tile]
        Note: it is uint8 data type
    """
    # compute nan mask
    nan_mask = fd_aggregated == INT_NODATA_VALUE

    # cast fd to float as the cfm
    fd_aggregated = np.float64(fd_aggregated)
    fd_aggregated[nan_mask] = np.nan

    # compute the uinion of L2a inputs CFM masks
    # "sum" equals to "or" (union) operator
    union_cfm = np.nansum(cfm_mat_input, axis=2)
    union_cfm[union_cfm > 1] = 1.0

    # Update the CFM, than re cast to uint8 and replace INT no data values
    cfm_aggregated = np.logical_and(1 - fd_aggregated, union_cfm).astype(np.uint8)

    cfm_aggregated[nan_mask] = INT_NODATA_VALUE

    return cfm_aggregated


@nb.njit(nogil=True, cache=True)
def is_odd(value):
    return int(np.rint(value)) % 2 != 0


@nb.njit(nogil=True, cache=True)
def all_is_changed(values):
    return int(np.rint(np.sum(values))) == values.size


@nb.njit(nogil=True, cache=True, parallel=True)
def find_aggregarted_fd_core(
    fd_mat_sorted: np.ndarray,
    proability_mat: np.ndarray,
    hm_contributing: np.ndarray,
    hm_agreeing: np.ndarray,
    fd_aggregated: np.ndarray,
    probability_aggregated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """find aggregated fd core
    Core functon which aggregates many L2a data in one l2b
    The function computes aggregated FD, Probability of change
    and the two heat maps
    """
    # cycle for each pixel (use numba for speed up the process)
    for lat_idx in nb.prange(fd_mat_sorted.shape[0]):
        for lon_idx in nb.prange(fd_mat_sorted.shape[1]):
            # one fd and one probability value for each l2a input
            fd = fd_mat_sorted[lat_idx, lon_idx, :]
            probability = proability_mat[lat_idx, lon_idx, :]

            # remove nan values from the vectors
            nan_mask = np.logical_not(np.isnan(fd))
            fd = fd[nan_mask]
            probability = probability[nan_mask]

            # computation of fd, probability and hm aggregated
            if fd.size == 0:
                # all nan in input fd, no contributuons
                hm_contributing[lat_idx, lon_idx] = 0
                hm_agreeing[lat_idx, lon_idx] = 0
                fd_aggregated[lat_idx, lon_idx] = INT_NODATA_VALUE
                probability_aggregated[lat_idx, lon_idx] = FLOAT_NODATA_VALUE

            elif fd.size == 1:
                # just one valid value in fd, no computation needed
                hm_contributing[lat_idx, lon_idx] = 1
                hm_agreeing[lat_idx, lon_idx] = 1
                fd_aggregated[lat_idx, lon_idx] = fd[0]
                probability_aggregated[lat_idx, lon_idx] = probability[0]

            else:
                # standard computation in case of more than one fd valid value
                transition = np.sum(np.abs(np.diff(fd)))

                # FD assumed:
                # 1) if count is odd
                # 2) if the sum is zero, detection is assumed only if all the acquisitions show detection
                if is_odd(transition) or all_is_changed(fd):
                    # FD assumed
                    fd_aggregated[lat_idx, lon_idx] = 1
                    probability_aggregated[lat_idx, lon_idx] = np.mean(probability[fd > 0])
                else:
                    # FD not assumed
                    fd_aggregated[lat_idx, lon_idx] = 0
                    probability_aggregated[lat_idx, lon_idx] = 0.0
                    hm_agreeing[lat_idx, lon_idx] = fd.size - hm_agreeing[lat_idx, lon_idx]

    return fd_aggregated, probability_aggregated, hm_contributing, hm_agreeing


def find_aggregarted_fd_core_serial(
    fd_mat_sorted: np.ndarray,
    proability_mat: np.ndarray,
    hm_contributing: np.ndarray,
    hm_agreeing: np.ndarray,
    fd_aggregated: np.ndarray,
    probability_aggregated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """find aggregated fd core
    Core functon which aggregates many L2a data in one l2b
    The function computes aggregated FD, Probability of change
    and the two heat maps
    """
    # cycle for each pixel (use numba for speed up the process)
    for lat_idx in nb.prange(fd_mat_sorted.shape[0]):
        for lon_idx in nb.prange(fd_mat_sorted.shape[1]):
            # one fd and one probability value for each l2a input
            fd = fd_mat_sorted[lat_idx, lon_idx, :]
            probability = proability_mat[lat_idx, lon_idx, :]

            # remove nan values from the vectors
            nan_mask = np.logical_not(np.isnan(fd))
            fd = fd[nan_mask]
            probability = probability[nan_mask]

            # computation of fd, probability and hm aggregated
            if fd.size == 0:
                # all nan in input fd, no contributuons
                hm_contributing[lat_idx, lon_idx] = 0
                hm_agreeing[lat_idx, lon_idx] = 0
                fd_aggregated[lat_idx, lon_idx] = INT_NODATA_VALUE
                probability_aggregated[lat_idx, lon_idx] = FLOAT_NODATA_VALUE

            elif fd.size == 1:
                # just one valid value in fd, no computation needed
                hm_contributing[lat_idx, lon_idx] = 1
                hm_agreeing[lat_idx, lon_idx] = 1
                fd_aggregated[lat_idx, lon_idx] = fd[0]
                probability_aggregated[lat_idx, lon_idx] = probability[0]

            else:
                # standard computation in case of more than one fd valid value
                transition = np.sum(np.abs(np.diff(fd)))

                # FD assumed:
                # 1) if count is odd
                # 2) if the sum is zero, detection is assumed only if all the acquisitions show detection
                if is_odd(transition) or all_is_changed(fd):
                    # FD assumed
                    fd_aggregated[lat_idx, lon_idx] = 1
                    probability_aggregated[lat_idx, lon_idx] = np.mean(probability[fd > 0])
                else:
                    # FD not assumed
                    fd_aggregated[lat_idx, lon_idx] = 0
                    probability_aggregated[lat_idx, lon_idx] = 0.0
                    hm_agreeing[lat_idx, lon_idx] = fd.size - hm_agreeing[lat_idx, lon_idx]

    return fd_aggregated, probability_aggregated, hm_contributing, hm_agreeing


def find_aggregarted_fd(
    fd_mat_sorted: np.ndarray, proability_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """find aggregated fd
    Aggregates many L2a data in one l2b
    The function computes aggregated FD, Probability of change
    and the two heat maps

    Parameters
    ----------
    fd_mat_sorted: np.ndarray
        Input FD for each l2a, temporal sorted from older to newer,
        is a 3D matrix of shape [num_lat_dgg_tile, num_lon_dgg_tile, num_l2a_products]
    proability_mat: np.ndarray
        Input probability of change for each l2a,
        is a 3D matrix of shape [num_lat_dgg_tile, num_lon_dgg_tile, num_l2a_products]

    Returns
    -------
    fd_aggregated: np.ndarray
        shape [num_lat_dgg_tile, num_lon_dgg_tile]
    probability_aggregated: np.ndarray
        shape [num_lat_dgg_tile, num_lon_dgg_tile]
    hm_contributing: np.ndarray
        shape [num_lat_dgg_tile, num_lon_dgg_tile]
    hm_agreeing: np.ndarray
        shape [num_lat_dgg_tile, num_lon_dgg_tile]
    """

    # Inptialize optupt matrices
    fd_aggregated = np.zeros((fd_mat_sorted.shape[0], fd_mat_sorted.shape[1]), dtype=np.uint8)
    probability_aggregated = np.zeros((fd_mat_sorted.shape[0], fd_mat_sorted.shape[1]), dtype=np.float64)
    hm_contributing = np.count_nonzero(~np.isnan(fd_mat_sorted), axis=2).astype(np.uint8)
    hm_agreeing = np.rint(np.nansum(fd_mat_sorted, axis=2)).astype(np.uint8)

    # run the core function
    (
        fd_aggregated,
        probability_aggregated,
        hm_contributing,
        hm_agreeing,
    ) = find_aggregarted_fd_core(
        fd_mat_sorted,
        proability_mat,
        hm_contributing,
        hm_agreeing,
        fd_aggregated,
        probability_aggregated,
    )

    return fd_aggregated, probability_aggregated, hm_contributing, hm_agreeing
