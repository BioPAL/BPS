# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2B FH commands
---------------
"""

from datetime import datetime
from pathlib import Path

import bps.l2b_fh_processor
import numba as nb
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common_types
from bps.l2b_fh_processor.core.aux_pp2_2b_fh import AuxProcessingParametersL2BFH
from bps.l2b_fh_processor.core.joborder_l2b_fh import L2bFHJobOrder
from bps.l2b_fh_processor.core.translate_job_order import (
    L2B_OUTPUT_PRODUCT_FH,
    L2B_OUTPUT_PRODUCT_TFH,
)
from bps.l2b_fh_processor.fh import BPS_L2B_FH_PROCESSOR_NAME
from bps.l2b_fh_processor.l2b_common_functionalities import (
    compute_l2a_contributing_heat_map,
    dgg_tiling,
    sort_footprints,
)
from bps.transcoder.io import common_annotation_models_l2, main_annotation_models_l2b_fh
from bps.transcoder.sarproduct.biomass_l2aproduct import BIOMASSL2aProductFH
from bps.transcoder.sarproduct.biomass_l2bfdproduct import BIOMASSL2bFDProduct
from bps.transcoder.sarproduct.biomass_l2bfhproduct import (
    BIOMASSL2bFHMainADSInputInformation,
    BIOMASSL2bFHMainADSproduct,
    BIOMASSL2bFHMainADSRasterImage,
    BIOMASSL2bFHProduct,
    BIOMASSL2bFHProductMeasurement,
    BIOMASSL2bMainADSProcessingParametersFH,
)
from bps.transcoder.sarproduct.biomass_l2bfhproduct_writer import (
    AVERAGING_FACTOR_QUICKLOOKS,
    COMPRESSION_EXIF_CODES_LERC_ZSTD,  # LERC, ZSTD
    DECIMATION_FACTOR_QUICKLOOKS,
    FLOAT_NODATA_VALUE,
    INT_NODATA_VALUE,
    BIOMASSL2bFHProductWriter,
)
from bps.transcoder.sarproduct.l2_annotations import COORDINATE_REFERENCE_SYSTEM, ground_corner_points


class FHL2B:
    """Forest Height L2b Processor"""

    def __init__(
        self,
        job_order: L2bFHJobOrder,
        aux_pp2_fh: AuxProcessingParametersL2BFH,
        working_dir: Path,
        l2a_fh_products_list: list[BIOMASSL2aProductFH],
        l2b_fd_product: BIOMASSL2bFDProduct | None = None,
    ) -> None:
        """
        Parameters
        ----------
        job_order  = L2bFHJobOrder
            content of the job order XML file
        aux_pp2_fh  = AuxProcessingParametersL2BFH
            content of the AUX PP2 FH XML file
        working_dir  = Path
            working directory
        l2a_fh_products_list  = List[BIOMASSL2aProductFH]
            list of all the L2a FH products paths
        l2b_fd_product = BIOMASSL2aProductFD
            Optional, input L2a FD products, one for each tile present in the L2a fh products
        """

        self.job_order = job_order
        self.aux_pp2_fh = aux_pp2_fh
        self.working_dir = working_dir
        self.l2a_fh_products_list = l2a_fh_products_list
        self.l2b_fd_product = l2b_fd_product
        self.output_baseline = 0

    def _get_l2a_inputs_information(self):
        # Precompute here the info to construct main_ads_input_information_l2b:

        mission_phase_id = None
        global_coverage_id = None
        l2a_inputs_list = []
        basin_id_list = []
        for idx, l2a_product in enumerate(self.l2a_fh_products_list):
            basin_id_list = basin_id_list + l2a_product.main_ads_product.basin_id_list

            footprint = main_annotation_models_l2b_fh.FloatArrayWithUnits(
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
        """Initialize the FH L2b processing"""

        self.processing_start_time = datetime.now()
        self.start_time = PreciseDateTime.now()
        bps_logger.info("%s started", BPS_L2B_FH_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baseline is not None:
            if self.job_order.output_product in [
                L2B_OUTPUT_PRODUCT_FH,
                L2B_OUTPUT_PRODUCT_TFH,
            ]:
                self.output_baseline = self.job_order.output_baseline

        self.product_type = (
            L2B_OUTPUT_PRODUCT_FH if self.job_order.output_product == L2B_OUTPUT_PRODUCT_FH else L2B_OUTPUT_PRODUCT_TFH
        )

    def _core_processing(self):
        """Execute core FH L2b processing"""

        bps_logger.info("L2a products fusion into L2b")
        bps_logger.info(
            f"        checking with AUX PP2 minimum L2a products coverage of {self.aux_pp2_fh.minimumL2acoverage}%"
        )
        # Fusion: dgg tiling > temporal sorting > derivative > fd steps count > consistency check

        # dgg tiling: put each L2A into L2B Tile
        # note: input data can have INT or FLOAT no data values,
        # this function converts them to np.nan
        (
            skip_fh_computation,
            data_3d_mat_dict,
            dgg_tile_latitude_axis,
            dgg_tile_longitude_axis,
            dgg_tile_footprint,
        ) = dgg_tiling(
            self.l2a_fh_products_list,
            self.aux_pp2_fh.minimumL2acoverage,
            self.job_order.processing_parameters.tile_id,
            L2B_OUTPUT_PRODUCT_FH,
            self.l2b_fd_product,
        )

        if not skip_fh_computation:
            bps_logger.info("   Fusion algorithm:")

            # initializing output l2b dictionary
            l2b_data_final_d = {
                "dgg_latitude_axis": dgg_tile_latitude_axis,
                "dgg_longitude_axis": dgg_tile_longitude_axis,
            }

            data_footprint_list = []
            for l2a_product in self.l2a_fh_products_list:
                data_footprint_list.append(np.array(l2a_product.main_ads_input_information.footprint))

            # Average
            (
                l2b_data_final_d["fh"],
                l2b_data_final_d["quality"],
                l2b_data_final_d["heat_map"],
            ) = average_weighted(
                data_3d_mat_dict,
                dgg_tile_latitude_axis,
                dgg_tile_longitude_axis,
                data_footprint_list,
                self.aux_pp2_fh.forest_masking_flag,
                self.l2b_fd_product is not None,  # l2b_product_is_present
                self.aux_pp2_fh.rollOffFactorAzimuth,
                self.aux_pp2_fh.rollOffFactorRange,
            )
            l2b_data_final_d["fh"] = l2b_data_final_d["fh"].astype(np.float32)
            l2b_data_final_d["quality"] = l2b_data_final_d["quality"].astype(np.float32)
            l2b_data_final_d["heat_map"] = l2b_data_final_d["heat_map"].astype(np.float32)

            l2b_data_final_d["bps_fnf"] = data_3d_mat_dict["bps_fnf"]
            # Construct main_ads_input_information_l2b, for the L2b FH writer:
            (
                l2a_inputs_list,
                mission_phase_id,
                global_coverage_id,
                self.basin_id_list,
            ) = self._get_l2a_inputs_information()

            main_ads_input_information_l2b = BIOMASSL2bFHMainADSInputInformation(
                common_annotation_models_l2.InputInformationL2BL3ListType(
                    l2a_inputs=l2a_inputs_list,
                    count=len(l2a_inputs_list),
                ),
            )

            # Compute additional heat map:
            l2b_data_final_d["acquisition_id_image"] = compute_l2a_contributing_heat_map(data_3d_mat_dict, "fh")

        else:
            l2b_data_final_d = None
            mission_phase_id = None
            global_coverage_id = None
            dgg_tile_footprint = None
            main_ads_input_information_l2b = None
            bps_logger.info("L2b FH processor: nothing to compute, exiting.")

        if self.l2b_fd_product:
            bps_fnf_type = main_annotation_models_l2b_fh.BpsFnfType.CFM
        else:
            bps_fnf_type = main_annotation_models_l2b_fh.BpsFnfType.FNF

        start_time_l2a = self.l2a_fh_products_list[0].main_ads_product.start_time
        stop_time_l2a = self.l2a_fh_products_list[0].main_ads_product.stop_time
        for l2a_product in self.l2a_fh_products_list:
            start_time_l2a = min(start_time_l2a, l2a_product.main_ads_product.start_time)
            stop_time_l2a = max(stop_time_l2a, l2a_product.main_ads_product.stop_time)

        # Footprint mask for quick looks transparency
        if AVERAGING_FACTOR_QUICKLOOKS > 1:
            fh_for_footprint = data_3d_mat_dict["fh"].squeeze()[
                ::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS
            ]
        else:
            fh_for_footprint = data_3d_mat_dict["fh"].squeeze()
        footprint_mask_for_quicklooks = np.logical_not(np.isnan(fh_for_footprint))

        return (
            l2b_data_final_d,
            self.job_order.processing_parameters.tile_id,
            mission_phase_id,
            global_coverage_id,
            dgg_tile_footprint,
            main_ads_input_information_l2b,
            bps_fnf_type,
            skip_fh_computation,
            start_time_l2a,
            stop_time_l2a,
            footprint_mask_for_quicklooks,
        )

    def _fill_product_for_writing(self, tile_id):
        # TEMP = Fill needed fields with dummy values:
        # fixed values (i.e. "mission=BIOMASS") are directly set in _write_to_output function

        meta_data_dict = {}  # one metadata for each tile, and one for each product (fh, fh quality, bps fnf, heat_map)

        # fill common fileds
        meta_data_temp = [
            BIOMASSL2bFHProductMeasurement.MetadataCOG(
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
        meta_data_dict = {
            "fh": meta_data_temp[0],
            "quality": meta_data_temp[1],
            "bps_fnf": meta_data_temp[2],
            "heat_map": meta_data_temp[3],
            "acquisition_id_image": meta_data_temp[4],
        }
        meta_data_dict["fh"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_HEIGHT_M.value
        )

        meta_data_dict["quality"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_HEIGHT_QUALITY.value
        )

        meta_data_dict["bps_fnf"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_MASK.value
        )

        meta_data_dict["bps_fnf"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data_dict["heat_map"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.HEAT_MAP.value
        )

        meta_data_dict["acquisition_id_image"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + "acquisition id image"
        )

        meta_data_dict["acquisition_id_image"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        return meta_data_dict

    def _write_to_output(
        self,
        processed_data_dict,
        tile_id,
        mission_phase_id,
        global_coverage_id,
        footprint,
        main_ads_input_information,
        bps_fnf_type,
        start_time_l2a,
        stop_time_l2a,
        footprint_mask_for_quicklooks,
    ):
        """Write output FH L2b products"""

        radar_carrier_frequency = 435000000.0

        dgg_latitude_axis = processed_data_dict["dgg_latitude_axis"]
        dgg_longitude_axis = processed_data_dict["dgg_longitude_axis"]

        metadata_dict = self._fill_product_for_writing(tile_id)

        self.processing_stop_time = datetime.now()
        self.stop_time = PreciseDateTime.now()

        # Fill input objects for BIOMASSL2aProductFH initialization:
        # measurement,
        # main_ads_product,
        # main_ads_raster_image,
        # main_ads_input_information,
        # main_ads_processing_parameters,
        # lut_ads,

        # measurement object
        measurement = BIOMASSL2bFHProductMeasurement(
            dgg_latitude_axis,
            dgg_longitude_axis,
            processed_data_dict,
            metadata_dict,
        )

        # main_ads_product
        mission = common_annotation_models_l2.MissionType.BIOMASS.value
        sensor_mode = common_annotation_models_l2.SensorModeType.MEASUREMENT.value
        main_ads_product = BIOMASSL2bFHMainADSproduct(
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
        pixel_representation_dict = {
            "fh": common_types.PixelRepresentationType.FOREST_HEIGHT_M,
            "quality": common_types.PixelRepresentationType.FOREST_HEIGHT_QUALITY,
            "bps_fnf": common_types.PixelRepresentationType.COMPUTED_FOREST_MASK,
            "heat_map": [common_types.PixelRepresentationType.HEAT_MAP],
            "acquisition_id_image": common_types.PixelRepresentationType.ACQUISITION_ID_IMAGE,
        }
        pixel_representation = common_annotation_models_l2.PixelRepresentationChoiceType(
            fh=pixel_representation_dict["fh"],
            quality=pixel_representation_dict["quality"],
            bps_fnf=pixel_representation_dict["bps_fnf"],
            fh_heat_map=pixel_representation_dict["heat_map"],
            acquisition_id_image=pixel_representation_dict["acquisition_id_image"],
        )
        pixel_type = common_annotation_models_l2.PixelTypeChoiceType(
            float_pixel_type=main_annotation_models_l2b_fh.PixelTypeType("32 bit Float"),
            int_pixel_type=main_annotation_models_l2b_fh.PixelTypeType("8 bit Unsigned Integer"),
        )

        no_data_value = common_annotation_models_l2.NoDataValueChoiceType(
            float_no_data_value=FLOAT_NODATA_VALUE,
            int_no_data_value=INT_NODATA_VALUE,
        )
        main_ads_raster_image = BIOMASSL2bFHMainADSRasterImage(
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

        # main_ads_processing_parameters
        compression_options_fh = main_annotation_models_l2b_fh.CompressionOptionsL2B(
            main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds(
                main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.Fh(
                    self.aux_pp2_fh.compression_options.mds.fh.compression_factor,
                    self.aux_pp2_fh.compression_options.mds.fh.max_z_error,
                ),
                main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.Quality(
                    self.aux_pp2_fh.compression_options.mds.fhquality.compression_factor,
                    self.aux_pp2_fh.compression_options.mds.fhquality.max_z_error,
                ),
                main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.BpsFnf(
                    self.aux_pp2_fh.compression_options.mds.bps_fnf.compression_factor
                ),
                main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.HeatMap(
                    self.aux_pp2_fh.compression_options.mds.heatmap.compression_factor,
                    self.aux_pp2_fh.compression_options.mds.heatmap.max_z_error,
                ),
                main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.AcquisitionIdImage(
                    self.aux_pp2_fh.compression_options.mds.acquisition_id_image.compression_factor
                ),
            ),
            self.aux_pp2_fh.compression_options.mds_block_size,
        )

        main_ads_processing_parameters = BIOMASSL2bMainADSProcessingParametersFH(
            bps.l2b_fh_processor.__version__,
            self.start_time,
            forest_masking_flag=self.aux_pp2_fh.forest_masking_flag,
            bps_fnf=bps_fnf_type,
            compression_options=compression_options_fh,
            minumum_l2a_coverage=self.aux_pp2_fh.minimumL2acoverage,
            roll_off_factor_azimuth=self.aux_pp2_fh.rollOffFactorAzimuth,
            roll_off_factor_range=self.aux_pp2_fh.rollOffFactorRange,
        )

        # Initialize FH Product
        product_to_write = BIOMASSL2bFHProduct(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            self.aux_pp2_fh.l2bFHProductDOI,
        )

        # Write to file the FH Product
        write_obj = BIOMASSL2bFHProductWriter(
            product_to_write,
            self.product_path,
            bps.l2b_fh_processor.BPS_L2B_FH_PROCESSOR_NAME,
            bps.l2b_fh_processor.__version__,
            footprint,
            [acq.name for acq in self.job_order.input_l2a_products],
            ground_corner_points(dgg_latitude_axis, dgg_longitude_axis),
            self.job_order.aux_pp2_fh_path.name,
            start_time_l2a,
            stop_time_l2a,
            footprint_mask_for_quicklooks,
            (self.job_order.input_l2b_fd_product.name if self.job_order.input_l2b_fd_product else None),
        )
        write_obj.write()

    def run_l2b_fh_processing(self):
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
            bps_fnf_type,
            skip_fh_computation,
            start_time_l2a,
            stop_time_l2a,
            footprint_mask_for_quicklooks,
        ) = self._core_processing()

        if not skip_fh_computation:
            assert l2b_data_final_d is not None

            self._write_to_output(
                l2b_data_final_d,
                tile_id,
                mission_phase_id,
                global_coverage_id,
                footprint,
                main_ads_input_information,
                bps_fnf_type,
                start_time_l2a,
                stop_time_l2a,
                footprint_mask_for_quicklooks,
            )


def average_weighted(
    data_3d_mat_dict: dict,
    dgg_tile_latitude_axis: np.ndarray,
    dgg_tile_longitude_axis: np.ndarray,
    data_footprint_list: list[np.ndarray],
    forest_masking_flag: bool,
    l2b_product_is_present: bool,
    roll_off_factor_azimuth: float,
    roll_off_factor_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FH aggregation rule: averaging.
    Compute the output L2B FH product by averaging the various L2a forest height maps.

    Parameters
    ----------
    data_3d_mat_dict: dict
        The dictionary contains all of the L2a MDS (FH, FH Quality, BPS_FNF, Heat map)
        Each dictionary entry is a 3d matrix of shape [num_lat_dgg_tile, num_lon_dgg_tile, num_l2a_products]
        Note: each 3D matrix third dimension is already temporal ordered (see l2a_temporal_sorting())
        Note. BPS_FNF can be the FNF or the CFM from FD product (see dgg_tiling)
    dgg_tile_latitude_axis: np.ndarray
        Latitude axis of the L2b DGG tile, [deg] [num_lat_dgg,]
    dgg_tile_longitude_axis: np.ndarray
        Longitude axis of the L2b DGG tile,, [frg] [num_lon_dgg,]
    data_footprint_list: List[np.ndarray]
    forest_masking_flag: bool
        Flag to mask or not the ageraging, with the Forest Mask from BPS_FNF
    l2b_product_is_present: bool
        Flag to choose the aggregation formula,
        basing on presence or absence of the l2b produt containing the CFM
    roll_off_factor_azimuth: float
        Roll off azimuth factor used for the feathering weights generation
    roll_off_factor_range: float
        Roll off range factor used for the feathering weights generation

    Returns
    -------
    fh_average: np.ndarray
        FH map, aggregated at pixel level, covering one DGG tile
        Dimensions: [num_lat_dgg_tile, num_lon_dff_tile]
    fhquality_average: np.ndarray
        FH quality map, aggregated at pixel level, covering one DGG tile
        Dimensions: [num_lat_dgg_tile, num_lon_dff_tile]
    heat_map_average: np.ndarray
        Heat map, aggregated at pixel level, covering one DGG tile
        Dimensions: [num_lat_dgg_tile, num_lon_dff_tile]
    """

    bps_logger.info("        Forest height aggregation at pixel level by weighted average:")

    if not forest_masking_flag:
        bps_logger.info("            without forest mask masking, due to AUX PP2 FH forest masking flag set to False")
        forest_mask = 1
    else:
        bps_logger.info("        forest masking flag is True in AUX PP FH")

        if l2b_product_is_present:
            bps_logger.info("        masking using input provided CFM, from L2B FD product")
        else:
            bps_logger.info("        masking using external FNF (CFM from L2B FD product not provided as input)")
        # Usage of the Forest Mask:
        # Where is 0 or no data value, place a nan. Valid only where it is 1
        forest_mask = np.where(
            np.logical_or(
                data_3d_mat_dict["bps_fnf"] == 0,
                data_3d_mat_dict["bps_fnf"] == INT_NODATA_VALUE,
            ),
            np.nan,
            data_3d_mat_dict["bps_fnf"],
        ).astype(np.float64)

    bps_logger.info(f"        using roll off factor azimuth of {roll_off_factor_azimuth}")
    bps_logger.info(f"        using roll off factor range of {roll_off_factor_range}")

    # creation of the feathering weights
    data_footprint_list = sort_footprints(data_footprint_list)
    weights = feathering_weights(
        dgg_tile_latitude_axis,
        dgg_tile_longitude_axis,
        data_footprint_list,
        roll_off_factor_azimuth,
        roll_off_factor_range,
    )

    # Aggregation
    quality_per_weights = (1 / (0.01 + data_3d_mat_dict["quality"])) * weights
    denomimator = np.nansum(quality_per_weights, axis=2)
    denomimator[denomimator == 0.0] = np.nan  # avoid divide by zero
    if l2b_product_is_present:
        # For CFM , forest masks are all equal (replicas)
        forest_mask = forest_mask[:, :, 0] if len(forest_mask.shape) == 3 else forest_mask
        # Formula with l2b product CFM as forest_mask
        fh_average = (
            forest_mask
            * np.nansum(
                quality_per_weights * data_3d_mat_dict["fh"],
                axis=2,
            )
            / denomimator
        )

        fhquality_average = (
            forest_mask
            * np.nansum(
                quality_per_weights * data_3d_mat_dict["quality"],
                axis=2,
            )
            / denomimator
        )

        heat_map_average = forest_mask * np.nansum(
            quality_per_weights,
            axis=2,
        )

    else:
        # Formula with annotated FNF as forest_mask
        fh_average = (
            np.nansum(
                quality_per_weights * data_3d_mat_dict["fh"] * forest_mask,
                axis=2,
            )
            / denomimator
        )

        fhquality_average = (
            np.nansum(
                (1 / (0.01 + data_3d_mat_dict["quality"])) * weights * data_3d_mat_dict["quality"] * forest_mask,
                axis=2,
            )
            / denomimator
        )

        heat_map_average = np.nansum(
            quality_per_weights * forest_mask,
            axis=2,
        )

    # convert nan to FLOAT_NODATA_VALUE
    fh_average = np.where(np.isnan(fh_average), FLOAT_NODATA_VALUE, fh_average)
    fhquality_average = np.where(np.isnan(fhquality_average), FLOAT_NODATA_VALUE, fhquality_average)
    heat_map_average = np.where(np.isnan(heat_map_average), 0.0, heat_map_average)

    return fh_average, fhquality_average, heat_map_average


@nb.njit(nogil=True, cache=True)
def raised_cosine_nb(x, roll_off):
    """Raised cosine computation
    See feathering_weights() for more details.

    Called with numba:
        fast for more than one footprint
        ultra fast when cached!
    """
    return 0.5 * (1.0 + np.cos(np.pi / 2.0 * x / roll_off))


@nb.njit(nogil=True, cache=True, parallel=True)
def feathering_weights_core(y, x, ne, se, sw, nw, roll_off_factor_azimuth, roll_off_factor_range):
    """Core function of feathering_weights
    See feathering_weights() for more details.

    Called with numba:
        fast for more than one footprint
        ultra fast when cached!
    """

    weights = np.zeros((y.size, x.size))

    # find midpoint
    mx_FP = np.mean(np.array([ne[0], se[0], sw[0], nw[0]]))
    my_FP = np.mean(np.array([ne[1], se[1], sw[1], nw[1]]))

    # find direction
    azimuth = np.arctan2((nw[0] + ne[0] - sw[0] - se[0]) / 2.0, (nw[1] + ne[1] - sw[1] - se[1]) / 2.0)

    cos_az = np.cos(azimuth)
    sin_az = np.sin(-azimuth)

    # rotate and shift footprint
    ne_r = [
        (ne[0] - mx_FP) * cos_az + (ne[1] - my_FP) * sin_az,
        -(ne[0] - mx_FP) * sin_az + (ne[1] - my_FP) * cos_az,
    ]
    se_r = [
        (se[0] - mx_FP) * cos_az + (se[1] - my_FP) * sin_az,
        -(se[0] - mx_FP) * sin_az + (se[1] - my_FP) * cos_az,
    ]
    sw_r = [
        (sw[0] - mx_FP) * cos_az + (sw[1] - my_FP) * sin_az,
        -(sw[0] - mx_FP) * sin_az + (sw[1] - my_FP) * cos_az,
    ]
    nw_r = [
        (nw[0] - mx_FP) * cos_az + (nw[1] - my_FP) * sin_az,
        -(nw[0] - mx_FP) * sin_az + (nw[1] - my_FP) * cos_az,
    ]

    # find half-axis length
    hdx = (ne_r[0] - nw_r[0] + se_r[0] - sw_r[0]) / 4.0
    hdy = (ne_r[1] - se_r[1] + nw_r[1] - sw_r[1]) / 4.0

    # dimensional roll-offs
    rollx = (roll_off_factor_range) * hdx
    rolly = (roll_off_factor_azimuth) * hdy
    rollx_limit = (1.0 - roll_off_factor_range) * hdx
    rolly_limit = (1.0 - roll_off_factor_azimuth) * hdy

    # iterate on all points
    for idy in nb.prange(y.size):
        ky = y[idy]

        for idx in nb.prange(x.size):
            kx = x[idx]

            # counter rotate point
            xr = (kx - mx_FP) * cos_az + (ky - my_FP) * sin_az
            yr = -(kx - mx_FP) * sin_az + (ky - my_FP) * cos_az

            if xr >= -hdx and xr <= hdx and yr >= -hdy and yr <= hdy:
                weights[idy, idx] = 1.0

                if xr < -rollx_limit:
                    weights[idy, idx] *= raised_cosine_nb(-xr - rollx_limit, rollx)
                elif xr > rollx_limit:
                    weights[idy, idx] *= raised_cosine_nb(xr - rollx_limit, rollx)

                if yr < -rolly_limit:
                    weights[idy, idx] *= raised_cosine_nb(-yr - rolly_limit, rolly)
                elif yr > rolly_limit:
                    weights[idy, idx] *= raised_cosine_nb(yr - rolly_limit, rolly)

    return weights


def feathering_weights_core_serial(y, x, ne, se, sw, nw, roll_off_factor_azimuth, roll_off_factor_range):
    """Core function of feathering_weights
    See feathering_weights() for more details.

    Called with numba:
        fast for more than one footprint
        ultra fast when cached!
    """

    weights = np.zeros((y.size, x.size))

    # find midpoint
    mx_FP = np.mean(np.array([ne[0], se[0], sw[0], nw[0]]))
    my_FP = np.mean(np.array([ne[1], se[1], sw[1], nw[1]]))

    # find direction
    azimuth = np.arctan2((nw[0] + ne[0] - sw[0] - se[0]) / 2.0, (nw[1] + ne[1] - sw[1] - se[1]) / 2.0)

    cos_az = np.cos(azimuth)
    sin_az = np.sin(-azimuth)

    # rotate and shift footprint
    ne_r = [
        (ne[0] - mx_FP) * cos_az + (ne[1] - my_FP) * sin_az,
        -(ne[0] - mx_FP) * sin_az + (ne[1] - my_FP) * cos_az,
    ]
    se_r = [
        (se[0] - mx_FP) * cos_az + (se[1] - my_FP) * sin_az,
        -(se[0] - mx_FP) * sin_az + (se[1] - my_FP) * cos_az,
    ]
    sw_r = [
        (sw[0] - mx_FP) * cos_az + (sw[1] - my_FP) * sin_az,
        -(sw[0] - mx_FP) * sin_az + (sw[1] - my_FP) * cos_az,
    ]
    nw_r = [
        (nw[0] - mx_FP) * cos_az + (nw[1] - my_FP) * sin_az,
        -(nw[0] - mx_FP) * sin_az + (nw[1] - my_FP) * cos_az,
    ]

    # find half-axis length
    hdx = (ne_r[0] - nw_r[0] + se_r[0] - sw_r[0]) / 4.0
    hdy = (ne_r[1] - se_r[1] + nw_r[1] - sw_r[1]) / 4.0

    # dimensional roll-offs
    rollx = (roll_off_factor_range) * hdx
    rolly = (roll_off_factor_azimuth) * hdy
    rollx_limit = (1.0 - roll_off_factor_range) * hdx
    rolly_limit = (1.0 - roll_off_factor_azimuth) * hdy

    # iterate on all points
    for idy in nb.prange(y.size):
        ky = y[idy]

        for idx in nb.prange(x.size):
            kx = x[idx]

            # counter rotate point
            xr = (kx - mx_FP) * cos_az + (ky - my_FP) * sin_az
            yr = -(kx - mx_FP) * sin_az + (ky - my_FP) * cos_az

            if xr >= -hdx and xr <= hdx and yr >= -hdy and yr <= hdy:
                weights[idy, idx] = 1.0

                if xr < -rollx_limit:
                    weights[idy, idx] *= raised_cosine_nb(-xr - rollx_limit, rollx)
                elif xr > rollx_limit:
                    weights[idy, idx] *= raised_cosine_nb(xr - rollx_limit, rollx)

                if yr < -rolly_limit:
                    weights[idy, idx] *= raised_cosine_nb(-yr - rolly_limit, rolly)
                elif yr > rolly_limit:
                    weights[idy, idx] *= raised_cosine_nb(yr - rolly_limit, rolly)

    return weights


def feathering_weights(
    dgg_tile_latitude_axis: np.ndarray,
    dgg_tile_longitude_axis: np.ndarray,
    data_footprint_deg_list: list[np.ndarray],
    roll_off_factor_azimuth: float,
    roll_off_factor_range: float,
):
    """
    FH aggregation rule: averaging.

    Parameters
    ----------
    dgg_tile_latitude_axis: np.ndarray
        Latitude axis of the L2b DGG tile, [deg] [num_lat_dgg,]
    dgg_tile_longitude_axis: np.ndarray
        Longitude axis of the L2b DGG tile,, [deg] [num_lon_dgg,]
    data_footprint_deg_list: List[np.ndarray]
        List: one element for each l2a product
        Each element is an array with footprint as:
        [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon] [deg]
    roll_off_factor_azimuth: float
        Roll off azimuth factor used for the feathering weights generation
    roll_off_factor_range: float
        Roll off range factor used for the feathering weights generation

    Returns
    -------
    weights: np.ndarray
    Feathering werights, one matrix already bidimensional
        dimensions: [num_lat_dgg, num_lon_dgg, num_l2a_products]
    """

    # number of images
    num_imms = len(data_footprint_deg_list)

    # midpoint  of the tile
    m_lat = (dgg_tile_latitude_axis[0] + dgg_tile_latitude_axis[-1]) / 2.0
    m_lon = (dgg_tile_longitude_axis[0] + dgg_tile_longitude_axis[-1]) / 2.0

    # scaling factors
    d_lat = 111000
    d_lon = 111000 * np.cos(m_lat / 180.0 * np.pi)

    # transform Lat, Lon in planar coords
    y = (dgg_tile_latitude_axis - m_lat) * d_lat
    x = (dgg_tile_longitude_axis - m_lon) * d_lon

    # build output domain
    weights = np.zeros(shape=(len(y), len(x), num_imms), dtype=np.float32)

    for idN, fp in enumerate(data_footprint_deg_list):
        # transform footprint vertices in planar coords
        ne = np.array([(fp[1] - m_lon) * d_lon, (fp[0] - m_lat) * d_lat])
        se = np.array([(fp[3] - m_lon) * d_lon, (fp[2] - m_lat) * d_lat])
        sw = np.array([(fp[5] - m_lon) * d_lon, (fp[4] - m_lat) * d_lat])
        nw = np.array([(fp[7] - m_lon) * d_lon, (fp[6] - m_lat) * d_lat])

        weights[:, :, idN] = feathering_weights_core(
            y, x, ne, se, sw, nw, roll_off_factor_azimuth, roll_off_factor_range
        )

    # from matplotlib import pyplot as plt

    # longitude_mesh, latitude_mesh = np.meshgrid(
    #     dgg_tile_longitude_axis, dgg_tile_latitude_axis
    # )
    # plt.scatter(longitude_mesh, latitude_mesh, s=1, c=weights[:, :, 0], cmap="jet")
    # # plt.plot(A[:, 0], A[:, 1], color="k")
    # plt.axis("equal")
    # plt.show()

    return weights
