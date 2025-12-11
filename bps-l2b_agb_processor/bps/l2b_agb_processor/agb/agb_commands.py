# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2B AGB commands
----------------
"""

import concurrent.futures
import multiprocessing
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import bps.l2b_agb_processor
import numba as nb
import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common_types
from bps.common.io.common_types import IntArray, MinMaxType, MinMaxTypeWithUnit
from bps.common.lcm_utils import LCMMask
from bps.l2b_agb_processor.agb import BPS_L2B_AGB_PROCESSOR_NAME
from bps.l2b_agb_processor.core.aux_pp2_2b_agb import AuxProcessingParametersL2BAGB
from bps.l2b_agb_processor.core.joborder_l2b_agb import L2bAGBJobOrder
from bps.l2b_agb_processor.core.translate_job_order import L2B_OUTPUT_PRODUCT_AGB
from bps.l2b_agb_processor.io.aux_pp2_2b_agb_models import AgbIndexingType
from bps.l2b_agb_processor.l2b_common_functionalities import _check_coverage, _get_l2a_cumulative_footprint, dgg_tiling
from bps.transcoder.io import common_annotation_models_l2, main_annotation_models_l2b_agb
from bps.transcoder.sarproduct.biomass_l2aproduct import BIOMASSL2aProductGN
from bps.transcoder.sarproduct.biomass_l2bagbproduct import (
    BIOMASSL2bAGBMainADSInputInformation,
    BIOMASSL2bAGBMainADSproduct,
    BIOMASSL2bAGBMainADSRasterImage,
    BIOMASSL2bAGBProduct,
    BIOMASSL2bAGBProductMeasurement,
    BIOMASSL2bMainADSProcessingParametersAGB,
    agbTileStatus,
)
from bps.transcoder.sarproduct.biomass_l2bagbproduct_writer import (
    AVERAGING_FACTOR_QUICKLOOKS,
    COMPRESSION_EXIF_CODES_LERC_ZSTD,  # LERC, ZSTD
    DECIMATION_FACTOR_QUICKLOOKS,
    FLOAT_NODATA_VALUE,
    INT_NODATA_VALUE,
    BIOMASSL2bAGBProductWriter,
)
from bps.transcoder.sarproduct.biomass_l2bfdproduct import BIOMASSL2bFDProduct
from bps.transcoder.sarproduct.l2_annotations import COORDINATE_REFERENCE_SYSTEM, ground_corner_points
from bps.transcoder.utils.dgg_utils import create_dgg_sampling_dict, dgg_search_tiles

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="overflow encountered in exp")

NUM_LCM_CLASSES = 220


class AGBL2B:
    """Above Ground Biomass L2b Processor"""

    def __init__(
        self,
        job_order: L2bAGBJobOrder,
        aux_pp2_ab: AuxProcessingParametersL2BAGB,
        working_dir: Path,
        l2a_gn_products_list: list[BIOMASSL2aProductGN],
        lcm_product: LCMMask,
        cal_agb_product_dict: dict[str, np.ndarray],
        l2b_agb_product_list: list[BIOMASSL2bAGBProduct] | None = None,
        l2b_fd_product_list: list[BIOMASSL2bFDProduct] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        job_order  = L2bFDJobOrder
            content of the job order XML file
        aux_pp2_ab  = AuxProcessingParametersL2BAGB
            content of the AUX PP2 AGB XML file
        working_dir  = Path
            working directory
        l2a_gn_products_list  = list[BIOMASSL2aProductGN]
            list of all the L2a GN products paths
        lcm_product: LCMMask
            LCM, Land Cover Map, with its lat lon axes
        cal_agb_product_dict: dict[str, np.ndarray]
            Dictionary containing, for each tile in the keys,
            Reference AGB product 3d matrix, (num_lat x num_lon x 2)
            containing agb reference and standard deviation
        l2b_agb_product_list: Optional[BIOMASSL2bAGBProduct]
            Optional L2B AGB product (Over green area)
        l2b_fd_product_list: Optional[BIOMASSL2bFDProduct]
            Optional L2B FD product (Over green + blue area, only INT phase)
        """

        self.job_order = job_order
        self.aux_pp2_ab = aux_pp2_ab
        self.working_dir = working_dir
        self.l2a_gn_products_list = l2a_gn_products_list
        self.lcm_product = lcm_product
        self.cal_agb_product_dict = cal_agb_product_dict
        self.l2b_agb_product_list = l2b_agb_product_list
        self.l2b_fd_product_list = l2b_fd_product_list
        self.output_baseline = 0

    def _get_l2a_inputs_information(self):
        # Precompute here the info to construct main_ads_input_information_l2b:

        l2a_tile_id_dict = {}  # {Tile: number of L2a contributing}
        mission_phase_id = None
        global_coverage_id = None
        l2a_inputs_list = []
        basin_id_list = []
        for idx, l2a_product in enumerate(self.l2a_gn_products_list):
            basin_id_list = basin_id_list + l2a_product.main_ads_product.basin_id_list

            footprint = main_annotation_models_l2b_agb.FloatArrayWithUnits(
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

            for tile_id in l2a_product.main_ads_product.tile_id_list:
                if tile_id not in l2a_tile_id_dict.keys():
                    l2a_tile_id_dict[tile_id] = 1
                else:
                    l2a_tile_id_dict[tile_id] += 1

        return (
            l2a_inputs_list,
            mission_phase_id,
            global_coverage_id,
            l2a_tile_id_dict,
            list(np.unique(basin_id_list)),
        )

    def _initialize_processing(self):
        """Initialize the AGB L2b processing"""

        self.processing_start_time = datetime.now()
        self.start_time = PreciseDateTime.now()
        bps_logger.info("%s started", BPS_L2B_AGB_PROCESSOR_NAME)

        self.product_path = self.job_order.output_directory

        if self.job_order.output_baseline is not None:
            if self.job_order.output_product == L2B_OUTPUT_PRODUCT_AGB:
                self.output_baseline = self.job_order.output_baseline

        self.product_type = L2B_OUTPUT_PRODUCT_AGB

    def _core_processing(self):
        """Execute core AGB L2b processing"""

        ####################################################################################
        ### getting L2A informations,  cross checks with Job Order requests and loggings ###

        # Getting all needed data and information from L2A inputs:
        (
            l2a_inputs_list,
            mission_phase_id,
            global_coverage_id,
            l2a_input_tiles_dict,
            self.basin_id_list,
        ) = self._get_l2a_inputs_information()
        l2a_input_tiles_id = list(l2a_input_tiles_dict.keys())

        # AGBMainADSInputInformation: same for all outpuit tiles in the 3x3
        main_ads_input_information_l2b = BIOMASSL2bAGBMainADSInputInformation(
            common_annotation_models_l2.InputInformationL2BL3ListType(
                l2a_inputs=l2a_inputs_list,
                count=len(l2a_inputs_list),
            ),
        )

        # Get the 5x5 and 3x3 grid of Tlie ID names surrounding job order central tile id
        tiles_ids_5x5_list = get_5x5_tiles_ids(self.job_order.processing_parameters.tile_id)
        tiles_ids_3x3_list = get_3x3_tiles_ids(self.job_order.processing_parameters.tile_id)

        # Checking optional l2b AGB and FD products
        bps_logger.info(f"    using AUX PP reference selection option {self.aux_pp2_ab.referenceSelection}")

        self.regression_matrix_subsampling_factor = 1
        if self.l2b_agb_product_list:
            new_list = []
            bps_logger.info(
                f"    The {len(self.l2b_agb_product_list)} optional L2B AGB products specified in job order, cover following Tiles: {[l2b_agb_product.main_ads_product.tile_id_list[0] for l2b_agb_product in self.l2b_agb_product_list]}"
            )
            for l2b_agb_product in self.l2b_agb_product_list:
                if l2b_agb_product.main_ads_product.tile_id_list[0] not in tiles_ids_5x5_list:
                    bps_logger.warning(
                        f"        optional L2B AGB covering Tile {l2b_agb_product.main_ads_product.tile_id_list[0]} is not part of the 5x5 tiles neighborhood"
                    )
                else:
                    new_list.append(l2b_agb_product)

            if len(new_list) == 0:
                bps_logger.warning("        Any of the optional L2B AGB products is inside the 5x5 tiles neighborhood")
                self.l2b_agb_product_list = None
            else:
                self.l2b_agb_product_list = new_list

                bps_logger.info(
                    f"    using {len(self.l2b_agb_product_list)} optional L2B AGB products specified in job order, covering following Tiles: {[l2b_agb_product.main_ads_product.tile_id_list[0] for l2b_agb_product in self.l2b_agb_product_list]}"
                )

        else:
            bps_logger.info("    No optional L2B AGB products specified in job order")

        if self.l2b_fd_product_list:
            bps_logger.info(
                f"    using {len(self.l2b_fd_product_list)} optional L2B FD products specified in job order, covering following Tiles: {[l2b_fd_product.main_ads_product.tile_id_list[0] for l2b_fd_product in self.l2b_fd_product_list]}"
            )
        else:
            bps_logger.info("    No optional L2B FD products specified in job order")

        if self.l2b_agb_product_list:
            bps_logger.info("Executing AGB, Second Iteration")
        else:
            bps_logger.info("Executing AGB, First Iteration")

        bps_logger.info(
            f"Summary of contributing inputs and resources for the 5x5 DGG tiles surrounding {self.job_order.processing_parameters.tile_id}:"
        )
        counter_3x3 = 0
        for tile_id in tiles_ids_5x5_list:
            num_l2a_contributing_for_this_tile = [
                (l2a_input_tiles_dict[tile_id] if tile_id in l2a_input_tiles_dict.keys() else 0)
            ][0]

            num_cal_ab_contributing_for_this_tile = 0
            for key in self.cal_agb_product_dict.keys():
                if key == tile_id:
                    num_cal_ab_contributing_for_this_tile += 1

            num_l2b_fd_contributing_for_this_tile = 0
            if self.l2b_fd_product_list is not None:
                for l2b_fd_product in self.l2b_fd_product_list:
                    if tile_id in l2b_fd_product.main_ads_product.tile_id_list:
                        num_l2b_fd_contributing_for_this_tile += 1

            num_l2b_agb_contributing_for_this_tile = 0
            if self.l2b_agb_product_list is not None:
                for l2b_agb_product in self.l2b_agb_product_list:
                    if tile_id in l2b_agb_product.main_ads_product.tile_id_list:
                        num_l2b_agb_contributing_for_this_tile += 1

            if self.l2b_agb_product_list is not None:
                # Second iteration
                bps_logger.info(
                    f"    Tile ID {tile_id} = L2A GN:{num_l2a_contributing_for_this_tile}, CAL_AB:{num_cal_ab_contributing_for_this_tile}, L2B_FD:{num_l2b_fd_contributing_for_this_tile}, L2B_AGB:{num_l2b_agb_contributing_for_this_tile}"
                )
            else:
                # First iteration
                bps_logger.info(
                    f"    Tile ID {tile_id} = L2A GN:{num_l2a_contributing_for_this_tile}, CAL_AB:{num_cal_ab_contributing_for_this_tile}, L2B_FD:{num_l2b_fd_contributing_for_this_tile}"
                )
            if tile_id in tiles_ids_3x3_list:
                counter_3x3 += num_l2a_contributing_for_this_tile

        skip_agb_computation = False
        if not counter_3x3:
            bps_logger.info(
                "Any of the input L2A GN is present in the 3x3 tiles group surrounding central tile, skipping processor."
            )
            skip_agb_computation = True

        data_object_5x5 = {}
        # following covragres are to fill main annotation xml section
        # computation is done in various sections of this main
        cal_ab_original_coverage_per_tile = None
        gn_original_coverage_per_tile = None
        cal_ab_coverage_per_tile = None
        gn_coverage_per_tile = None
        if not skip_agb_computation:
            num_tiles_in = len(l2a_input_tiles_id)
            bps_logger.info(
                f"The {len(self.job_order.input_l2a_products)} L2a GN products in input, cover a total of {num_tiles_in} DGG tiles"
            )

            #######################################################################################
            ### Costruct a single data object containing all the data needed for the inversion,
            # organized as dictionaries and lists, containing data for all the used Tiles, up to 25

            # Discard from the input L2A GN Tile IDS, the ones outside the 5x5 Grid
            l2a_input_tiles_id = extract_5x5_tiles_from_total(
                l2a_input_tiles_id,
                tiles_ids_5x5_list,
            )

            # Filling the data object dictionary, containing all the needed AGB processor inputs for each available tile in the 5x5 grid
            (
                data_object_5x5,
                dgg_tile_footprint_per_tile,
                gn_original_coverage_per_tile,  # for main annotation
                skip_agb_computation,
            ) = construct_5x5_data_object(
                self.l2a_gn_products_list,
                l2a_input_tiles_id,
                self.lcm_product,
                self.cal_agb_product_dict,
                self.aux_pp2_ab.minimumL2acoverage,
                self.l2b_agb_product_list,
                self.l2b_fd_product_list,
            )

            # Fill cal_ab_original_coverage_per_tile
            # Differently from "gn_original_coverage_per_tile", this is done outside construct_5x5_data_object,
            # because that function skips cal_ab if output of GN coverage
            cal_ab_original_coverage_per_tile = {}
            for tile_id, cal_ab in self.cal_agb_product_dict.items():
                cal_ab_original_coverage_per_tile[tile_id] = _compute_coverage_percentage_and_pixels(
                    cal_ab["cal_ab"][:, :, 0]
                )

            # _agb_unit_test_preparation(
            #     self.l2a_gn_products_list, data_object_5x5, tiles_ids_5x5_list
            # )

        if not skip_agb_computation:
            # Consolidating the 5x5 data block dictionary,
            (
                data_object_5x5,
                different_dates_mjd,
                cal_ab_coverage_per_tile,  # contains filtered coverages only
                gn_coverage_per_tile,  # contains filtered coverages only
            ) = consolidate_5x5_data_block(
                data_object_5x5,
                self.aux_pp2_ab.referenceSelection,
                self.aux_pp2_ab.rejected_landcover_classes,
                self.aux_pp2_ab.backscatterLimits,
                self.aux_pp2_ab.angleLimits,
                self.aux_pp2_ab.meanAGBLimits,
                self.aux_pp2_ab.stdAGBLimits,
                self.aux_pp2_ab.relativeAGBLimits,
                self.aux_pp2_ab.forest_masking_flag,
            )

            # Complete filtered cal_ab_coverage_per_tile
            # Because data_object_5x5 does not contain cal_ab which are out of GN coverage
            tile_ids_cal_ab_discarted = []
            for tile_id in self.cal_agb_product_dict.keys():
                if tile_id not in list(data_object_5x5.keys()):
                    tile_ids_cal_ab_discarted.append(tile_id)
            data_object_with_added_cal_abs = _create_cal_ab_object(
                self.cal_agb_product_dict,
                self.lcm_product,
                tile_ids_cal_ab_discarted,
            )

            (
                additional_cal_ab_after_global_filtering_per_tile,
                additional_cal_ab_after_agbvalue_filtering_per_tile,
                additional_cal_ab_after_agbstd_filtering_per_tile,
                additional_cal_ab_after_agbrelative_std_filtering_per_tile,
                additional_cal_ab_after_lcmclass_filtering_per_tile,
            ) = compute_filtered_coverage_cal_ab(
                data_object_with_added_cal_abs,
                self.aux_pp2_ab.rejected_landcover_classes,
                self.aux_pp2_ab.meanAGBLimits,
                self.aux_pp2_ab.stdAGBLimits,
                self.aux_pp2_ab.relativeAGBLimits,
            )
            for tile_id in additional_cal_ab_after_global_filtering_per_tile.keys():
                cal_ab_coverage_per_tile[tile_id] = {}
                cal_ab_coverage_per_tile[tile_id]["after_global_filtering"] = (
                    additional_cal_ab_after_global_filtering_per_tile[tile_id]
                )
                cal_ab_coverage_per_tile[tile_id]["after_agbvalue_filtering"] = (
                    additional_cal_ab_after_agbvalue_filtering_per_tile[tile_id]
                )
                cal_ab_coverage_per_tile[tile_id]["after_agbstd_filtering"] = (
                    additional_cal_ab_after_agbstd_filtering_per_tile[tile_id]
                )
                cal_ab_coverage_per_tile[tile_id]["after_agbrelative_std_filtering"] = (
                    additional_cal_ab_after_agbrelative_std_filtering_per_tile[tile_id]
                )
                cal_ab_coverage_per_tile[tile_id]["after_lcmclass_filtering"] = (
                    additional_cal_ab_after_lcmclass_filtering_per_tile[tile_id]
                )

            # Union of original and filtered coverages dictionaries for the main annotation xml
            #   Also, fill with zeros, all tiles or acquisitions missing in coverage dictionaries:
            #       coverage in main annotation is mandatory for all 5x5 tiles and for all input GN
            assert cal_ab_original_coverage_per_tile is not None
            assert gn_coverage_per_tile is not None
            for tile_id in cal_ab_original_coverage_per_tile.keys():
                cal_ab_coverage_per_tile[tile_id]["original_coverage"] = cal_ab_original_coverage_per_tile[tile_id]
            for tile_id in gn_coverage_per_tile.keys():
                gn_coverage_per_tile[tile_id]["original_coverage"] = gn_original_coverage_per_tile[tile_id]

            # Fill with zeros all the tiles not covered
            for tile_id in tiles_ids_5x5_list:
                if tile_id not in list(cal_ab_coverage_per_tile.keys()):
                    cal_ab_coverage_per_tile[tile_id] = {}
                    for coverage_string in [
                        "after_global_filtering",
                        "after_agbvalue_filtering",
                        "after_agbstd_filtering",
                        "after_agbrelative_std_filtering",
                        "after_lcmclass_filtering",
                        "original_coverage",
                    ]:
                        cal_ab_coverage_per_tile[tile_id][coverage_string] = [
                            float(0.0),
                            int(0),
                        ]

            for tile_id in tiles_ids_5x5_list:
                if tile_id not in list(gn_coverage_per_tile.keys()):
                    gn_coverage_per_tile[tile_id] = {
                        "after_global_filtering": {},
                        "after_sigma_filtering": {},
                        "after_angle_filtering": {},
                        "original_coverage": {},
                    }
                for coverage_string in [
                    "after_global_filtering",
                    "after_sigma_filtering",
                    "after_angle_filtering",
                    "original_coverage",
                ]:
                    for gn_product in self.l2a_gn_products_list:
                        name = gn_product.name

                        if name not in list(gn_coverage_per_tile[tile_id][coverage_string].keys()):
                            gn_coverage_per_tile[tile_id][coverage_string][name] = [
                                float(0.0),
                                int(0),
                            ]

            ################################################
            ### Costruct the input vectors for inversion ###
            reference_vectors, lcm_absolute_values, lcm_abs_to_rel_indices = build_reference_vectors(data_object_5x5)

            if len(reference_vectors["AGB_mean"]) == 0:
                bps_logger.warning("No valid data/reference pairs found in neighbourhood")
                skip_agb_computation = True

            if not skip_agb_computation:
                #############################
                ### Inversion preparation ###

                bps_logger.info("AGB inversion")
                bps_logger.info(f"Using AUX PP regressionSolver: {self.aux_pp2_ab.regressionSolver}")
                bps_logger.info(
                    f"Using AUX PP minimumPercentageOfFillableVoids: {self.aux_pp2_ab.minimumPercentageOfFillableVoids}"
                )
                bps_logger.info(f"Using AUX PP use constant N: {self.aux_pp2_ab.useConstantN}")
                if self.aux_pp2_ab.useConstantN:
                    bps_logger.info(f"Using AUX PP valuesConstantN: {self.aux_pp2_ab.valuesConstantN}")
                if self.aux_pp2_ab.useConstantN and not self.aux_pp2_ab.indexingN == "p":
                    bps_logger.info(
                        f"    indexingN modified from {self.aux_pp2_ab.indexingN} to 'p', since use constant N is {self.aux_pp2_ab.useConstantN}"
                    )
                    self.aux_pp2_ab.indexingN = AgbIndexingType.P.value

                bps_logger.info(
                    f"    parameter L depends on: {'polarization (p)' if 'p' in self.aux_pp2_ab.indexingL else ''}{', date (j)' if 'j' in self.aux_pp2_ab.indexingL else ''} {', forest class (k)' if 'k' in self.aux_pp2_ab.indexingL else ''}"
                )
                bps_logger.info(
                    f"    parameter A depends on: {'polarization (p)' if 'p' in self.aux_pp2_ab.indexingA else ''}{', date (j)' if 'j' in self.aux_pp2_ab.indexingA else ''} {', forest class (k)' if 'k' in self.aux_pp2_ab.indexingA else ''}"
                )
                bps_logger.info(
                    f"    parameter N depends on: {'polarization (p)' if 'p' in self.aux_pp2_ab.indexingN else ''}{', date (j)' if 'j' in self.aux_pp2_ab.indexingN else ''} {', forest class (k)' if 'k' in self.aux_pp2_ab.indexingN else ''}"
                )

                start_time = datetime.now()

                # build mapping volumes for unknowns eq. 4.14
                n_dates_l2a = len(different_dates_mjd)
                n_pol = data_object_5x5[next(iter(data_object_5x5))]["data_list"][0]["data"].shape[2]
                n_lcm = len(np.unique(lcm_abs_to_rel_indices)) - 1

                M_l, u_l = build_unknowns_matrix(n_pol, n_dates_l2a, n_lcm, self.aux_pp2_ab.indexingL)
                M_a, u_a = build_unknowns_matrix(n_pol, n_dates_l2a, n_lcm, self.aux_pp2_ab.indexingA)
                M_n, u_n = build_unknowns_matrix(n_pol, n_dates_l2a, n_lcm, self.aux_pp2_ab.indexingN)

                # build regression matrix eq. 4.16
                bps_logger.info("    Build regression matrix")
                A = build_regression_matrix_jit(
                    reference_vectors["AGB_mean"],
                    reference_vectors["inc_angle"],
                    reference_vectors["pol_idx"],
                    reference_vectors["date_idx"],
                    reference_vectors["LCM_idx"],
                    M_l,
                    M_a + u_l,
                    M_n + u_l + u_a,
                    u_l + u_a + u_n,
                )

                stop_time = datetime.now()
                elapsed_time = (stop_time - start_time).total_seconds()
                remove_value = 0
                if self.aux_pp2_ab.useConstantN:
                    # See inside compute_linear_regression(), when self.aux_pp2_ab.useConstantN is True
                    remove_value = 3
                bps_logger.info(
                    f"       Regression unknowns, Incidence matrix entries : {u_l + u_a + u_n - remove_value}, {A.shape[0]}"
                )
                bps_logger.info(f"    Matrix creation : {elapsed_time:.3f} s")

                start_time = datetime.now()

                # compute regression eq. 4.18, 4.19
                if self.aux_pp2_ab.regressionSolver == "double":
                    astype_string = "float64"
                elif self.aux_pp2_ab.regressionSolver == "float":
                    astype_string = "float32"
                else:
                    raise ValueError(
                        f"Unrecognided AUX_PP parameter regressionSolver {self.aux_pp2_ab.regressionSolver}"
                    )

                # Subsample inputs for linear regression if avaliable RAM is notenough:

                # Used RAM live (considering here only major contributes)
                used_ram_live_MB = (
                    (
                        get_size(data_object_5x5)
                        + get_size(reference_vectors)
                        + get_size(self.l2a_gn_products_list[0].measurement.data_dict) * len(self.l2a_gn_products_list)
                    )
                    / 1024
                    / 1024
                )
                # Available RAM for regression, basing on input job order available RAM
                remaining_ram_for_regression_MB = self.job_order.device_resources.available_ram - used_ram_live_MB
                used_ram_A_MB = get_size(A) / 1024 / 1024
                while used_ram_A_MB * 4 / self.regression_matrix_subsampling_factor > remaining_ram_for_regression_MB:
                    self.regression_matrix_subsampling_factor += 1
                    if self.regression_matrix_subsampling_factor >= 8:
                        break

                bps_logger.info("Compute linear regression:")
                bps_logger.info(
                    f"    available RAM, from Job Order: {self.job_order.device_resources.available_ram} MB"
                )
                bps_logger.info(
                    f"    subsampling factor used, considering remaining RAM available for regression: {self.regression_matrix_subsampling_factor}"
                )

                u, std = compute_linear_regression(
                    np.copy(
                        reference_vectors["sigma"][:: self.regression_matrix_subsampling_factor].astype(astype_string)
                    ),
                    A[:: self.regression_matrix_subsampling_factor, :].astype(astype_string),
                    (np.array(self.aux_pp2_ab.valuesConstantN) if self.aux_pp2_ab.useConstantN else None),
                )

                stop_time = datetime.now()
                elapsed_time = (stop_time - start_time).total_seconds()

                bps_logger.info(f"    Matrix inversion: {elapsed_time:.3f} s")

                # build mapping volumes for regression parameters (L,alfa,n LUTs)

                L_unknowns = u[M_l]
                alfa_unknowns = u[M_a + u_l]
                n_unknowns = u[M_n + u_l + u_a]

                L_stddev = std[M_l]
                alfa_stddev = std[M_a + u_l]
                n_stddev = std[M_n + u_l + u_a]

                start_time = datetime.now()

                # compute rho eq. 4.22
                self.rho = compute_rho_jit(
                    reference_vectors["AGB_mean"],
                    reference_vectors["sigma"],
                    reference_vectors["inc_angle"],
                    reference_vectors["point_idx"],
                    reference_vectors["pol_idx"],
                    reference_vectors["date_idx"],
                    reference_vectors["LCM_idx"],
                    L_unknowns,
                    alfa_unknowns,
                    n_unknowns,
                )

                stop_time_all = datetime.now()
                elapsed_time = (stop_time_all - start_time).total_seconds()
                bps_logger.info(f"    Rho value : {self.rho} s")
                bps_logger.info(f"    Rho computation : {elapsed_time:.3f} s")

                ###################################
                ### start agb computation phase ###

                # agb computation is made per output DGG tile
                # One AGB for each of the 3x3 tiles is produced here
                bps_logger.info("AGB computation for the available tiles part of the 3x3 grid")
                start_time = datetime.now()
                l2b_data_final_per_tile = {}
                agb_tile_iteration = {}
                agb_tile_status = {}
                for tile_id in tiles_ids_3x3_list:
                    if tile_id in data_object_5x5.keys():
                        bps_logger.info(f"    Computation for Tile {tile_id}:")

                        num_lat = len(data_object_5x5[tile_id]["latitude_axis"])
                        num_lon = len(data_object_5x5[tile_id]["longitude_axis"])

                        # hold num and den terms of eq 4.36

                        W_A2 = np.zeros([num_lat, num_lon], "float32")
                        STD_A2 = np.zeros([num_lat, num_lon], "float32")
                        A2 = np.zeros([num_lat, num_lon], "float32")

                        # cycle over each polarization and each L2A_GN input projected onto DGG grid while computing eq. 4.30 4.35 and updating terms of eq 4.36
                        for list_entry in data_object_5x5[tile_id]["data_list"]:  # Cycling each data in the Tile
                            for pol_idx in np.arange(list_entry["data"].shape[2]):
                                W_A2, STD_A2, A2 = compute_agb_jit(
                                    W_A2,
                                    STD_A2,
                                    A2,
                                    list_entry["data"][:, :, pol_idx],
                                    list_entry["incidence_angle"],
                                    pol_idx,
                                    list_entry["date_idx"],
                                    lcm_abs_to_rel_indices[data_object_5x5[tile_id]["lcm"]],
                                    L_unknowns,
                                    alfa_unknowns,
                                    n_unknowns,
                                    L_stddev,
                                    alfa_stddev,
                                    n_stddev,
                                )

                        # finalize Agb and Std computation by eq. 4.37, 4.38
                        (
                            agb_current_tile,
                            agb_standard_deviation_current_tile,
                        ) = finalize_agb(W_A2, STD_A2, A2, self.rho)

                        # Now that the AGB is finalized, it is the time to update output fields, in detail:
                        # > Final AGB values (pixel wise) will be the previous ones from AGB L2B input (if present) and not NaN,
                        #                                 otherwise the new value here computed will be used
                        # > AGB tile iteration has already been updated during data consolidation, and is here retreived
                        # > AGB tile status is here computed, depending on previous state from AGB L2B input and on current AGB estimation results

                        # Update the AGB estimation, condidering previous cycle AGB from AGB L2B input
                        if data_object_5x5[tile_id]["l2b_agb"] is not None:
                            agb_current_tile = np.where(
                                np.logical_not(np.isnan(data_object_5x5[tile_id]["l2b_agb"][:, :, 0])),
                                data_object_5x5[tile_id]["l2b_agb"][:, :, 0],
                                agb_current_tile,
                            )

                        # AGB tile iteration to be saved, already updated during data loading:
                        agb_tile_iteration[tile_id] = data_object_5x5[tile_id]["agb_tile_iteration_updated"]

                        # Update the AGB Tile Status:

                        # Number of NaNs in the estimated AGB (whole)
                        num_nans_out = np.sum(np.isnan(agb_current_tile))

                        if agb_tile_iteration[tile_id] == 1:
                            # First iteration

                            # Number of NaNs in the estimated AGB (computed where the input data exists, in percentage)
                            num_nans_out_where_input_exists = (
                                100
                                * np.sum(
                                    np.isnan(agb_current_tile[np.logical_not(data_object_5x5[tile_id]["NANMASK"])])
                                )
                                / np.sum(data_object_5x5[tile_id]["NANMASK"] == False)  # noqa: E712
                            )

                            # Compute agb_tile_status at first iteration
                            if num_nans_out_where_input_exists > self.aux_pp2_ab.minimumPercentageOfFillableVoids:
                                agb_tile_status[tile_id] = agbTileStatus.PARTIAL
                            elif num_nans_out > 0:
                                agb_tile_status[tile_id] = agbTileStatus.NOTCOMPLETE
                            else:
                                agb_tile_status[tile_id] = agbTileStatus.COMPLETE

                        else:
                            # Second iteration
                            if num_nans_out == 0:
                                agb_tile_status[tile_id] = agbTileStatus.COMPLETE
                            else:
                                agb_tile_status[tile_id] = agbTileStatus.NOTCOMPLETE

                        bps_logger.info("        after AGB computation:")
                        bps_logger.info(f"          AGB Tile Iteration: {agb_tile_iteration[tile_id]}")
                        bps_logger.info(f"          AGB Tile Status: {agb_tile_status[tile_id].value}")

                        # Heat Maps
                        bps_logger.info("        Compute Heat Maps")
                        heat_maps_dict, acquisition_id_image = compute_heat_maps(
                            lat_n=len(data_object_5x5[tile_id]["latitude_axis"]),
                            lon_n=len(data_object_5x5[tile_id]["longitude_axis"]),
                            input_data_list=[
                                list_entry["data"] for list_entry in data_object_5x5[tile_id]["data_list"]
                            ],
                            denominator_eq_436=A2,
                            agb_estimation_current=agb_current_tile,
                            agb_estimation_previous=(
                                data_object_5x5[tile_id]["l2b_agb"][:, :, 0]
                                if data_object_5x5[tile_id]["l2b_agb"] is not None
                                else None
                            ),
                        )

                        # Fill final object
                        l2b_data_final_per_tile[tile_id] = {
                            "dgg_latitude_axis": data_object_5x5[tile_id]["latitude_axis"].astype(np.float32),
                            "dgg_longitude_axis": data_object_5x5[tile_id]["longitude_axis"].astype(np.float32),
                            "agb": agb_current_tile.astype(np.float32),
                            "agbstandard_deviation": agb_standard_deviation_current_tile.astype(np.float32),
                            "heat_map": heat_maps_dict,
                            "acquisition_id_image": acquisition_id_image,
                            "bps_fnf": (
                                data_object_5x5[tile_id]["bps_fnf"]
                                if tile_id in list(data_object_5x5.keys())
                                else np.zeros(
                                    data_object_5x5[self.job_order.processing_parameters.tile_id]["bps_fnf"].shape,
                                    dtype=np.uint8,
                                )
                            ),
                        }
                        stop_time = datetime.now()
                        elapsed_time = (stop_time - start_time).total_seconds()
                        bps_logger.info(f"    Elapsed time: {elapsed_time:.3f} s")

                bps_logger.info(
                    f"AGB has been computed for {len(l2b_data_final_per_tile.keys())} DGG tiles, part of the output 3x3 group of DGG tiles around central tile {self.job_order.processing_parameters.tile_id}"
                )

                # To log and write the estimation parameters
                polarization_names = self._get_polarisation_names()
                num_pols = len(polarization_names)

                if "j" in self.aux_pp2_ab.indexingL + self.aux_pp2_ab.indexingA + self.aux_pp2_ab.indexingN:
                    # At least one parameter depends on date
                    num_dates = L_unknowns.shape[1]
                    dates_values = different_dates_mjd
                else:
                    # No parameters depend on class, so write all in a single element attribute
                    num_dates = 1
                    dates_values = ""
                    for date in different_dates_mjd:
                        dates_values += date + " "
                    dates_values = [dates_values[:-1]]

                if "k" in self.aux_pp2_ab.indexingL + self.aux_pp2_ab.indexingA + self.aux_pp2_ab.indexingN:
                    # At least one parameter depends on class
                    num_classes = len(lcm_absolute_values)

                    classes_values = []
                    for lcm_absolute_value in lcm_absolute_values:
                        classes_values.append(str(lcm_absolute_value))
                else:
                    # No parameters depend on class, so write all in a single element attribute
                    num_classes = 1
                    classes_values = ""
                    for class_value in lcm_absolute_values:
                        classes_values += str(class_value) + " "
                    classes_values = [classes_values[:-1]]

                estimated_parameters = {
                    "indexingL": self.aux_pp2_ab.indexingL,
                    "indexingA": self.aux_pp2_ab.indexingA,
                    "indexingN": self.aux_pp2_ab.indexingN,
                    "L_unknowns": L_unknowns.astype(np.float64),
                    "alfa_unknowns": alfa_unknowns.astype(np.float64),
                    "n_unknowns": n_unknowns.astype(np.float64),
                    "L_stddev": L_stddev.astype(np.float64),
                    "alfa_stddev": alfa_stddev.astype(np.float64),
                    "n_stddev": n_stddev.astype(np.float64),
                    "different_dates_mjd": different_dates_mjd,
                    "num_pols": num_pols,
                    "polarization_names": polarization_names,
                    "num_dates": num_dates,
                    "dates_values": dates_values,
                    "num_classes": num_classes,
                    "classes_values": classes_values,
                }

                bps_logger.info("Estimated parameters:")
                for pol_idx in range(num_pols):
                    bps_logger.info(f"    Polarisation {polarization_names[pol_idx]}")

                    for class_idx in range(num_classes):
                        bps_logger.info(f"        LCM class {classes_values[class_idx]}")

                        for date_idx in range(num_dates):
                            bps_logger.info(f"            Date {dates_values[date_idx]}")

                            bps_logger.info(
                                f"                L mean  {estimated_parameters['L_unknowns'][pol_idx, date_idx, class_idx]}"
                            )
                            bps_logger.info(
                                f"                L std   {estimated_parameters['L_stddev'][pol_idx, date_idx, class_idx]}"
                            )

                            bps_logger.info(
                                f"                A mean  {estimated_parameters['alfa_unknowns'][pol_idx, date_idx, class_idx]}"
                            )
                            bps_logger.info(
                                f"                A std   {estimated_parameters['alfa_stddev'][pol_idx, date_idx, class_idx]}"
                            )

                            bps_logger.info(
                                f"                N mean  {estimated_parameters['n_unknowns'][pol_idx, date_idx, class_idx]}"
                            )
                            bps_logger.info(
                                f"                N std   {estimated_parameters['n_stddev'][pol_idx, date_idx, class_idx]}"
                            )

        # Contributing tiles

        if skip_agb_computation:
            if not counter_3x3 or len(data_object_5x5) == 0:
                # Any of the input L2A GN is present in the 3x3 tiles group surrounding central tile, skipping processor
                l2b_data_final_per_tile = None
                mission_phase_id = None
                global_coverage_id = None
                dgg_tile_footprint_per_tile = None
                main_ads_input_information_l2b = None
                agb_tile_status = None
                agb_tile_iteration = {}
                for tile_id in get_3x3_tiles_ids(self.job_order.processing_parameters.tile_id):
                    agb_tile_iteration[tile_id] = np.uint8(1)
                estimated_parameters = None
                cal_ab_coverage_per_tile = None
                gn_coverage_per_tile = None
                bps_logger.info("L2b AGB processor: nothing to compute, exiting")

            else:
                # write empty products, for all the tiles in the 3x3 which have an input L2A
                l2b_data_final_per_tile = {}
                agb_tile_iteration = {}
                agb_tile_status = {}
                for tile_id in tiles_ids_3x3_list:
                    if tile_id in list(data_object_5x5.keys()):
                        nan_matrix = (
                            np.ones(
                                data_object_5x5[tile_id]["data_list"][0]["data"][:, :, 0].shape,
                                dtype=np.float32,
                            )
                            * np.nan
                        )
                        # Fill final object
                        l2b_data_final_per_tile[tile_id] = {
                            "dgg_latitude_axis": data_object_5x5[tile_id]["latitude_axis"].astype(np.float32),
                            "dgg_longitude_axis": data_object_5x5[tile_id]["longitude_axis"].astype(np.float32),
                            "agb": nan_matrix,
                            "agbstandard_deviation": nan_matrix,
                            "heat_map": {
                                "heat_map": nan_matrix,
                                "heat_map_ref_data": nan_matrix,
                                "heat_map_additional_ref_data": nan_matrix,
                            },
                            "acquisition_id_image": nan_matrix,
                            "bps_fnf": np.zeros(
                                (
                                    nan_matrix.shape[0],
                                    nan_matrix.shape[1],
                                    data_object_5x5[tile_id]["bps_fnf"].shape[2],
                                ),
                                dtype=np.uint8,
                            ),
                        }
                        agb_tile_iteration[tile_id] = np.uint8(1)
                        agb_tile_status[tile_id] = agbTileStatus.PARTIAL

                polarization_names = self._get_polarisation_names()
                num_pols = len(polarization_names)
                num_classes = 1
                num_dates = 1
                nan_est_param_matrix = (
                    np.ones(
                        (num_pols, num_classes, num_dates),
                        dtype=np.float64,
                    )
                    * np.nan
                )
                self.rho = 0.0
                estimated_parameters = {
                    "indexingL": self.aux_pp2_ab.indexingL,
                    "indexingA": self.aux_pp2_ab.indexingA,
                    "indexingN": self.aux_pp2_ab.indexingN,
                    "L_unknowns": nan_est_param_matrix,
                    "alfa_unknowns": nan_est_param_matrix,
                    "n_unknowns": nan_est_param_matrix,
                    "L_stddev": nan_est_param_matrix,
                    "alfa_stddev": nan_est_param_matrix,
                    "n_stddev": nan_est_param_matrix,
                    "different_dates_mjd": ["nan" for idx in range(num_dates)],
                    "num_pols": num_pols,
                    "polarization_names": polarization_names,
                    "num_dates": num_dates,
                    "dates_values": ["nan" for idx in range(num_dates)],
                    "num_classes": num_classes,
                    "classes_values": ["nan" for idx in range(num_classes)],
                }

                skip_agb_computation = False  # to enable the writing of empty products
                bps_logger.info(
                    "L2b AGB processor: nothing to compute, writing empty product for each L2A tile with data"
                )

                # Fill with zeros, if not already computed:
                if cal_ab_coverage_per_tile is None:
                    for tile_id in tiles_ids_5x5_list:
                        cal_ab_coverage_per_tile[tile_id] = {
                            "original_coverage": [0.0, 0],
                            "after_global_filtering": [0.0, 0],
                            "after_agbvalue_filtering": [0.0, 0],
                            "after_agbstd_filtering": [0.0, 0],
                            "after_agbrelative_std_filtering": [0.0, 0],
                            "after_lcmclass_filtering": [0.0, 0],
                        }

                # Fill with zeros, if not already computed:
                if gn_coverage_per_tile is None:
                    for tile_id in tiles_ids_5x5_list:
                        gn_coverage_per_tile[tile_id] = {
                            "original_coverage": {},
                            "after_global_filtering": {},
                            "after_sigma_filtering": {},
                            "after_angle_filtering": {},
                        }

                        for gn_product in self.l2a_gn_products_list:
                            name = gn_product.name

                            gn_coverage_per_tile[tile_id]["original_coverage"][name] = [
                                0.0,
                                0,
                            ]
                            gn_coverage_per_tile[tile_id]["after_global_filtering"][name] = [0.0, 0]
                            gn_coverage_per_tile[tile_id]["after_sigma_filtering"][name] = [
                                0.0,
                                0,
                            ]
                            gn_coverage_per_tile[tile_id]["after_angle_filtering"][name] = [
                                0.0,
                                0,
                            ]

        start_time_l2a = self.l2a_gn_products_list[0].main_ads_product.start_time
        stop_time_l2a = self.l2a_gn_products_list[0].main_ads_product.stop_time
        for l2a_product in self.l2a_gn_products_list:
            start_time_l2a = min(start_time_l2a, l2a_product.main_ads_product.start_time)
            stop_time_l2a = max(stop_time_l2a, l2a_product.main_ads_product.stop_time)

        footprint_mask_for_quicklooks = {}
        for tile_id, data_object_1x1 in data_object_5x5.items():
            # Footprint mask for quick looks transparency
            if AVERAGING_FACTOR_QUICKLOOKS > 1:
                agb_for_footprint = data_object_1x1["data_list"][0]["incidence_angle"].squeeze()[
                    ::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS
                ]
            else:
                agb_for_footprint = data_object_1x1["data_list"][0]["incidence_angle"].squeeze()
            footprint_mask_for_quicklooks[tile_id] = np.logical_not(np.isnan(agb_for_footprint))

        return (
            l2b_data_final_per_tile,
            l2a_input_tiles_id,
            mission_phase_id,
            global_coverage_id,
            dgg_tile_footprint_per_tile,
            main_ads_input_information_l2b,
            skip_agb_computation,
            start_time_l2a,
            stop_time_l2a,
            agb_tile_iteration,
            agb_tile_status,
            estimated_parameters,
            tiles_ids_5x5_list,
            cal_ab_coverage_per_tile,
            gn_coverage_per_tile,
            footprint_mask_for_quicklooks,
        )

    def _fill_product_for_writing(self, tile_id):
        # TEMP = Fill needed fields with dummy values:
        # fixed values (i.e. "mission=BIOMASS") are directly set in _write_to_output function

        meta_data_dict = {}  # one metadata for each tile, and one for each product (agb, agb std, heat_map, acquisition id image)

        # fill common fileds
        meta_data_temp = [
            BIOMASSL2bAGBProductMeasurement.MetadataCOG(
                tile_id_list=[tile_id],
                basin_id_list=self.basin_id_list,
                compression=COMPRESSION_EXIF_CODES_LERC_ZSTD,  #  [LERC, ZSTD]
                image_description="",
                software="",
                dateTime=self.start_time.isoformat(timespec="seconds")[:-1],
            )
            for idx in range(5)
        ]

        # specific fileds
        meta_data_dict = {
            "agb": meta_data_temp[0],
            "agbstandard_deviation": meta_data_temp[1],
            "heat_map": meta_data_temp[2],
            "acquisition_id_image": meta_data_temp[3],
            "bps_fnf": meta_data_temp[4],
        }
        meta_data_dict["agb"].image_description = (
            f"BIOMASS L2b {self.product_type}"
            + ": "
            + common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_T_HA.value
        )
        meta_data_dict["agb"].bits_per_sample = 32
        meta_data_dict["agb"].max_z_error = self.aux_pp2_ab.compression_options.mds.AGB.max_z_error
        meta_data_dict["agb"].sample_format = 3
        meta_data_dict["agb"].gdal_nodata = FLOAT_NODATA_VALUE

        meta_data_dict["agbstandard_deviation"].image_description = (
            f"BIOMASS L2b {self.product_type}"
            + ": "
            + common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_QUALITY_T_HA.value
        )
        meta_data_dict["agbstandard_deviation"].bits_per_sample = 32
        meta_data_dict[
            "agbstandard_deviation"
        ].max_z_error = self.aux_pp2_ab.compression_options.mds.AGBstandardDeviation.max_z_error
        meta_data_dict["agbstandard_deviation"].sample_format = 3
        meta_data_dict["agbstandard_deviation"].gdal_nodata = FLOAT_NODATA_VALUE

        meta_data_dict["heat_map"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.HEAT_MAP.value
        )
        meta_data_dict["heat_map"].bits_per_sample = 8
        meta_data_dict["heat_map"].max_z_error = 0
        meta_data_dict["heat_map"].sample_format = 1
        meta_data_dict["heat_map"].gdal_nodata = INT_NODATA_VALUE
        meta_data_dict["heat_map"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data_dict["acquisition_id_image"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + "acquisition id image"
        )
        meta_data_dict["acquisition_id_image"].bits_per_sample = 8
        meta_data_dict["acquisition_id_image"].max_z_error = 0
        meta_data_dict["acquisition_id_image"].sample_format = 1
        meta_data_dict["acquisition_id_image"].gdal_nodata = INT_NODATA_VALUE
        meta_data_dict["acquisition_id_image"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        meta_data_dict["bps_fnf"].image_description = (
            f"BIOMASS L2b {self.product_type}" + ": " + common_types.PixelRepresentationType.FOREST_MASK.value
        )
        meta_data_dict["bps_fnf"].compression = [COMPRESSION_EXIF_CODES_LERC_ZSTD[1]]  # [ZSTD]

        return meta_data_dict

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
        agb_tile_iteration,
        agb_tile_status,
        estimated_parameters,
        contributing_tiles,
        cal_ab_coverage_per_tile,
        gn_coverage_per_tile,
        footprint_mask_for_quicklooks,
    ):
        """Write output AGB L2b product"""

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
        measurement = BIOMASSL2bAGBProductMeasurement(
            dgg_tile_latitude_axis,
            dgg_tile_longitude_axis,
            tiled_data_d,
            metadata_d,
        )

        # main_ads_product
        # Fill the cal_ab_coverage_per_tile object with proper types
        list_cal_ab_coverage_per_tile = []
        for cal_ab_tile_id in list(cal_ab_coverage_per_tile):
            cal_ab_original_coverage = common_annotation_models_l2.PercentPixelsType(
                percentage=cal_ab_coverage_per_tile[cal_ab_tile_id]["original_coverage"][0],
                pixels=cal_ab_coverage_per_tile[cal_ab_tile_id]["original_coverage"][1],
            )

            cal_ab_filtered_coverage = common_annotation_models_l2.CalAbfilteredCoverageType(
                after_global_filtering=common_annotation_models_l2.PercentPixelsType(
                    percentage=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_global_filtering"][0],
                    pixels=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_global_filtering"][1],
                ),
                after_agbvalue_filtering=common_annotation_models_l2.PercentPixelsType(
                    percentage=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_agbvalue_filtering"][0],
                    pixels=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_agbvalue_filtering"][1],
                ),
                after_agbstd_filtering=common_annotation_models_l2.PercentPixelsType(
                    percentage=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_agbstd_filtering"][0],
                    pixels=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_agbstd_filtering"][1],
                ),
                after_agbrelative_std_filtering=common_annotation_models_l2.PercentPixelsType(
                    percentage=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_agbrelative_std_filtering"][0],
                    pixels=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_agbrelative_std_filtering"][1],
                ),
                after_lcmclass_filtering=common_annotation_models_l2.PercentPixelsType(
                    percentage=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_lcmclass_filtering"][0],
                    pixels=cal_ab_coverage_per_tile[cal_ab_tile_id]["after_lcmclass_filtering"][1],
                ),
            )

            list_cal_ab_coverage_per_tile.append(
                common_annotation_models_l2.CalAbcoverageType(
                    original_coverage=cal_ab_original_coverage,
                    filtered_coverage=cal_ab_filtered_coverage,
                    id=cal_ab_tile_id,
                )
            )
        cal_ab_coverage_per_tile_obj = common_annotation_models_l2.CalAbcoverageTilesListType(
            list_cal_ab_coverage_per_tile
        )

        # Fill the gn_coverage_per_tile object with proper types
        list_gn_coverage_per_tile = []
        for gn_coverage_tile_id in list(gn_coverage_per_tile):
            list_gn_coverage_one_tile = []
            for gn_product in self.l2a_gn_products_list:
                name = gn_product.name

                gn_original_coverage = common_annotation_models_l2.PercentPixelsType(
                    percentage=gn_coverage_per_tile[gn_coverage_tile_id]["original_coverage"][name][0],
                    pixels=gn_coverage_per_tile[gn_coverage_tile_id]["original_coverage"][name][1],
                )

                gn_filtered_coverage = common_annotation_models_l2.GnfilteredCoverageType(
                    after_global_filtering=common_annotation_models_l2.PercentPixelsType(
                        percentage=gn_coverage_per_tile[gn_coverage_tile_id]["after_global_filtering"][name][0],
                        pixels=gn_coverage_per_tile[gn_coverage_tile_id]["after_global_filtering"][name][1],
                    ),
                    after_sigma_filtering=common_annotation_models_l2.PercentPixelsType(
                        percentage=gn_coverage_per_tile[gn_coverage_tile_id]["after_sigma_filtering"][name][0],
                        pixels=gn_coverage_per_tile[gn_coverage_tile_id]["after_sigma_filtering"][name][1],
                    ),
                    after_angle_filtering=common_annotation_models_l2.PercentPixelsType(
                        percentage=gn_coverage_per_tile[gn_coverage_tile_id]["after_angle_filtering"][name][0],
                        pixels=gn_coverage_per_tile[gn_coverage_tile_id]["after_angle_filtering"][name][1],
                    ),
                )

                list_gn_coverage_one_tile.append(
                    common_annotation_models_l2.GncoverageType(
                        original_coverage=gn_original_coverage,
                        filtered_coverage=gn_filtered_coverage,
                        acquisition_id=name,
                    )
                )
            list_gn_coverage_per_tile.append(
                common_annotation_models_l2.GncoverageListType(gn=list_gn_coverage_one_tile, id=gn_coverage_tile_id)
            )

        gn_coverage_per_tile_obj = common_annotation_models_l2.GncoverageTilesListType(tile=list_gn_coverage_per_tile)

        mission = common_annotation_models_l2.MissionType.BIOMASS.value
        sensor_mode = common_annotation_models_l2.SensorModeType.MEASUREMENT.value
        main_ads_product = BIOMASSL2bAGBMainADSproduct(
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
            contributing_tiles,
            cal_ab_coverage_per_tile_obj,
            gn_coverage_per_tile_obj,
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
            "agb": common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_T_HA,
            "agb_standard_deviation": common_types.PixelRepresentationType.ABOVE_GROUND_BIOMASS_QUALITY_T_HA,
            "heat_map": [common_types.PixelRepresentationType.HEAT_MAP],
            "bps_fnf": common_types.PixelRepresentationType.COMPUTED_FOREST_MASK,
            "acquisition_id_image": common_types.PixelRepresentationType.ACQUISITION_ID_IMAGE,
        }
        pixel_representation = common_annotation_models_l2.PixelRepresentationChoiceType(
            agb=pixel_representation_d["agb"],
            agb_standard_deviation=pixel_representation_d["agb_standard_deviation"],
            agb_heat_map=pixel_representation_d["heat_map"],
            bps_fnf=pixel_representation_d["bps_fnf"],
            acquisition_id_image=pixel_representation_d["acquisition_id_image"],
        )
        pixel_type = common_annotation_models_l2.PixelTypeChoiceType(
            float_pixel_type=main_annotation_models_l2b_agb.PixelTypeType("32 bit Float"),
            int_pixel_type=main_annotation_models_l2b_agb.PixelTypeType("8 bit Unsigned Integer"),
        )

        no_data_value = common_annotation_models_l2.NoDataValueChoiceType(
            float_no_data_value=FLOAT_NODATA_VALUE,
            int_no_data_value=INT_NODATA_VALUE,
        )
        main_ads_raster_image = BIOMASSL2bAGBMainADSRasterImage(
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

        # Estimated Parameters
        num_pols = len(estimated_parameters["polarization_names"])
        num_dates = len(estimated_parameters["dates_values"])
        num_classes = len(estimated_parameters["classes_values"])

        polarization_list = []
        for pol_idx in range(num_pols):
            lcm_list = []
            for class_idx in range(num_classes):
                date_list = []
                for date_idx in range(num_dates):
                    date_list.append(
                        main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date(
                            n=main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date.N(
                                mean=float(estimated_parameters["n_unknowns"][pol_idx, date_idx, class_idx]),
                                std=float(estimated_parameters["n_stddev"][pol_idx, date_idx, class_idx]),
                            ),
                            a=main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date.A(
                                mean=float(estimated_parameters["alfa_unknowns"][pol_idx, date_idx, class_idx]),
                                std=float(estimated_parameters["alfa_stddev"][pol_idx, date_idx, class_idx]),
                            ),
                            l=main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date.L(
                                mean=float(estimated_parameters["L_unknowns"][pol_idx, date_idx, class_idx]),
                                std=float(estimated_parameters["L_stddev"][pol_idx, date_idx, class_idx]),
                            ),
                            dates=estimated_parameters["dates_values"][date_idx],
                        ),
                    )

                lcm_list.append(
                    main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm(
                        date=date_list,
                        classes=estimated_parameters["classes_values"][class_idx],
                    )
                )
            polarization_list.append(
                main_annotation_models_l2b_agb.EstimatedParametersPolarisationType(
                    lcm=lcm_list,
                    polarisations=estimated_parameters["polarization_names"][pol_idx],
                )
            )

        estimated_parameters = main_annotation_models_l2b_agb.EstimatedParametersL2BAgb(
            rho=self.rho,
            polarisation=polarization_list,
        )

        compression_options_agb = main_annotation_models_l2b_agb.CompressionOptionsL2B(
            main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds(
                main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.Agb(
                    self.aux_pp2_ab.compression_options.mds.AGB.compression_factor,
                    self.aux_pp2_ab.compression_options.mds.AGB.max_z_error,
                ),
                main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.AgbstandardDeviation(
                    self.aux_pp2_ab.compression_options.mds.AGBstandardDeviation.compression_factor,
                    self.aux_pp2_ab.compression_options.mds.AGBstandardDeviation.max_z_error,
                ),
                main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.BpsFnf(
                    self.aux_pp2_ab.compression_options.mds.bps_fnf.compression_factor
                ),
                main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.HeatMap(
                    self.aux_pp2_ab.compression_options.mds.heatmap.compression_factor
                ),
                main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.AcquisitionIdImage(
                    self.aux_pp2_ab.compression_options.mds.acquisition_id_image.compression_factor
                ),
            ),
            self.aux_pp2_ab.compression_options.mds_block_size,
        )

        main_ads_processing_parameters = BIOMASSL2bMainADSProcessingParametersAGB(
            bps.l2b_agb_processor.__version__,
            self.start_time,
            forest_masking_flag=self.aux_pp2_ab.forest_masking_flag,
            minumum_l2a_coverage=self.aux_pp2_ab.minimumL2acoverage,
            rejected_landcover_classes=self.aux_pp2_ab.rejected_landcover_classes,
            backscatter_limits=self.aux_pp2_ab.backscatterLimits,
            angle_limits=self.aux_pp2_ab.angleLimits,
            mean_agblimits=self.aux_pp2_ab.meanAGBLimits,
            std_agblimits=self.aux_pp2_ab.stdAGBLimits,
            relative_agblimits=self.aux_pp2_ab.relativeAGBLimits,
            reference_selection=self.aux_pp2_ab.referenceSelection,
            indexing_l=self.aux_pp2_ab.indexingL,
            indexing_a=self.aux_pp2_ab.indexingA,
            indexing_n=self.aux_pp2_ab.indexingN,
            use_constant_n=self.aux_pp2_ab.useConstantN,
            values_constant_n=self.aux_pp2_ab.valuesConstantN,
            regression_solver=self.aux_pp2_ab.regressionSolver,
            regression_matrix_subsampling_factor=self.regression_matrix_subsampling_factor,
            minimum_percentage_of_fillable_voids=self.aux_pp2_ab.minimumPercentageOfFillableVoids,
            estimated_parameters=estimated_parameters,
            compression_options=compression_options_agb,
        )

        # Initialize AGB Product
        product_to_write = BIOMASSL2bAGBProduct(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
            main_ads_processing_parameters,
            self.aux_pp2_ab.l2bAGBProductDOI,
            agb_tile_iteration,
            agb_tile_status,
        )

        # Write to file the AGB Product
        write_obj = BIOMASSL2bAGBProductWriter(
            product_to_write,
            self.product_path,
            bps.l2b_agb_processor.BPS_L2B_AGB_PROCESSOR_NAME,
            bps.l2b_agb_processor.__version__,
            footprint,
            [acq.name for acq in self.job_order.input_l2a_products],
            ground_corner_points(dgg_tile_latitude_axis, dgg_tile_longitude_axis),
            self.job_order.aux_pp2_agb_path.name,
            self.job_order.cal_ab_product.name,
            self.job_order.lcm_product.name,
            start_time_l2a,
            stop_time_l2a,
            footprint_mask_for_quicklooks,
            (
                [acq.name for acq in self.job_order.input_l2b_fd_products]
                if self.job_order.input_l2b_fd_products
                else None
            ),
            (
                [acq.name for acq in self.job_order.input_l2b_agb_products]
                if self.job_order.input_l2b_agb_products
                else None
            ),
        )
        write_obj.write()

    def run_l2b_agb_processing(self):
        """Performs processing as described in job order.

        Parameters
        ----------

        Returns
        -------
        """

        self._initialize_processing()

        (
            l2b_data_final_per_tile,
            l2a_input_tiles_id,
            mission_phase_id,
            global_coverage_id,
            footprint_per_tile,
            main_ads_input_information,
            skip_agb_computation,
            start_time_l2a,
            stop_time_l2a,
            agb_tile_iteration,
            agb_tile_status,
            estimated_parameters,
            contributing_tiles,
            cal_ab_coverage_per_tile,
            gn_coverage_per_tile,
            footprint_mask_for_quicklooks,
        ) = self._core_processing()

        if not skip_agb_computation:
            assert l2b_data_final_per_tile is not None
            num_tiles = len(l2b_data_final_per_tile.keys())
            for idx, (tile_id, l2b_data_final_dict) in enumerate(l2b_data_final_per_tile.items()):
                bps_logger.info(f"AGB writing tile {idx + 1} of {num_tiles}, {tile_id}")
                self._write_to_output(
                    l2b_data_final_dict,
                    tile_id,
                    mission_phase_id,
                    global_coverage_id,
                    footprint_per_tile[tile_id],
                    main_ads_input_information,
                    start_time_l2a,
                    stop_time_l2a,
                    agb_tile_iteration[tile_id],
                    agb_tile_status[tile_id],
                    estimated_parameters,
                    contributing_tiles,
                    cal_ab_coverage_per_tile,
                    gn_coverage_per_tile,
                    footprint_mask_for_quicklooks,
                )

    def _get_polarisation_names(self):
        if "p" in self.aux_pp2_ab.indexingL + self.aux_pp2_ab.indexingA + self.aux_pp2_ab.indexingN:
            # At least one parameter depends on polarization
            return [
                main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations.HH.value,
                main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations.VH.value,
                main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations.VV.value,
            ]
        else:
            # No parameters depend on class, so write all in a single element attribute
            return [main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations.HH_VH_VV.value]


def construct_5x5_data_object(
    l2a_gn_products_list: list[BIOMASSL2aProductGN],
    used_l2a_input_tiles_id: list[str],
    lcm_product: LCMMask,
    cal_agb_product_dict: dict[str, np.ndarray],
    minimumL2acoverage: float,
    l2b_agb_product_list: list[BIOMASSL2bAGBProduct] | None = None,
    l2b_fd_product_list: list[BIOMASSL2bFDProduct] | None = None,
) -> tuple[dict, dict, dict, bool]:
    """
    Construct the data object dictionary, containing all the needed AGB processor inputs for each available tile in the 5x5 grid

    Parameters
    ----------
    l2a_gn_products_list  = list[BIOMASSL2aProductGN]
        list of all the L2a GN products paths in input job order
    used_l2a_input_tiles_id: list[str]
        list of all the Tiles to be used from the ones in the l2a_gn_products_list
        (only the ones part of the 5x5 grid)
    lcm_product: LCMMask
        LCM, Land Cover Map, with its lat lon axes
    cal_agb_product_dict: dict[str, np.ndarray]
        Dictionary containing, for each tile in the keys,
        Reference AGB product 3d matrix, (num_lat x num_lon x 2)
        containing agb reference and standard deviation
    minimumL2acoverage: float
    l2b_agb_product_list: Optional[BIOMASSL2bAGBProduct]
        Optional L2B AGB product (Over green area)
    l2b_fd_product_list: Optional[BIOMASSL2bFDProduct]
        Optional L2B FD product (Over green + blue area, only INT phase)


    Returns
    -------
    data_object_5x5 = dict:
        5x5_object: dict[tile_id, 1x1_object] # (At most 25 dictionary tile_id keys, or less, if not found in input)

            1x1_object: dict[str, obj] # (one for each tile_id)
                    "latitude_axis": 1xM np.ndarray; dgg tile latitude vector [deg]
                    "longitude_axis": 1xN np.ndarray; dgg tile longitude vector [deg]
                    "bps_fnf": MxN np.ndarray; can be the FNF or the CFM (CFM if L2B FD is available for the tile)
                    "lcm": MxN np.ndarray; interpolated over the DGG tile, from the whole in input
                    "reference_map": MxNx2 np.ndarray or None; 2 maps, value and uncertainity, computed mixing CAL_AB and L2B_AGB; filled during data consolidation
                    "cal_ab": MxNx2 np.ndarray or None; 2 maps, value and uncertainity, from CAL_AB input: can be None
                    "l2b_agb" MxNx2 np.ndarray or None; 2 maps, value and uncertainity, from L2B AGB input: can be None
                    "agb_tile_iteration_updated": 1x1 int; agb tile iteration, from L2B AGB input: 1 if L2B AGB input is not present, else incremented by 1 from L2B AGB input value
                    "NANMASK": MxN np.ndarray: initial mask of nans in the input data, never updated, used at the end to update the agb_tile_status;
                                                filled during data consolidation; for each pixel, nan_mask is true only where the pixel in ALL the data and in all the polarizations is nan
                    "data_list": list[dict[str, dict]] (one list entry for each data falling in the tile_id)
                            one list entry: dict[str, dict]
                                "data": MxNx3 np.ndarray; input L2A GN data, 3 Polarizatons in the order: HH, VH, VV
                                "incidence_angle":MxN np.ndarray; from input L2A GN
                                "temporal_date": 1x1 double; sec85 date, from input L2A GN
                                "date_idx": 1x1 int; filled during data consolidation

    dgg_tile_footprint_per_tile: dict
        One key for each Tile ID, containing footprint
    gn_original_coverage_per_tile: dict
        Dictionary containing the original coverage of each GN for each used tile, to fill main annotation xml.
        Outside of this function, it needs to be completed with the coverages after filterings and with the remaining tiles of the 5x5
    skip_agb_computation: bool
        Flag to disable AGB computation if any of the input l2a_gn_products_list has enough coverage over the used_l2a_input_tiles_id
    """

    skip_agb_computation = False
    gn_original_coverage_per_tile = {}

    # Initlialize data object
    data_object_5x5 = {}  # one key for each tile, containing a data_object_1x1
    dgg_tile_footprint_per_tile = {}
    counter_skip = 0

    bps_logger.info(f"DGG tiling of L2a products into each of the {len(used_l2a_input_tiles_id)} tiles")
    bps_logger.info(f"    checking with AUX PP2 minimum L2a products coverage of {minimumL2acoverage}%")
    for tile_id in used_l2a_input_tiles_id:
        (
            skip_agb_computation_curr,  # Skip due to poor coverage
            data_3d_mat_dict_curr,  # [num_lat_dgg_tile, num_lon_dgg_tile, num_l2a_products]
            dgg_tile_latitude_axis_curr,
            dgg_tile_longitude_axis_curr,
            dgg_tile_footprint_curr,
        ) = dgg_tiling(
            l2a_gn_products_list,
            minimumL2acoverage,
            tile_id,
            L2B_OUTPUT_PRODUCT_AGB,
        )

        # Computation of original coverage for all the GN in input over the 5x5 tiles
        # This is needed to fill the main annotation xml dedicated section
        # Same dictionary, for the cal_ab is done outside this function
        gn_original_coverage_per_tile[tile_id] = {}
        for l2a_gn_product in l2a_gn_products_list:
            latlon_coverage = _get_l2a_cumulative_footprint([l2a_gn_product])

            lon_meshed, lat_meshed = np.meshgrid(dgg_tile_longitude_axis_curr, dgg_tile_latitude_axis_curr)
            (
                _,
                original_coverage_percentage,
                original_coverage_number,
            ) = _check_coverage(
                latlon_coverage,
                lat_meshed,
                lon_meshed,
            )

            gn_original_coverage_per_tile[tile_id][l2a_gn_product.name] = [
                float(original_coverage_percentage),
                int(original_coverage_number),
            ]

        # After tiling over current tile, save all in the dictionaries
        skip_agb_computation = skip_agb_computation and skip_agb_computation_curr

        if not skip_agb_computation_curr:
            dgg_tile_footprint_per_tile[tile_id] = dgg_tile_footprint_curr  # for product MPH and Main Annotaton writing

            list_entry = []
            for data_idx in np.arange(data_3d_mat_dict_curr["HH"].shape[2]):
                list_entry.append(
                    {
                        "data": np.moveaxis(
                            np.array(
                                [
                                    data_3d_mat_dict_curr["HH"][:, :, data_idx],
                                    data_3d_mat_dict_curr["VH"][:, :, data_idx],
                                    data_3d_mat_dict_curr["VV"][:, :, data_idx],
                                ]
                            ),
                            0,
                            2,  # to be MxNx3
                        ),
                        "acquisition_id": data_3d_mat_dict_curr["acquisition_ids"][data_idx],
                        "incidence_angle": data_3d_mat_dict_curr["local_incidence_angle"][:, :, data_idx],
                        "temporal_date": data_3d_mat_dict_curr["temporal_date_sec85"][data_idx],
                        "date_idx": 0,  # initilaization only
                    }
                )

            # References management (here, fill both CAL_AB and AGB, they are mergedf in consolidation function)
            cal_ab = cal_agb_product_dict[tile_id]["cal_ab"] if tile_id in cal_agb_product_dict.keys() else None

            l2b_agb = (
                [
                    l2b_agb_product
                    for l2b_agb_product in l2b_agb_product_list
                    if l2b_agb_product.main_ads_product.tile_id_list[0] == tile_id
                ]
                if l2b_agb_product_list is not None
                else None
            )

            if l2b_agb is not None and len(l2b_agb) > 1:
                raise ValueError(f"    optional L2B AGB products have more than one product over tile {tile_id}")

            agb_tile_iteration = np.uint8(1)
            if l2b_agb is not None and len(l2b_agb) == 0:
                l2b_agb = None

            elif l2b_agb is not None:
                agb_tile_iteration = l2b_agb[0].agb_tile_iteration
                agb_tile_status = l2b_agb[0].agb_tile_status.value
                l2b_agb = np.moveaxis(
                    np.array(
                        [
                            l2b_agb[0].measurement.data_dict["agb"],
                            l2b_agb[0].measurement.data_dict["agbstandard_deviation"],
                        ]
                    ),
                    0,
                    2,
                )

                bps_logger.info("        Found optional input L2B AGB in this tile, with:")
                bps_logger.info(f"            Tile Status: {agb_tile_status}")
                bps_logger.info(f"            Tile Iteration: {agb_tile_iteration}")

                # Increase the iteration over the tile
                agb_tile_iteration = agb_tile_iteration + 1

            # LCM is one for all the inputs, interpolation over the tile axis needed
            # interpolator Nearest Neigbour
            lcm_mask = _interpolate_lcm_over_tile(
                lcm_product,
                dgg_tile_latitude_axis_curr,
                dgg_tile_longitude_axis_curr,
            ).astype(np.int32)

            # Manage BPS_FNF, CFM or FNF
            cfm = (
                [
                    l2b_fd_product.measurement.data_dict["cfm"][
                        ::2, ::2
                    ]  # FD is more doubled samples than the other processors
                    for l2b_fd_product in l2b_fd_product_list
                    if l2b_fd_product.main_ads_product.tile_id_list[0] == tile_id
                ]
                if l2b_fd_product_list is not None
                else None
            )
            if cfm is not None and len(cfm) == 0:
                cfm = None
            if cfm is not None and len(cfm) > 1:
                raise ValueError(f"    optional L2B FD products have more than one product over tile {tile_id}")
            elif cfm is not None:
                cfm = cfm[0]
                cfm = np.repeat(cfm[:, :, np.newaxis], data_3d_mat_dict_curr["HH"].shape[2], axis=2)

            data_object_1x1 = {
                "latitude_axis": dgg_tile_latitude_axis_curr,
                "longitude_axis": dgg_tile_longitude_axis_curr,
                "bps_fnf": data_3d_mat_dict_curr["bps_fnf"] if cfm is None else cfm,
                "lcm": lcm_mask,
                "reference_map": np.zeros(
                    (
                        len(dgg_tile_latitude_axis_curr),
                        len(dgg_tile_longitude_axis_curr),
                        2,
                    )
                ),  # Filled during consolidation, using cal_ab and l2b_agb:
                "cal_ab": cal_ab,  # Can be None
                "l2b_agb": l2b_agb,  # Can be None
                "agb_tile_iteration_updated": agb_tile_iteration,
                "NANMASK": None,
                "data_list": list_entry,
            }

            data_object_5x5[tile_id] = data_object_1x1

        else:
            counter_skip += 1

    if counter_skip:
        bps_logger.info(f"{counter_skip} DGG tiles skipped for not satisfying minimum L2a products coverage")
        num_remaining = len(data_object_5x5.keys())
        if num_remaining:
            bps_logger.info(
                f"Remaining {len(data_object_5x5.keys())} DGG tiles used for processing are the following: {list(data_object_5x5.keys())}"
            )
        else:
            bps_logger.info("    no tiles remaining for processing, due to not enough L2a coverage")
            skip_agb_computation = True

    return (
        data_object_5x5,
        dgg_tile_footprint_per_tile,
        gn_original_coverage_per_tile,
        skip_agb_computation,
    )


def consolidate_5x5_data_block(
    data_object_5x5: dict,
    referenceSelection: str,
    rejected_landcover_classes: IntArray,
    backscatterLimits: dict[str, MinMaxType],
    angleLimits: MinMaxTypeWithUnit,
    meanAGBLimits: MinMaxTypeWithUnit,
    stdAGBLimits: MinMaxTypeWithUnit,
    relativeAGBLimits: MinMaxType,
    forest_masking_flag: bool,
) -> tuple[dict, list[PreciseDateTime], dict, dict]:
    """
    Consolidating here:
        Reference Data: filtering and migxing CAL_AB with L2B AGB
        Data (all data in each Tile, for all polarizations ); rejecting values out of AUX PP limits and rejected_landcover_classes
        Dates: from input dates in sec85, create date indices
        Reference Data, Data and incidence angle: converted in decibels 10*log10(), this is needed by the inversion algorithm

        Note. LCM consolidation is performed in another section (see build_reference_vectors)
    """

    bps_logger.info("Data consolidation")

    ### 1/3) Reference consolidation (CAL_AB with L2B AGB)
    bps_logger.info("    Reference data consolidation (filtering and mixing)")
    bps_logger.info(
        f"    using mean AGB limits min={meanAGBLimits.min.value}, max={meanAGBLimits.max.value} as specified in AUX PP2 AB"
    )
    bps_logger.info(
        f"    using std AGB limits min={stdAGBLimits.min.value}, max={stdAGBLimits.max.value} as specified in AUX PP2 AB"
    )
    bps_logger.info(
        f"    using relative AGB limits limits min={relativeAGBLimits.min}, max={relativeAGBLimits.max} as specified in AUX PP2 AB"
    )

    # Computation of filtered coverage for the cal_ab
    # This is needed to fill the main annotation xml dedicated section
    (
        cal_ab_after_global_filtering_per_tile,
        cal_ab_after_agbvalue_filtering_per_tile,
        cal_ab_after_agbstd_filtering_per_tile,
        cal_ab_after_agbrelative_std_filtering_per_tile,
        cal_ab_after_lcmclass_filtering_per_tile,
    ) = compute_filtered_coverage_cal_ab(
        data_object_5x5,
        rejected_landcover_classes,
        meanAGBLimits,
        stdAGBLimits,
        relativeAGBLimits,
    )

    # CAL_AB global flter for the AGB algorithm
    for tile_id, data_object_1x1 in data_object_5x5.items():  # Cycling each Tile
        bps_logger.info(f"        Tile {tile_id}")
        # First filter
        if data_object_1x1["cal_ab"] is not None:
            bps_logger.info(
                f"            CAL_AB valid samples before filtering: {np.sum(np.isfinite(data_object_1x1['cal_ab'][:, :, 0]))}"
            )
            # CAL_AB global flter for the AGB algorithm
            cal_ab_mean, cal_ab_std = filter_agb_reference_global(
                agb_mean=data_object_1x1["cal_ab"][:, :, 0],
                agb_std=data_object_1x1["cal_ab"][:, :, 1],
                mean_min=meanAGBLimits.min.value,
                mean_max=meanAGBLimits.max.value,
                std_min=stdAGBLimits.min.value,
                std_max=stdAGBLimits.max.value,
                rel_std_min=relativeAGBLimits.min,
                rel_std_max=relativeAGBLimits.max,
            )
            bps_logger.info(f"            CAL_AB valid samples after filtering: {np.sum(np.isfinite(cal_ab_mean))}")

            # "cal_ab" filed no more needed, it will be updated by "reference_map", which is the mix of cal_ab and l2b_agb:
            #   free this space, do not put None, for further checks of "data_object_1x1["cal_ab"] is None"
            data_object_1x1["cal_ab"] = -1

        else:
            bps_logger.info("            CAL_AB not present")
            # Initialize matrices of nan, with correct dimension, for mix_agb_sources()
            cal_ab_mean = (
                np.zeros(
                    (
                        len(data_object_1x1["latitude_axis"]),
                        len(data_object_1x1["longitude_axis"]),
                    )
                )
                * np.nan
            )
            cal_ab_std = (
                np.zeros(
                    (
                        len(data_object_1x1["latitude_axis"]),
                        len(data_object_1x1["longitude_axis"]),
                    )
                )
                * np.nan
            )

        if data_object_1x1["l2b_agb"] is not None:
            bps_logger.info(
                f"            L2B AGB valid samples before filtering: {np.sum(np.isfinite(data_object_1x1['l2b_agb'][:, :, 0]))}"
            )
            l2b_agb_mean, l2b_agb_std = filter_agb_reference_global(
                agb_mean=data_object_1x1["l2b_agb"][:, :, 0],
                agb_std=data_object_1x1["l2b_agb"][:, :, 1],
                mean_min=meanAGBLimits.min.value,
                mean_max=meanAGBLimits.max.value,
                std_min=stdAGBLimits.min.value,
                std_max=stdAGBLimits.max.value,
                rel_std_min=relativeAGBLimits.min,
                rel_std_max=relativeAGBLimits.max,
            )
            bps_logger.info(f"            L2B AGB valid samples after filtering: {np.sum(np.isfinite(l2b_agb_mean))}")

        else:
            bps_logger.info("            L2B AGB not present")
            # Initialize matrices of nan, with correct dimension, for mix_agb_sources()
            l2b_agb_mean = (
                np.zeros(
                    (
                        len(data_object_1x1["latitude_axis"]),
                        len(data_object_1x1["longitude_axis"]),
                    )
                )
                * np.nan
            )
            l2b_agb_std = (
                np.zeros(
                    (
                        len(data_object_1x1["latitude_axis"]),
                        len(data_object_1x1["longitude_axis"]),
                    )
                )
                * np.nan
            )

        # Mixing CAL_AB with L2B AGB, using referenceSelection regula
        (
            data_object_1x1["reference_map"][:, :, 0],
            data_object_1x1["reference_map"][:, :, 1],
        ) = mix_agb_sources(
            cal_ab_mean,
            cal_ab_std,
            l2b_agb_mean,
            l2b_agb_std,
            mixing=referenceSelection,
        )
        if np.logical_not(np.logical_and(np.all(cal_ab_mean) == np.nan, np.all(l2b_agb_mean) == np.nan)):
            bps_logger.info(
                f"            Valid reference samples after mixing: {np.sum(np.isfinite(data_object_1x1['reference_map'][:, :, 0]))}"
            )
        else:
            bps_logger.info("            No references available for the tile")

        # Convert reference map in decibels:
        data_object_1x1["reference_map"] = 10 * np.log10(data_object_1x1["reference_map"])

    ### 2/3) Data consolidation
    bps_logger.info("    Input data consolidation")
    bps_logger.info(
        f"    rejecting following landcover classes {rejected_landcover_classes} as specified in AUX PP2 AB"
    )
    bps_logger.info(
        f"    using backscatter limits min_hh={backscatterLimits['HH'].min}, max_hh={backscatterLimits['HH'].max} min_vh={backscatterLimits['HV'].min}, max_vh={backscatterLimits['HV'].max} min_vv={backscatterLimits['VV'].min}, max_vv={backscatterLimits['VV'].max} as specified in AUX PP2 AB"
    )
    bps_logger.info(
        f"    using angles limits min={angleLimits.min.value}, max={angleLimits.max.value} as specified in AUX PP2 AB"
    )
    bps_logger.info(f"    FNF masking flag set to {forest_masking_flag} as specified in AUX PP2 AB")

    # Computation of filtered coverage for the GN, external initialization
    # This is needed to fill the main annotation xml dedicated section
    gn_after_global_filtering_per_tile = {}
    gn_after_sigma_filtering_per_tile = {}
    gn_after_angle_filtering_per_tile = {}
    for tile_id, data_object_1x1 in data_object_5x5.items():  # Cycling each Tile
        bps_logger.info(f"        Tile {tile_id}")

        # Computation of filtered coverage for the GN, internal initialization for the tile
        gn_after_global_filtering_per_tile[tile_id] = {}
        gn_after_sigma_filtering_per_tile[tile_id] = {}
        gn_after_angle_filtering_per_tile[tile_id] = {}

        # Partial computations of filtered coverage for the GN, using gn_temp variable
        # Partial computation filtered coverage for main annotation
        for data_idx, list_entry in enumerate(data_object_1x1["data_list"]):
            gn_temp = np.copy(list_entry["data"])
            for pol_idx, pol_name in enumerate(backscatterLimits.keys()):  # Cycling each polarization
                gn_temp[:, :, pol_idx] = filter_gn_core(
                    gn_temp[:, :, pol_idx],
                    gn_temp[:, :, pol_idx],
                    backscatterLimits[pol_name].min,
                    backscatterLimits[pol_name].max,
                )

            gn_after_sigma_filtering_per_tile[tile_id][list_entry["acquisition_id"]] = (
                _compute_coverage_percentage_and_pixels(np.sum(gn_temp, axis=2))
            )

        # Partial computation filtered coverage for main annotation
        for data_idx, list_entry in enumerate(data_object_1x1["data_list"]):
            gn_temp = np.copy(list_entry["data"])
            for pol_idx, pol_name in enumerate(backscatterLimits.keys()):  # Cycling each polarization
                gn_temp[:, :, pol_idx] = filter_gn_core(
                    gn_temp[:, :, pol_idx],
                    list_entry["incidence_angle"],
                    np.rad2deg(angleLimits.min.value),
                    np.rad2deg(angleLimits.max.value),
                )

            gn_after_angle_filtering_per_tile[tile_id][list_entry["acquisition_id"]] = (
                _compute_coverage_percentage_and_pixels(np.sum(gn_temp, axis=2))
            )

        # GN global flter for the AGB algorithm
        for data_idx, list_entry in enumerate(data_object_1x1["data_list"]):
            for pol_idx, pol_name in enumerate(backscatterLimits.keys()):  # Cycling each polarization
                bps_logger.info(f"            Input data #{data_idx + 1}, {pol_name}")
                bps_logger.info(
                    f"                valid samples before filtering: {np.sum(np.isfinite(list_entry['data'][:, :, pol_idx]))}"
                )

                list_entry["data"][:, :, pol_idx] = filter_gn_core(
                    list_entry["data"][:, :, pol_idx],
                    list_entry["data"][:, :, pol_idx],
                    backscatterLimits[pol_name].min,
                    backscatterLimits[pol_name].max,
                )

                list_entry["data"][:, :, pol_idx] = filter_gn_core(
                    list_entry["data"][:, :, pol_idx],
                    list_entry["incidence_angle"],
                    np.rad2deg(angleLimits.min.value),
                    np.rad2deg(angleLimits.max.value),
                )

                list_entry["data"][:, :, pol_idx] = filter_for_lcm_class(
                    list_entry["data"][:, :, pol_idx],
                    data_object_1x1["lcm"],
                    list(rejected_landcover_classes),
                )

                bps_logger.info(
                    f"                valid samples after filtering for LCM values: {np.sum(np.isfinite(list_entry['data'][:, :, pol_idx]))}"
                )

                # Global computation filtered coverage for main annotation
                gn_after_global_filtering_per_tile[tile_id][list_entry["acquisition_id"]] = (
                    _compute_coverage_percentage_and_pixels(np.sum(list_entry["data"], axis=2))
                )

                if forest_masking_flag:
                    if len(data_object_1x1["bps_fnf"].shape) == 3:
                        # Logical OR of the masks
                        whole_fnf_mask = deepcopy(data_object_1x1["bps_fnf"][:, :, 0])
                        whole_fnf_mask[whole_fnf_mask == INT_NODATA_VALUE] = 0

                        num_fnf = data_object_1x1["bps_fnf"].shape[2]
                        for mask_idx in range(num_fnf):
                            curr_fnf_mask = deepcopy(data_object_1x1["bps_fnf"][:, :, mask_idx])
                            curr_fnf_mask[curr_fnf_mask == INT_NODATA_VALUE] = 0

                            whole_fnf_mask = np.logical_or(
                                whole_fnf_mask,
                                curr_fnf_mask,
                            ).astype(np.uint8)

                    else:
                        whole_fnf_mask = data_object_1x1["bps_fnf"]

                    list_entry["data"][:, :, pol_idx] = np.where(
                        whole_fnf_mask == 0,
                        np.nan,
                        list_entry["data"][:, :, pol_idx],
                    )

                    bps_logger.info(
                        f"                valid samples after filtering for BPS FNF: {np.sum(np.isfinite(list_entry['data'][:, :, pol_idx]))}"
                    )

        # Compute nan_mask after data consolitation, for the current tile:
        # for each pixel, nan_mask is true only where the pixel in ALL the data and in all the polarizations is nan
        nan_mask = np.ones(
            (
                len(data_object_1x1["latitude_axis"]),
                len(data_object_1x1["longitude_axis"]),
            ),
            dtype=bool,
        )
        for list_entry in data_object_1x1["data_list"]:
            nan_mask = np.logical_and(
                nan_mask, np.isnan(list_entry["data"]).all(axis=2)
            )  # axis)=2 is to check all 3 polarizations
        data_object_1x1["NANMASK"] = nan_mask

    # 3/3) Date consolidation
    bps_logger.info("    Date consolidation (sorting)")
    # First fill a vector with all the dates
    dates_sec85_vector = []
    for data_object_1x1 in data_object_5x5.values():  # Cycling each Tile
        for list_entry in data_object_1x1["data_list"]:  # Cycling each data in the Tile
            dates_sec85_vector.append(list_entry["temporal_date"])

    # Sort the vector of dates
    dates_sec85_vector = np.sort(dates_sec85_vector)
    dates_relative_vector = dates_sec85_vector - dates_sec85_vector[0]

    date_idx = np.zeros(len(dates_relative_vector)).astype(np.int32)
    ref_date = np.int32(0)
    previous_date_idx = 0
    different_dates_mjd = []
    for idx, relative_date in enumerate(dates_relative_vector):
        if idx == 0:
            different_dates_mjd.append(
                PreciseDateTime.from_sec85(relative_date + dates_sec85_vector[0]).isoformat(timespec="milliseconds")[
                    :-1
                ]
            )
            continue
        # Check if current date minus previous date is > one day
        if relative_date - dates_relative_vector[previous_date_idx] > 86400:
            ref_date += 1
            different_dates_mjd.append(
                PreciseDateTime.from_sec85(relative_date + dates_sec85_vector[0]).isoformat(timespec="milliseconds")[
                    :-1
                ]
            )

        previous_date_idx += 1
        date_idx[idx] = ref_date

    bps_logger.info(f"        Different dates for parameters estimation: {len(different_dates_mjd)}:")
    for date_mjd in different_dates_mjd:
        bps_logger.info(f"            {date_mjd}")

    # Finalizing data_object_5x5:
    #   > Fill data_object_5x5 with date_idx, to be placed in the correct position
    #   > convert in decibels the data and incidence angle
    for data_object_1x1 in data_object_5x5.values():  # Cycling each Tile
        for list_entry in data_object_1x1["data_list"]:  # Cycling each data in the Tile
            # Fill data_object_5x5 with date_idx, to be placed in the correct position
            for idx, date in enumerate(dates_sec85_vector):
                if list_entry["temporal_date"] == date:
                    list_entry["date_idx"] = date_idx[idx]

            # convert in decibels the data and incidence angle
            list_entry["data"] = 10 * np.log10(list_entry["data"])
            list_entry["incidence_angle"] = 10 * np.log10(
                np.cos(np.deg2rad(list_entry["incidence_angle"]))
            )  # dB for incidence angle is for the AGB inversion

    # fill one dictionary for the output, contaning all the sub-dictionaries
    # for the cal_ab and gn filtered coverage (for main annotation)
    cal_ab_filtered_coverage_per_tile = {}
    gn_filtered_coverage_per_tile = {}
    for tile_id in cal_ab_after_global_filtering_per_tile.keys():
        cal_ab_filtered_coverage_per_tile[tile_id] = {}

        cal_ab_filtered_coverage_per_tile[tile_id]["after_global_filtering"] = cal_ab_after_global_filtering_per_tile[
            tile_id
        ]
        cal_ab_filtered_coverage_per_tile[tile_id]["after_agbvalue_filtering"] = (
            cal_ab_after_agbvalue_filtering_per_tile[tile_id]
        )
        cal_ab_filtered_coverage_per_tile[tile_id]["after_agbstd_filtering"] = cal_ab_after_agbstd_filtering_per_tile[
            tile_id
        ]
        cal_ab_filtered_coverage_per_tile[tile_id]["after_agbrelative_std_filtering"] = (
            cal_ab_after_agbrelative_std_filtering_per_tile[tile_id]
        )

        cal_ab_filtered_coverage_per_tile[tile_id]["after_lcmclass_filtering"] = (
            cal_ab_after_lcmclass_filtering_per_tile[tile_id]
        )

    for tile_id in gn_after_global_filtering_per_tile.keys():
        gn_filtered_coverage_per_tile[tile_id] = {}

        gn_filtered_coverage_per_tile[tile_id]["after_global_filtering"] = gn_after_global_filtering_per_tile[tile_id]
        gn_filtered_coverage_per_tile[tile_id]["after_sigma_filtering"] = gn_after_sigma_filtering_per_tile[tile_id]
        gn_filtered_coverage_per_tile[tile_id]["after_angle_filtering"] = gn_after_angle_filtering_per_tile[tile_id]

    return (
        data_object_5x5,
        different_dates_mjd,
        cal_ab_filtered_coverage_per_tile,
        gn_filtered_coverage_per_tile,
    )


def build_reference_vectors(
    data_object_5x5: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    vector_dict = {
        "AGB_mean": np.array([], dtype=np.float32),
        "AGB_std": np.array([], dtype=np.float32),
        "point_idx": np.array([], dtype=np.int32),
        "pol_idx": np.array([], dtype=np.int32),
        "LCM_idx": np.array([], dtype=np.int32),
        "date_idx": np.array([], dtype=np.int32),
        "inc_angle": np.array([], dtype=np.float32),
        "sigma": np.array([], dtype=np.float32),
    }

    for tile_idx, data_object_1x1 in enumerate(data_object_5x5.values()):  # Cycling each Tile
        num_lat = len(data_object_1x1["latitude_axis"])
        num_lon = len(data_object_1x1["longitude_axis"])
        data_size = num_lat * num_lon

        # For AGB equation 4.22
        point_idx = np.arange(data_size) + tile_idx * data_size

        for list_entry in data_object_1x1["data_list"]:  # Cycling each data in the Tile
            for pol_idx in np.arange(3):
                mask_valid = np.logical_and(
                    np.isfinite(data_object_1x1["reference_map"][:, :, 0]),  # mean AGB
                    np.isfinite(list_entry["data"][:, :, pol_idx]),
                ).flatten()

                # Flatten each vector, than
                # remove not finite (nan) elements

                # Sigma is the data, present for each polarization
                vector_dict["sigma"] = np.append(
                    vector_dict["sigma"],
                    list_entry["data"][:, :, pol_idx].flatten()[mask_valid],
                )

                # Following ones are copied equal, for each polarization of sigma:
                vector_dict["point_idx"] = np.append(vector_dict["point_idx"], point_idx.flatten()[mask_valid])
                vector_dict["AGB_mean"] = np.append(
                    vector_dict["AGB_mean"],
                    data_object_1x1["reference_map"][:, :, 0].flatten()[mask_valid],
                )
                vector_dict["AGB_std"] = np.append(
                    vector_dict["AGB_std"],
                    data_object_1x1["reference_map"][:, :, 1].flatten()[mask_valid],
                )
                vector_dict["inc_angle"] = np.append(
                    vector_dict["inc_angle"],
                    list_entry["incidence_angle"].flatten()[mask_valid],
                )
                vector_dict["LCM_idx"] = np.append(vector_dict["LCM_idx"], data_object_1x1["lcm"].flatten()[mask_valid])

                # Following ones are scalars, to be repeated for all the data size
                vector_dict["pol_idx"] = np.append(vector_dict["pol_idx"], np.repeat(pol_idx, data_size)[mask_valid])
                vector_dict["date_idx"] = np.append(
                    vector_dict["date_idx"],
                    np.repeat(list_entry["date_idx"].flatten(), data_size)[mask_valid],
                )

    # Consolidating here the LCM
    lcm_absolute_values = np.unique(vector_dict["LCM_idx"])

    lcm_abs_to_rel_indices = build_lcm_global_abs_to_rel(data_object_5x5)

    vector_dict["LCM_idx"] = lcm_abs_to_rel_indices[vector_dict["LCM_idx"]]

    states = lcm_abs_to_rel_indices > -1
    lcm_in_data = np.where(states)[0]
    bps_logger.info(f"    Land Cover Classes in data: {len(lcm_in_data)}; values: {lcm_in_data}")
    bps_logger.info(
        f"        of which: {len(lcm_absolute_values)} classes (values: {lcm_absolute_values}) are covered by reference AGB values"
    )

    return vector_dict, lcm_absolute_values, lcm_abs_to_rel_indices


def build_lcm_global_abs_to_rel(data_object_5x5):
    absolute_lcm_vec = np.array([], dtype=np.int32)
    for data_object_1x1 in data_object_5x5.values():  # Cycling each Tile
        data_mask = np.ones(data_object_1x1["data_list"][0]["incidence_angle"].shape, dtype=bool)

        for list_entry in data_object_1x1["data_list"]:
            data_mask[np.isnan(list_entry["incidence_angle"])] = 0

            absolute_lcm_vec = np.append(absolute_lcm_vec, data_object_1x1["lcm"][data_mask].flatten())

    return consolidate_lcm(absolute_lcm_vec)


def consolidate_lcm(lcm_values):
    # consolidates lcm classes present in reference data, as a 1D array 0...N

    lcm_map = np.zeros(NUM_LCM_CLASSES, dtype=np.int32)
    lcm_map[lcm_values] = 1
    lcm_map = np.cumsum(lcm_map) * lcm_map - 1

    return lcm_map


def compute_heat_maps(
    lat_n: int,
    lon_n: int,
    input_data_list: list[np.ndarray],
    denominator_eq_436: np.ndarray,
    agb_estimation_current: np.ndarray,
    agb_estimation_previous: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute Heat Maps

    Compute the three heat maps as described in ATBD (see Returns for details)
    And one acquisition id image

    Parameters
    ----------
    input_data_list: list[np.ndarray],
        input L2A GN data consolidated and falling into the tile, each list entry is a MxNx3 (3 polarizations)
    denominator_eq_436: np.ndarray
        See ATBD, is the denominator of equation 4.36
    agb_estimation_current: np.ndarray
        AGB estimation result at current run, MxN
    agb_estimation_previous: Optional[np.ndarray] = None,
        AGB estimation result at previous run, MxN if second run, None if first run

    Returns
    -------
    heat_maps_dict: dict
        Contains the three computed heat maps in a dictionary (they will be saved in the same geotiff as different layers)
        "heat_map": ATBD eq 4.40, is the denominator of ATBD eq. 4.36
        "heat_map_ref_data": pixel wise, how many times reference data are used for the current regressions
        "heat_map_additional_ref_data": pixel wise, how many times additional reference data (i.e., belonging to a set previously solved by regression) are used for the current regressions
    acquisition_id_image: np.ndarray
        the number of layers corresponds number of contributing L2a data, where i-th refers to the referred to the i-th L2a contributing data
        for each pixel and layer, it assumes 0 value if the i-th L2a selected image did not contribute to the j-th pixel of the L2b product, 1 otherwise.
    """

    # Three Heat Maps, one float, two integers
    heat_maps_dict = {}

    # Heat Map
    heat_maps_dict["heat_map"] = denominator_eq_436.astype(np.float32)

    # Heat Map reference data
    # Same for first and second cycle
    heat_maps_dict["heat_map_ref_data"] = np.where(np.isnan(agb_estimation_current), 0, 1).astype(np.uint8)

    # Heat Map additional reference data
    if agb_estimation_previous is None:
        # First cycle
        heat_maps_dict["heat_map_additional_ref_data"] = np.zeros((lat_n, lon_n), dtype=np.uint8)
    else:
        # Second cycle:
        heat_maps_dict["heat_map_additional_ref_data"] = np.where(
            np.logical_and(
                np.isnan(agb_estimation_previous),
                np.logical_not(np.isnan(agb_estimation_current)),
            ),
            1,
            0,
        ).astype(np.uint8)

    # Acquisition Id Image
    acquisition_id_image = np.zeros((lat_n, lon_n), dtype=np.uint8)
    for data_3_pols in input_data_list:
        acquisition_id_image = np.where(
            np.isnan(np.sum(data_3_pols, axis=2)),
            acquisition_id_image,
            acquisition_id_image + 1,
        )

    return heat_maps_dict, acquisition_id_image


def _build_neighborhood(central_tile, radius):
    ctile_lat = int(central_tile[0:3].replace("N", "+").replace("S", "-"))
    ctile_lon = int(central_tile[3:].replace("E", "+").replace("W", "-"))

    neighborhood = []

    for lat in range(ctile_lat - radius, ctile_lat + radius + 1):
        for lon in range(ctile_lon - radius, ctile_lon + radius + 1):
            if lat > 90 or lat < -90:
                continue
            lon = (lon + 180) % 360 - 180
            if lat >= 0 and lon >= 0:
                neighborhood.append(f"N{abs(lat):02d}E{abs(lon):03d}")
            elif lat < 0 and lon >= 0:
                neighborhood.append(f"S{abs(lat):02d}E{abs(lon):03d}")
            elif lat >= 0 and lon < 0:
                neighborhood.append(f"N{abs(lat):02d}W{abs(lon):03d}")
            else:
                neighborhood.append(f"S{abs(lat):02d}W{abs(lon):03d}")

    return neighborhood


def filter_for_lcm_class(data, lcm, rejected_landcover_classes: list):
    for rejected_landcover_class in rejected_landcover_classes:
        data = np.where(
            lcm == rejected_landcover_class,
            np.nan,
            data,
        )
    return data


def filter_gn_core(data, reference_data, min_limit, max_limit):
    return np.where(
        np.logical_or(
            reference_data < min_limit,
            reference_data > max_limit,
        ),
        np.nan,
        data,
    )


def filter_agb_reference_global(agb_mean, agb_std, mean_min, mean_max, std_min, std_max, rel_std_min, rel_std_max):
    """
    Global filtering the AGB reference (CAL_AB or L2B_AGB)
    The agb (mean and std values) will be filtered (placing no data value) when value is outside of:
        mean_min, mean_max: mean AGB values
        std_min, std_max: AGB std values
        rel_std_min, rel_std_max: relative AGB std values
    """

    agb_rel_std = agb_std / agb_mean
    mask = np.where(
        np.isnan(agb_mean)
        | (agb_mean < mean_min)
        | (agb_mean > mean_max)
        | (agb_std < std_min)
        | (agb_std > std_max)
        | (agb_rel_std < rel_std_min)
        | (agb_rel_std > rel_std_max),
        np.nan,
        1,
    )

    agb_mean = agb_mean * mask
    agb_std = agb_std * mask
    return agb_mean, agb_std


def _filter_agb_reference_for_agb_values(
    agb_mean,
    mean_min,
    mean_max,
):
    """
    See filter_agb_reference_global()
    This internal function computes same filtering, but only for mean_min, mean_max: mean AGB values
    This function is needed to fill the main annotation xml dedicated section
    """

    mask = np.where(
        np.isnan(agb_mean) | (agb_mean < mean_min) | (agb_mean > mean_max),
        np.nan,
        1,
    )

    agb_mean = agb_mean * mask
    return agb_mean


def _filter_agb_reference_for_agb_std_values(
    agb_mean,
    agb_std,
    std_min,
    std_max,
):
    """
    See filter_agb_reference_global()
    This internal function computes same filtering, but only for std_min, std_max: AGB std values
    This function is needed to fill the main annotation xml dedicated section
    """

    mask = np.where(
        np.isnan(agb_mean) | (agb_std < std_min) | (agb_std > std_max),
        np.nan,
        1,
    )

    agb_mean = agb_mean * mask
    agb_std = agb_std * mask
    return agb_mean


def _filter_agb_reference_for_agb_relative_std_values(agb_mean, agb_std, rel_std_min, rel_std_max):
    """
    See filter_agb_reference_global()
    This internal function computes same filtering, but only for rel_std_min, rel_std_max: relative AGB std values
    This function is needed to fill the main annotation xml dedicated section
    """

    agb_rel_std = agb_std / agb_mean
    mask = np.where(
        np.isnan(agb_mean) | (agb_rel_std < rel_std_min) | (agb_rel_std > rel_std_max),
        np.nan,
        1,
    )

    agb_mean = agb_mean * mask
    return agb_mean


def compute_filtered_coverage_cal_ab(
    data_object_5x5,
    rejected_landcover_classes,
    meanAGBLimits,
    stdAGBLimits,
    relativeAGBLimits,
):
    # Computation of filtered coverage for the cal_ab
    # This is needed to fill the main annotation xml dedicated section
    cal_ab_after_global_filtering_per_tile = {}
    cal_ab_after_agbvalue_filtering_per_tile = {}
    cal_ab_after_agbstd_filtering_per_tile = {}
    cal_ab_after_agbrelative_std_filtering_per_tile = {}
    cal_ab_after_lcmclass_filtering_per_tile = {}
    for tile_id, data_object_1x1 in data_object_5x5.items():  # Cycling each Tile
        # First filter
        if data_object_1x1["cal_ab"] is not None:
            # Partial computations of filtered coverage for the cal_ab, using cal_ab_filtered_temp variable
            # Partial computation filtered coverage for main annotation
            cal_ab_filtered_temp = _filter_agb_reference_for_agb_values(
                agb_mean=data_object_1x1["cal_ab"][:, :, 0],
                mean_min=meanAGBLimits.min.value,
                mean_max=meanAGBLimits.max.value,
            )
            cal_ab_after_agbvalue_filtering_per_tile[tile_id] = _compute_coverage_percentage_and_pixels(
                cal_ab_filtered_temp
            )
            # Partial computation filtered coverage for main annotation
            cal_ab_filtered_temp = _filter_agb_reference_for_agb_std_values(
                agb_mean=data_object_1x1["cal_ab"][:, :, 0],
                agb_std=data_object_1x1["cal_ab"][:, :, 1],
                std_min=stdAGBLimits.min.value,
                std_max=stdAGBLimits.max.value,
            )
            cal_ab_after_agbstd_filtering_per_tile[tile_id] = _compute_coverage_percentage_and_pixels(
                cal_ab_filtered_temp
            )
            # Partial computation filtered coverage for main annotation
            cal_ab_filtered_temp = _filter_agb_reference_for_agb_relative_std_values(
                agb_mean=data_object_1x1["cal_ab"][:, :, 0],
                agb_std=data_object_1x1["cal_ab"][:, :, 1],
                rel_std_min=relativeAGBLimits.min,
                rel_std_max=relativeAGBLimits.max,
            )
            cal_ab_after_agbrelative_std_filtering_per_tile[tile_id] = _compute_coverage_percentage_and_pixels(
                cal_ab_filtered_temp
            )
            # Partial computation filtered coverage for main annotation
            cal_ab_filtered_temp = np.copy(data_object_1x1["cal_ab"][:, :, 0])
            cal_ab_filtered_temp = filter_for_lcm_class(
                cal_ab_filtered_temp,
                data_object_1x1["lcm"],
                list(rejected_landcover_classes),
            )
            cal_ab_after_lcmclass_filtering_per_tile[tile_id] = _compute_coverage_percentage_and_pixels(
                cal_ab_filtered_temp
            )

            # Partial computation filtered coverage for main annotation
            cal_ab_filtered_temp, _ = filter_agb_reference_global(
                agb_mean=data_object_1x1["cal_ab"][:, :, 0],
                agb_std=data_object_1x1["cal_ab"][:, :, 1],
                mean_min=meanAGBLimits.min.value,
                mean_max=meanAGBLimits.max.value,
                std_min=stdAGBLimits.min.value,
                std_max=stdAGBLimits.max.value,
                rel_std_min=relativeAGBLimits.min,
                rel_std_max=relativeAGBLimits.max,
            )

            # Global computation filtered coverage for main annotation
            cal_ab_after_global_filtering_per_tile[tile_id] = _compute_coverage_percentage_and_pixels(
                cal_ab_filtered_temp
            )

    return (
        cal_ab_after_global_filtering_per_tile,
        cal_ab_after_agbvalue_filtering_per_tile,
        cal_ab_after_agbstd_filtering_per_tile,
        cal_ab_after_agbrelative_std_filtering_per_tile,
        cal_ab_after_lcmclass_filtering_per_tile,
    )


def mix_agb_sources(cal_ab_mean, cal_ab_std, l2b_agb_mean, l2b_agb_std, mixing="refOnly"):
    """
     Parameters
     ----------
     cal_ab_mean: reference agb value
         numpy array, [NLat*NLon], NaN as invalid
     cal_ab_std: reference agb std
         numpy array, [NLat*NLon], NaN as invalid
     l2b_agb_mean: agb value from first cycle
         numpy array, [NLat*NLon], NaN as invalid
     l2b_agb_std: agb std from first cycle
         numpy array, [NLat*NLon], NaN as invalid
     mixing: configurable 'referenceSelection'
         str, valid values=[refOnly|firstIterationOnly|weightedMean]
    To be applied on already filtered reference data
    """

    if mixing == "refOnly":
        ref_agb_mean = np.where(np.isnan(cal_ab_mean), l2b_agb_mean, cal_ab_mean)
        ref_agb_std = np.where(np.isnan(cal_ab_mean), l2b_agb_std, cal_ab_std)
    elif mixing == "firstIterationOnly":
        ref_agb_mean = np.where(np.isnan(l2b_agb_mean), cal_ab_mean, l2b_agb_mean)
        ref_agb_std = np.where(np.isnan(l2b_agb_mean), cal_ab_std, l2b_agb_std)
    else:
        w1 = 1 / cal_ab_std**2
        w2 = 1 / l2b_agb_std**2
        den = np.nansum(np.vstack((w1, w2)), axis=0)
        ref_agb_mean = np.nansum(np.vstack((w1 * cal_ab_mean, w2 * l2b_agb_mean)), axis=0) / den
        ref_agb_std = np.sqrt(1 / den)
        ref_agb_mean = np.where(
            np.isnan(l2b_agb_mean * cal_ab_mean),
            cal_ab_mean if not np.all(np.isnan(cal_ab_mean)) else ref_agb_mean,
            ref_agb_mean if not np.all(np.isnan(ref_agb_mean)) else cal_ab_mean,
        )
        ref_agb_std = np.where(
            np.isnan(l2b_agb_mean * cal_ab_mean),
            cal_ab_std if not np.all(np.isnan(cal_ab_std)) else ref_agb_std,
            ref_agb_std if not np.all(np.isnan(ref_agb_std)) else cal_ab_std,
        )

    return ref_agb_mean, ref_agb_std


def get_5x5_tiles_ids(center_tile_id: str) -> list[str]:
    """Return the list of 5x5 Tile IDs grid, surrounding input central tile.

    Parameters
    ----------
    center_tile_id:str
        Tile Identifier used as central one in the 5x5 grid

    Returns
    -------
    neighborhood: list[str]
        list of all Tile Identifiers part of the 5x5 Tiles grid, with center_tile_id in central position
    """

    return _build_neighborhood(center_tile_id, radius=2)


def get_3x3_tiles_ids(center_tile_id):
    """Return the list of 3x3 Tile IDs grid, surrounding input central tile.

    Parameters
    ----------
    center_tile_id:str
        Tile Identifier used as central one in the 3x3 grid

    Returns
    -------
    neighborhood: list[str]
        list of all Tile Identifiers part of the 3x3 Tiles grid, with center_tile_id in central position
    """

    return _build_neighborhood(center_tile_id, radius=1)


def extract_5x5_tiles_from_total(l2a_tiles_id: list[str], tiles_ids_5x5_list: list[str]) -> list[str]:
    """
    Discard from the input L2A GN Tile IDS, the ones outside the 5x5 Grid

    Parameters
    ----------
    l2a_tiles_id: list[str]
        list of Tile IDs present in the input L2A GN products

    tiles_ids_5x5_list: list[str]
        list of the 5x grid Tile IDs used for AGB processing

    Returns
    ------
    l2a_tiles_id: list[str]
       list of Tile IDs present in the input L2A GN products, where Tiles out from 5x5 grid have been discarded

    """
    # Input L2a can be defined on more than a tile, so, tiles outside the 5x5 should be not used
    bps_logger.info("Selecting all and only the L2A inputs falling into this 5x5 tile grid:")

    l2a_tiles_id_extracted = []
    for tile in l2a_tiles_id:
        if tile in tiles_ids_5x5_list:
            l2a_tiles_id_extracted.append(tile)

    num_tiles_in = len(l2a_tiles_id)
    num_tiles_extracted = len(l2a_tiles_id_extracted)
    if num_tiles_extracted == 0:
        raise ValueError("    any input L2A tile falls into the 5x5 group")

    if num_tiles_extracted < num_tiles_in:
        bps_logger.info(
            f"    only {num_tiles_extracted} of {num_tiles_in} L2a GN products DGG tiles fall into the 5x5 group and will be used for processing"
        )
    else:
        bps_logger.info(
            f"    all the {num_tiles_extracted} L2a GN products DGG tiles fall into the 5x5 group and will be used for processing"
        )
    bps_logger.info(f"    the {num_tiles_extracted} DGG tiles used for processing are the following:")
    bps_logger.info(f"        {l2a_tiles_id_extracted}")

    return l2a_tiles_id_extracted


def build_unknowns_matrix(pol_num, date_num, lcm_num, choice):
    m = np.zeros([pol_num, date_num, lcm_num], dtype=np.int32)

    # build a mapping volume for unknown position in regression solution matrix

    unknowns = 1

    if choice == "p":
        unknowns = pol_num
        for i in range(pol_num):
            m[i, :, :] = i

    elif choice == "j":
        unknowns = date_num
        for i in range(date_num):
            m[:, i, :] = i

    elif choice == "k":
        unknowns = lcm_num
        for i in range(lcm_num):
            m[:, :, i] = i

    elif choice == "jk":
        unknowns = date_num * lcm_num
        for i in range(lcm_num):
            for j in range(date_num):
                m[:, j, i] = i * date_num + j

    elif choice == "pj":
        unknowns = pol_num * date_num
        for i in range(date_num):
            for j in range(pol_num):
                m[j, i, :] = i * pol_num + j

    elif choice == "pk":
        unknowns = pol_num * lcm_num
        for i in range(lcm_num):
            for j in range(pol_num):
                m[j, :, i] = i * pol_num + j

    elif choice == "pjk":
        unknowns = pol_num * date_num * lcm_num
        for i in range(lcm_num):
            for j in range(date_num):
                for k in range(pol_num):
                    m[k, j, i] = i * pol_num * date_num + j * pol_num + k

    return m, unknowns


@nb.njit(nogil=True, cache=True, parallel=True)
def build_regression_matrix_jit(agb_db, angle_db, pol_idx, date_idx, lcm_idx, M_l, M_alfa, M_n, unknowns):
    # build regression solution matrix. Mapping volumes must be computed in advance, and reference inputs must be consolidated

    n = agb_db.shape[0]

    A = np.zeros((n, unknowns), dtype="float32")

    for i in nb.prange(n):
        A[i, M_l[pol_idx[i], date_idx[i], lcm_idx[i]]] = 1

    for i in nb.prange(n):
        A[i, M_alfa[pol_idx[i], date_idx[i], lcm_idx[i]]] = agb_db[i]

    for i in nb.prange(n):
        A[i, M_n[pol_idx[i], date_idx[i], lcm_idx[i]]] = angle_db[i]

    return A


def build_regression_matrix(agb_db, angle_db, pol_idx, date_idx, lcm_idx, M_l, M_alfa, M_n, unknowns):
    # build regression solution matrix. Mapping volumes must be computed in advance, and reference inputs must be consolidated

    n = agb_db.shape[0]

    A = np.zeros((n, unknowns), dtype="float32")

    for i in range(n):
        A[i, M_l[pol_idx[i], date_idx[i], lcm_idx[i]]] = 1

    for i in range(n):
        A[i, M_alfa[pol_idx[i], date_idx[i], lcm_idx[i]]] = agb_db[i]

    for i in range(n):
        A[i, M_n[pol_idx[i], date_idx[i], lcm_idx[i]]] = angle_db[i]

    return A


def check_matrix_singularity(A):
    # remove empty columns in regression matrix. A mapping array is also returned

    valid_unknowns = np.arange(A.shape[1])

    #    null_columns_idx = np.argwhere(np.all(A[..., :] == 0, axis=0))

    null_columns_idx = np.argwhere(np.sum(A, axis=0) == 0)

    if np.any(null_columns_idx):
        return np.delete(A, null_columns_idx, axis=1), np.delete(valid_unknowns, null_columns_idx, axis=0)

    return A, valid_unknowns


def compute_linear_regression(gn, A_true, N_constants=None):
    # compute linear regression, taking care of unsolvable unknowns

    # Manage constant N case

    if N_constants is not None:
        gn -= (A_true[:, -3:] @ np.reshape(N_constants, (3, 1))).squeeze()
        A_true = A_true[:, :-3]
    # full number of unknowns
    N_true = A_true.shape[1]
    # no-singular matrix building

    A, idx = check_matrix_singularity(A_true)

    # also take care of all zero rows
    nonEmptyRows = ~(A == 0).all(axis=1)
    A = A[nonEmptyRows]
    gn = gn[nonEmptyRows]
    # convenience matrices
    inv_At_A = np.linalg.pinv(A.T @ A)

    # solve for unknowns
    x = inv_At_A @ A.T @ gn
    x_true = np.empty(N_true, dtype="float32")
    x_true.fill(np.nan)
    x_true[idx] = x

    # solve for standard deviation
    r = gn - A @ x
    r_sq = r.T @ r
    std = np.sqrt(r_sq * np.diag(inv_At_A) / (A.shape[0] - A.shape[1]))
    std_true = np.empty(N_true, dtype="float32")
    std_true.fill(np.nan)
    std_true[idx] = std

    # append known solution for constant N case
    if N_constants is not None:
        x_true = np.append(x_true, np.reshape(N_constants, (3, 1)))
        std_true = np.append(std_true, np.reshape(0.0 * N_constants, (3, 1)))
    # return regression solution
    return x_true, std_true


@nb.njit(nogil=True, cache=True, parallel=False)
def compute_rho_jit(
    agb_db,
    gn_db,
    angle_db,
    point_idx,
    pol_idx,
    date_idx,
    lcm_idx,
    L_unknowns,
    alfa_unknowns,
    n_unknowns,
):
    # point indices are not required to start from 0, and can be non consecutive series
    # nans in L, alfa, n estimates are nicely managed
    # safe not to parallelize (as output pos is indexed on same arrays)

    # allocate point related maps

    first_point = np.min(point_idx)
    N_points = np.max(point_idx) - first_point + 1
    w_true = np.zeros(N_points, dtype="float64")
    w_est = np.zeros(N_points, dtype="float64")
    a2 = np.zeros(N_points, dtype="float64")

    for i in nb.prange(agb_db.shape[0]):
        # current index for unknown maps

        pol_i = pol_idx[i]
        date_i = date_idx[i]
        lcm_i = lcm_idx[i]
        # compute estimated agb for current point
        curr_alfa = alfa_unknowns[pol_i, date_i, lcm_i]
        agb_est = (
            gn_db[i] - L_unknowns[pol_i, date_i, lcm_i] - n_unknowns[pol_i, date_i, lcm_i] * angle_db[i]
        ) / curr_alfa

        # accumulate only valid (non-nan) estimates
        if np.isfinite(agb_est) and np.isfinite(agb_db[i]):
            # position in point array
            pos = point_idx[i] - first_point
            # averaging term on point
            a = curr_alfa * curr_alfa
            # accumulate numerator on point
            w_est[pos] = w_est[pos] + agb_est * a
            # accumulate denominator on point
            a2[pos] = a2[pos] + a
            w_true[pos] = agb_db[i]

    # now put back nans in map (as 10^0 = 1.0, 0.0 is not correctly handled in rho computationrho)

    w_true[a2 == 0.0] = np.nan
    w_est[a2 == 0.0] = np.nan
    rho = np.nansum(np.power(10.0, 0.1 * w_true)) / np.nansum(np.power(10.0, 0.1 * (w_est / a2)))

    return rho


def estimate_agb_only(gn_db, angle_db, pol_idx, date_idx, lcm_idx, L_unknowns, alfa_unknowns, n_unknowns):
    return (
        gn_db - L_unknowns[pol_idx, date_idx, lcm_idx] - n_unknowns[pol_idx, date_idx, lcm_idx] * angle_db
    ) / alfa_unknowns[pol_idx, date_idx, lcm_idx]


def compute_rho_dummy(
    agb_db,
    gn_db,
    angle_db,
    point_idx,
    pol_idx,
    date_idx,
    lcm_idx,
    L_unknowns,
    alfa_unknowns,
    n_unknowns,
):
    agb_est = (
        gn_db
        - L_unknowns[
            pol_idx,
            date_idx,
            lcm_idx,
        ]
        - n_unknowns[
            pol_idx,
            date_idx,
            lcm_idx,
        ]
        * angle_db
    ) / alfa_unknowns[
        pol_idx,
        date_idx,
        lcm_idx,
    ]

    mask = np.isfinite(agb_est)
    rho = np.nansum(np.power(10.0, 0.1 * agb_db[mask])) / np.nansum(np.power(10.0, 0.1 * agb_est[mask]))

    return rho


def compute_rho(
    agb_db,
    gn_db,
    angle_db,
    point_idx,
    pol_idx,
    date_idx,
    lcm_idx,
    L_unknowns,
    alfa_unknowns,
    n_unknowns,
):
    N_points = np.max(point_idx) + 1

    w_true = np.zeros(N_points, dtype="float32")
    w_est = np.zeros(N_points, dtype="float32")
    a2 = np.zeros(N_points, dtype="float32")

    agb_est = estimate_agb_only(
        gn_db,
        angle_db,
        pol_idx,
        date_idx,
        lcm_idx,
        L_unknowns,
        alfa_unknowns,
        n_unknowns,
    )

    for i in range(agb_db.shape[0]):
        pos = point_idx[i]
        w_true[pos] = agb_db[i]
        a = alfa_unknowns[pol_idx[i], date_idx[i], lcm_idx[i]]
        a = a * a
        w_est[pos] = w_est[pos] + agb_est[i] * a
        a2[pos] = a2[pos] + a

    rho = np.nansum(np.power(10.0, 0.1 * w_true)) / np.nansum(np.power(10.0, 0.1 * (w_est / a2)))

    return rho


@nb.njit(nogil=True, cache=True, parallel=True)
def compute_agb_jit(
    est_agb_mean_a2,
    est_agb_std_a2,
    est_agb_a2,
    gn_db,
    angle_db,
    pol_idx,
    date_idx,
    lcm_idx,
    L_unknowns,
    alfa_unknowns,
    n_unknowns,
    L_stddev,
    alfa_stddev,
    n_stddev,
):
    for i in nb.prange(est_agb_a2.shape[0]):
        for k in nb.prange(est_agb_a2.shape[1]):
            pol_i = pol_idx
            date_i = date_idx
            lcm_i = lcm_idx[i, k]
            if (
                lcm_i == -1
                or np.isnan(gn_db[i, k])
                or np.isnan(angle_db[i, k])
                or np.isnan(L_unknowns[pol_i, date_i, lcm_i])
                or np.isnan(n_unknowns[pol_i, date_i, lcm_i])
                or np.isnan(alfa_unknowns[pol_i, date_i, lcm_i])
            ):
                continue
            curr_a2 = alfa_unknowns[pol_i, date_i, lcm_i] ** 2
            curr_agb = (
                gn_db[i, k] - L_unknowns[pol_i, date_i, lcm_i] - n_unknowns[pol_i, date_i, lcm_i] * angle_db[i, k]
            ) * alfa_unknowns[pol_i, date_i, lcm_i]
            curr_std_sq = (
                curr_a2
                * (n_stddev[pol_i, date_i, lcm_i] ** 2 * angle_db[i, k] ** 2 + L_stddev[pol_i, date_i, lcm_i] ** 2)
                + alfa_stddev[pol_i, date_i, lcm_i] ** 2 * curr_agb**2 / curr_a2
            )

            est_agb_mean_a2[i, k] = curr_agb + est_agb_mean_a2[i, k]
            est_agb_std_a2[i, k] = curr_std_sq + est_agb_std_a2[i, k]
            est_agb_a2[i, k] = curr_a2 + est_agb_a2[i, k]

    return est_agb_mean_a2, est_agb_std_a2, est_agb_a2


def compute_agb(
    est_agb_mean_a2,
    est_agb_std_a2,
    est_agb_a2,
    gn_db,
    angle_db,
    pol_idx,
    date_idx,
    lcm_idx,
    L_unknowns,
    alfa_unknowns,
    n_unknowns,
    L_stddev,
    alfa_stddev,
    n_stddev,
):
    curr_a2 = alfa_unknowns[pol_idx, date_idx, lcm_idx] ** 2
    curr_agb = (
        gn_db - L_unknowns[pol_idx, date_idx, lcm_idx] - n_unknowns[pol_idx, date_idx, lcm_idx] * angle_db
    ) * alfa_unknowns[pol_idx, date_idx, lcm_idx]
    curr_std_sq = (
        n_stddev[pol_idx, date_idx, lcm_idx] ** 2 * angle_db**2
        + L_stddev[pol_idx, date_idx, lcm_idx] ** 2
        + alfa_stddev[pol_idx, date_idx, lcm_idx] ** 2 * curr_agb**2
    )

    mask = np.isnan(curr_agb)
    est_agb_mean_a2 = np.where(mask, est_agb_mean_a2, curr_agb + est_agb_mean_a2)
    est_agb_std_a2 = np.where(mask, est_agb_std_a2, curr_std_sq + est_agb_std_a2)
    est_agb_a2 = np.where(mask, est_agb_a2, curr_a2 + est_agb_a2)

    return est_agb_mean_a2, est_agb_std_a2, est_agb_a2


def finalize_agb(est_agb_mean_a2, est_agb_std_a2, est_agb_a2, rho):
    corr_term = 10.0 * np.log10(np.e)

    est_agb = np.where(est_agb_a2 == 0.0, np.nan, est_agb_mean_a2 / est_agb_a2 / corr_term)
    est_std_sq = np.where(est_agb_a2 == 0.0, np.nan, est_agb_std_a2 / est_agb_a2 / corr_term**2)

    W = rho * np.exp(est_agb) * np.exp(est_std_sq / 2.0)
    STD = rho * np.sqrt((np.exp(est_std_sq) - 1) * np.exp(2.0 * est_agb + est_std_sq))

    return W, STD


def _interpolate_lcm_over_tile(
    lcm_product: LCMMask,
    dgg_tile_latitude_axis: np.ndarray,
    dgg_tile_longitude_axis: np.ndarray,
):
    # manage decrescent axes case
    invert_lat_out, invert_lon_out = __check_lat_lon_orientation(dgg_tile_latitude_axis, dgg_tile_longitude_axis)

    lcm_interp = parallel_nn_reinterpolate(
        [lcm_product.mask],  # function works with lists
        lcm_product.lat_axis,  # fnf orientation is our default
        lcm_product.lon_axis,  # fnf orientation is our default
        np.flip(dgg_tile_latitude_axis) if invert_lat_out else dgg_tile_latitude_axis,
        np.flip(dgg_tile_longitude_axis) if invert_lon_out else dgg_tile_longitude_axis,
        fill_value=INT_NODATA_VALUE,
    )[0]  # function works with lists
    lcm_interp.astype(np.uint8)

    return lcm_interp


def __check_lat_lon_orientation(latitude_axis, longitude_axis):
    invert_latitude = False
    invert_longitude = False
    if latitude_axis[-1] - latitude_axis[0] < 0:
        invert_latitude = True
    if longitude_axis[-1] - longitude_axis[0] < 0:
        invert_longitude = True

    return invert_latitude, invert_longitude


def parallel_nn_reinterpolate(data_lst, axis1_in, axis2_in, axis1_out, axis2_out, fill_value=np.nan):
    N = len(data_lst)
    execution_lst = [[] for _ in range(N)]
    for i in range(N):
        execution_lst[i].extend([data_lst[i], axis1_in, axis2_in, axis1_out, axis2_out])

    results_lst = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for number, res in zip(
            execution_lst,
            executor.map(lambda f: reinterpolate_nn_core(*f, fill_value), execution_lst),
        ):
            results_lst.append(res)

    return results_lst


def reinterpolate_nn_core(data, dx_az, dx_rg, sx_az, sx_rg, fill_value):
    return _nearest_interpolation_2d(data, dx_az, dx_rg, sx_az, sx_rg, fill_value).astype(type(data[0, 0]))


@nb.njit(nogil=True, cache=True, parallel=False)
def _regular_nearest_interpolator(outputM, inputM, ax0in, ax1in, ax0out, ax1out, outOfBounds, tolerance):
    innerM = np.zeros((inputM.shape[0], outputM.shape[1]), dtype=inputM.dtype)

    dx0 = np.mean(ax0in[1:] - ax0in[:-1])
    dx1 = np.mean(ax1in[1:] - ax1in[:-1])

    sign0 = 1.0
    if dx0 < 0.0:
        sign0 = -1.0

    sign1 = 1.0
    if dx1 < 0.0:
        sign1 = -1.0

    tol0 = sign0 * dx0 * tolerance
    tol1 = sign1 * dx1 * tolerance

    mx0 = sign0 * ax0in[0]
    Mx0 = sign0 * ax0in[-1]

    mx1 = sign1 * ax1in[0]
    Mx1 = sign1 * ax1in[-1]

    dx0 *= sign0
    dx1 *= sign1

    for i1 in range(outputM.shape[1]):
        px1 = sign1 * ax1out[i1]
        if px1 < mx1 - tol1:
            innerM[:, i1] = outOfBounds
        elif px1 <= mx1:
            innerM[:, i1] = inputM[:, 0]
        elif px1 > Mx1 + tol1:
            innerM[:, i1] = outOfBounds
        elif px1 >= Mx1:
            innerM[:, i1] = inputM[:, -1]
        else:
            c = int((px1 - mx1) // dx1)
            w = np.round((px1 - sign1 * ax1in[c]) / dx1)
            innerM[:, i1] = (1.0 - w) * inputM[:, c] + w * inputM[:, c + 1]

    for i0 in range(outputM.shape[0]):
        px0 = sign0 * ax0out[i0]
        if px0 < mx0 - tol0:
            outputM[i0, :] = outOfBounds
        elif px0 <= mx0:
            outputM[i0, :] = innerM[0, :]
        elif px0 > Mx0 + tol0:
            outputM[i0, :] = outOfBounds
        elif px0 >= Mx0:
            outputM[i0, :] = innerM[-1, :]
        else:
            c = int((px0 - mx0) // dx0)
            w = np.round((px0 - sign0 * ax0in[c]) / dx0)
            outputM[i0, :] = (1.0 - w) * innerM[c, :] + w * innerM[c + 1, :]

    return


def _nearest_interpolation_2d(
    data_in,
    axis_az_in,
    axis_sr_in,
    axis_sub_az_out,
    axis_sr_out,
    fill_value=np.nan,
    tolerance=0.001,
):
    data_out = np.zeros((axis_sub_az_out.shape[0], axis_sr_out.shape[0]), dtype=data_in.dtype)

    _regular_nearest_interpolator(
        data_out,
        data_in,
        axis_az_in,
        axis_sr_in,
        axis_sub_az_out,
        axis_sr_out,
        fill_value,
        tolerance,
    )

    return data_out


def dgg_tiling_create_axis_custom(
    tile_id,
    l2b_product_type,
    l2a_products_list,
    geotransform_d,
    tiles_dict,
):
    from bps.l2b_agb_processor.l2b_common_functionalities import _get_latitude_band_key

    assert tile_id in geotransform_d

    geotransform = geotransform_d[tile_id]
    tiles_info = tiles_dict[tile_id]
    lat_min = np.min(tiles_info["latitude_vector"])
    lat_max = np.max(tiles_info["latitude_vector"])

    # Tiling
    # Construct the lat lon axis of the output Tile:
    dgg_sampling_dict = create_dgg_sampling_dict(l2b_product_type)
    dgg_band_key = _get_latitude_band_key(lat_min, lat_max)
    dgg_sampling = dgg_sampling_dict[dgg_band_key]

    dgg_lat_sign = 1 if geotransform[5] > 0 else -1
    l2b_dgg_tile_latitude_axis_deg = geotransform[3] + dgg_lat_sign * dgg_sampling["latitude_spacing_deg"] * np.arange(
        dgg_sampling["n_lat"]
    )

    dgg_lon_sign = 1 if geotransform[1] > 0 else -1
    l2b_dgg_tile_longitude_axis_deg = geotransform[0] + dgg_lon_sign * dgg_sampling[
        "longitude_spacing_deg"
    ] * np.arange(dgg_sampling["n_lon"])

    # now check the convention of dgg axis respect to the one in the l2a products
    # and flip dgg if necessary:
    l2a_lat_sign = l2a_products_list[0].measurement.latitude_vec[5] - l2a_products_list[0].measurement.latitude_vec[4]
    l2a_lon_sign = l2a_products_list[0].measurement.longitude_vec[5] - l2a_products_list[0].measurement.longitude_vec[4]
    if (l2a_lat_sign > 0 and dgg_lat_sign < 0) or (l2a_lat_sign < 0 and dgg_lat_sign > 0):
        l2b_dgg_tile_latitude_axis_deg = np.flip(l2b_dgg_tile_latitude_axis_deg)
    if (l2a_lon_sign > 0 and dgg_lon_sign < 0) or (l2a_lon_sign < 0 and dgg_lon_sign > 0):
        l2b_dgg_tile_longitude_axis_deg = np.flip(l2b_dgg_tile_longitude_axis_deg)

    # [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon]
    # This is easy to write, because Tile is oriented without inclinations
    dgg_tile_footprint = [
        max(l2b_dgg_tile_latitude_axis_deg),
        max(l2b_dgg_tile_longitude_axis_deg),
        min(l2b_dgg_tile_latitude_axis_deg),
        max(l2b_dgg_tile_longitude_axis_deg),
        min(l2b_dgg_tile_latitude_axis_deg),
        min(l2b_dgg_tile_longitude_axis_deg),
        max(l2b_dgg_tile_latitude_axis_deg),
        min(l2b_dgg_tile_longitude_axis_deg),
    ]

    return (
        l2b_dgg_tile_latitude_axis_deg,
        l2b_dgg_tile_longitude_axis_deg,
        dgg_tile_footprint,
        dgg_band_key,
    )


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, str | bytes | bytearray):
        size += sum([get_size(i, seen) for i in obj])
    return size


def _compute_coverage_percentage_and_pixels(matrix_in: np.ndarray):
    coverage_number = np.sum(np.logical_not(np.isnan(matrix_in)))
    coverage_percentage = coverage_number * 100 / matrix_in.size

    # Return a list with coverage percentage and number of pixels
    return [
        float(coverage_percentage),
        int(coverage_number),
    ]


def _agb_unit_test_preparation(l2a_gn_products_list, data_object_5x5, tiles_ids_5x5_list):
    # Tool for generating TDS-BPS-L2B-009
    #   If generation_flag is set to"agb_live_run", AGB is injected in data_object_5x5 and returned for a live run
    #   If generation_flag is set to"for_tds_generation", GN and AGB are saved to np, for the TDS Generation
    #       After generation, stop this processor and generate TDS with "\biomass_l2_prototype\BPS\create_agb_tds_009_012.py"
    #       degraded_flag is used to inject many no data values in the generated AGB and GN (used for TDS.BPS-L2B-012)

    # INPUTS:
    save_path = Path("C:/ARESYS_PROJ/workingDir/BPS/test_cases/TEST-BPS-L2B-013/for_TDS_009")
    generation_flag = "for_tds_generation"  # "agb_live_run", "for_tds_generation"
    degraded_flag = False

    num_m, num_n = data_object_5x5[next(iter(data_object_5x5))]["data_list"][0]["incidence_angle"].shape

    # Parameters to be injected in the data
    L_HH = -25.0
    L_VV = -27.0
    L_XX = -32.0
    alfa = 0.8
    n = 2.0

    if generation_flag == "agb_live_run":
        for idx, (tile_id, data_object_1x1) in enumerate(data_object_5x5.items()):
            np.random.seed(idx)
            agb = np.random.rand(num_m, num_n).astype(np.float32) * 300.0 + 150.0

            if data_object_1x1["cal_ab"] is not None:
                agb = np.where(
                    np.isnan(data_object_1x1["cal_ab"][:, :, 0]),
                    agb,
                    data_object_1x1["cal_ab"][:, :, 0],
                )

            for data_dict in data_object_1x1["data_list"]:
                gn_db = (
                    L_HH
                    + alfa * 10.0 * np.log10(agb)
                    + n * 10.0 * np.log10(np.cos(np.deg2rad(data_dict["incidence_angle"])))
                )
                data_dict["data"][:, :, 0] = np.power(10, gn_db / 10.0)

                gn_db = (
                    L_XX
                    + alfa * 10.0 * np.log10(agb)
                    + n * 10.0 * np.log10(np.cos(np.deg2rad(data_dict["incidence_angle"])))
                )
                data_dict["data"][:, :, 1] = np.power(10, gn_db / 10.0)

                gn_db = (
                    L_VV
                    + alfa * 10.0 * np.log10(agb)
                    + n * 10.0 * np.log10(np.cos(np.deg2rad(data_dict["incidence_angle"])))
                )
                data_dict["data"][:, :, 2] = np.power(10, gn_db / 10.0)
        return data_object_5x5

    elif generation_flag == "for_tds_generation":
        # Dimensions of a 5x5 tile:
        num_m = 6001
        num_n = 6001
        np.random.seed(0)
        agb = np.random.rand(num_m, num_n).astype(np.float32) * 300.0 + 150.0
        cal_ab_5x5 = np.zeros((num_m, num_n)).astype(np.float32) * np.nan

        lat_axis = {}
        lon_axis = {}
        (
            _,
            geotransform_d,
            _,
            tiles_dict,
        ) = dgg_search_tiles([-5, 5, 15, 25], True)
        for tile_id in tiles_ids_5x5_list:
            (
                l2b_dgg_tile_latitude_axis_deg,
                l2b_dgg_tile_longitude_axis_deg,
                _,
                _,
            ) = dgg_tiling_create_axis_custom(
                tile_id,
                "FP_GN__L2A",
                l2a_gn_products_list,
                geotransform_d,
                tiles_dict,
            )
            lat_axis[tile_id] = l2b_dgg_tile_latitude_axis_deg
            lon_axis[tile_id] = l2b_dgg_tile_longitude_axis_deg

        lon_step = (lon_axis["S01E023"][-1] - lon_axis["S01E019"][0]) / 6000
        lon_axis_5x5 = np.arange(6001) * lon_step + lon_axis["S01E019"][0]

        lat_step = (lat_axis["S05E023"][-1] - lat_axis["S01E023"][0]) / 6000
        lat_axis_5x5 = np.arange(6001) * lat_step + lat_axis["S01E023"][0]

        for idx, (tile_id, data_object_1x1) in enumerate(data_object_5x5.items()):
            if data_object_1x1["cal_ab"] is not None:
                print(f"Found cal AB for tile {tile_id}")

                pixel_lon_pos = np.argmin(np.abs(lon_axis_5x5 - lon_axis[tile_id][0]))
                pixel_lat_pos = np.argmin(np.abs(lat_axis_5x5 - lat_axis[tile_id][0]))
                cal_ab_5x5[
                    pixel_lat_pos : pixel_lat_pos + 1201,
                    pixel_lon_pos : pixel_lon_pos + 1201,
                ] = data_object_1x1["cal_ab"][:, :, 0]

        agb = np.where(
            np.isnan(cal_ab_5x5),
            agb,
            cal_ab_5x5,
        )

        if degraded_flag:
            mask = np.logical_or(
                np.round(np.random.uniform(low=0, high=1, size=agb.shape)).astype(bool),
                np.round(np.random.uniform(low=0, high=1, size=agb.shape)).astype(bool),
            )
            agb[mask] = np.nan

        gn_to_save = {}
        lon_axis_to_save = {"assi": []}
        lat_axis_to_save = {"assi": []}
        tiles_dict = {}
        for idx, l2a_gn_product in enumerate(l2a_gn_products_list):
            gn_to_save[idx] = []

            pixel_lon_pos = np.argmin(np.abs(lon_axis_5x5 - l2a_gn_product.measurement.longitude_vec[0]))
            pixel_lat_pos = np.argmin(np.abs(lat_axis_5x5 - l2a_gn_product.measurement.latitude_vec[0]))

            measurement_lat_len = len(l2a_gn_product.measurement.latitude_vec)
            measurement_lon_len = len(l2a_gn_product.measurement.longitude_vec)
            gn_db = (
                L_HH
                + alfa
                * 10.0
                * np.log10(
                    agb[
                        pixel_lat_pos : pixel_lat_pos + measurement_lat_len,
                        pixel_lon_pos : pixel_lon_pos + measurement_lon_len,
                    ]
                )
                + n * 10.0 * np.log10(np.cos(np.deg2rad(l2a_gn_product.lut_ads.lut_local_incidence_angle)))
            )
            gn_to_save[idx].append(np.power(10, gn_db / 10.0))

            gn_db = (
                L_XX
                + alfa
                * 10.0
                * np.log10(
                    agb[
                        pixel_lat_pos : pixel_lat_pos + measurement_lat_len,
                        pixel_lon_pos : pixel_lon_pos + measurement_lon_len,
                    ]
                )
                + n * 10.0 * np.log10(np.cos(np.deg2rad(l2a_gn_product.lut_ads.lut_local_incidence_angle)))
            )
            gn_to_save[idx].append(np.power(10, gn_db / 10.0))

            gn_db = (
                L_VV
                + alfa
                * 10.0
                * np.log10(
                    agb[
                        pixel_lat_pos : pixel_lat_pos + measurement_lat_len,
                        pixel_lon_pos : pixel_lon_pos + measurement_lon_len,
                    ]
                )
                + n * 10.0 * np.log10(np.cos(np.deg2rad(l2a_gn_product.lut_ads.lut_local_incidence_angle)))
            )
            gn_to_save[idx].append(np.power(10, gn_db / 10.0))

            lon_axis_to_save["assi"].append(lon_axis_5x5[pixel_lon_pos : pixel_lon_pos + measurement_lon_len])
            lat_axis_to_save["assi"].append(lat_axis_5x5[pixel_lat_pos : pixel_lat_pos + measurement_lat_len])
            tiles_dict[idx] = l2a_gn_product.main_ads_product.tile_id_list

        save_path.mkdir(parents=True, exist_ok=False)
        np.save(
            save_path.joinpath("gn_for_agb.npy"),
            gn_to_save,
        )
        np.save(
            save_path.joinpath("lon_axis_to_save.npy"),
            lon_axis_to_save,
        )
        np.save(
            save_path.joinpath("lat_axis_to_save.npy"),
            lat_axis_to_save,
        )
        np.save(
            save_path.joinpath("agb.npy"),
            agb,
        )
        np.save(
            save_path.joinpath("agb_lon_axis.npy"),
            lon_axis_5x5,
        )
        np.save(
            save_path.joinpath("agb_lat_axis.npy"),
            lat_axis_5x5,
        )
        np.save(
            save_path.joinpath("tiles_dict.npy"),
            tiles_dict,
        )
        return 0


def _create_cal_ab_object(
    cal_agb_product_dict: dict,
    lcm_product,
    tile_ids_cal_ab_discarted: list[str],
):
    cal_ab_object = {}  # one key for each tile, containing a data_object_1x1
    for tile_id in tile_ids_cal_ab_discarted:
        dgg_tile_latitude_axis_curr = cal_agb_product_dict[tile_id]["dgg_lat_axis_deg"]
        dgg_tile_longitude_axis_curr = cal_agb_product_dict[tile_id]["dgg_lon_axis_deg"]

        # References management (here, fill both CAL_AB and AGB, they are mergedf in consolidation function)
        cal_ab = cal_agb_product_dict[tile_id]["cal_ab"] if tile_id in cal_agb_product_dict.keys() else None

        # LCM is one for all the inputs, interpolation over the tile axis needed
        # interpolator Nearest Neigbour
        lcm_mask = _interpolate_lcm_over_tile(
            lcm_product,
            dgg_tile_latitude_axis_curr,
            dgg_tile_longitude_axis_curr,
        ).astype(np.int32)

        data_object_1x1 = {
            "lcm": lcm_mask,
            "cal_ab": cal_ab,  # Can be None
        }

        cal_ab_object[tile_id] = data_object_1x1

    return cal_ab_object
