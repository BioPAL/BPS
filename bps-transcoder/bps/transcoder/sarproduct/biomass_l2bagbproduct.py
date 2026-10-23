# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import os
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io.mph import get_mph_path
from bps.transcoder.io import (
    common_annotation_l2l3,
    main_annotation_l2b_agb,
)
from bps.transcoder.utils.production_model_utils import encode_product_name_id_value
from bps.transcoder.utils.time_conversions import pdt_to_compact_date


class agbTileStatus(Enum):
    """agb Tile Status

    three statuses are identified:
    “Complete”: all pixels are correctly processed
    “NotComplete”: some pixels are invalid and cannot be recovered by a subsequent iteration (or their number is under the selected threshold in eq. 4.39 ATBD)
    “Partial”; corresponding to rerun Boolean flag set to true (eq. 4.39 ATBD): a subsequent iteration is required to fill-up estimation voids.
    """

    COMPLETE = "Complete"
    NOTCOMPLETE = "NotComplete"
    PARTIAL = "Partial"


class BIOMASSL2bAGBProductStructure:
    """_summary_"""

    def __init__(self, product_path) -> None:
        """_summary_

        Parameters
        ----------
        product_path : _type_
            _description_
        """

        self.product_path = product_path

        self.product_type = "FP_AGB_L2B"

        self.measurement_subfolder = "measurement"
        self.annotation_subfolder = "annotation"
        self.preview_subfolder = "preview"
        self.schema_subfolder = "schema"

        self.mph_file = None
        self.vrt_file = None
        self.agb_file = None
        self.agb_standard_deviation_file = None
        self.heatmap_file = None
        self.acquisition_id_image_file = None
        self.bps_fnf_file = None
        self.main_annotation_file = None
        self.stac_file = None
        self.quicklook_files = None
        self.overlay_files = None
        self.schema_files = None

        self._set_product_paths()

    def _set_product_paths(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        if os.path.exists(self.product_path):
            # Set paths of BIOMASS L2b product single files starting from an existing SAR product
            # - MPH file
            self.mph_file = (glob(os.path.join(self.product_path, "*.xml")) or [None])[0]
            # - STAC file
            self.stac_file = (glob(os.path.join(self.product_path, "*.json")) or [None])[0]
            # - Measurement files
            self.agb_file = glob(os.path.join(self.product_path, self.measurement_subfolder, "*_i_agb.tiff"))[0]

            self.agb_standard_deviation_file = glob(
                os.path.join(
                    self.product_path,
                    self.measurement_subfolder,
                    "*_i_agb_std_dev.tiff",
                )
            )[0]
            self.vrt_file = (glob(os.path.join(self.product_path, self.measurement_subfolder, "*_i.vrt")) or [None])[0]
            # - Annotation files
            self.main_annotation_file = (
                glob(os.path.join(self.product_path, self.annotation_subfolder, "*_annot.xml")) or [None]
            )[0]

            self.heatmap_file = glob(os.path.join(self.product_path, self.annotation_subfolder, "*_i_heatmap.tiff"))[0]

            self.acquisition_id_image_file = glob(
                os.path.join(
                    self.product_path,
                    self.annotation_subfolder,
                    "*_i_acquisition_id_image.tiff",
                )
            )[0]

            self.bps_fnf_file = glob(os.path.join(self.product_path, self.annotation_subfolder, "*_i_bps_fnf.tiff"))[0]

            # - Quick-look file
            self.quicklook_files = glob(os.path.join(self.product_path, self.preview_subfolder, "*_ql.png")) or [None]

            # - Overlay files
            self.overlay_files = glob(os.path.join(self.product_path, self.preview_subfolder, "*_map.kml")) or [None]

            # - Schema files
            self.schema_files = glob(os.path.join(self.product_path, self.schema_subfolder, "*.xsd"))
        else:
            # Set paths of BIOMASS L2b product single files starting from product name
            product_name = os.path.basename(self.product_path)
            # - MPH file
            self.mph_file = str(get_mph_path(Path(self.product_path)))
            # - Measurement files
            measurement_file_root = os.path.join(
                self.product_path,
                self.measurement_subfolder,
                product_name.lower()[:-10],
            )
            self.vrt_file = measurement_file_root + "_i.vrt"
            self.agb_file = measurement_file_root + "_i_agb.tiff"
            self.agb_standard_deviation_file = measurement_file_root + "_i_agb_std_dev.tiff"

            # - Annotation files
            annotation_file_root = os.path.join(
                self.product_path, self.annotation_subfolder, product_name.lower()[:-10]
            )
            self.main_annotation_file = annotation_file_root + "_annot.xml"
            self.heatmap_file = annotation_file_root + "_i_heatmap.tiff"  # 3 layers for AGB
            self.acquisition_id_image_file = annotation_file_root + "_i_acquisition_id_image.tiff"
            self.bps_fnf_file = annotation_file_root + "_i_bps_fnf.tiff"

            # - STAC file
            self.stac_file = os.path.join(self.product_path, product_name.lower()[:-10]) + ".json"
            # - Quick-look files
            preview_file_root = os.path.join(self.product_path, self.preview_subfolder, product_name.lower()[:-10])
            self.quicklook_files = [
                preview_file_root + "_agb_ql.png",
                preview_file_root + "_agb_std_dev_ql.png",
                preview_file_root + "_hm_ql.png",
                preview_file_root + "_hm_reference_ql.png",
                preview_file_root + "_hm_additional_reference_ql.png",
            ]
            # - Overlay files
            self.overlay_files = [
                preview_file_root + "_agb_map.kml",
                preview_file_root + "_agb_std_dev_map.kml",
                preview_file_root + "_hm_map.kml",
                preview_file_root + "_hm_reference_map.kml",
                preview_file_root + "_hm_additional_reference_map.kml",
            ]

            # - Schema files
            schema_dir = Path(self.product_path, self.schema_subfolder)
            self.l2b_agb_main_ann_xsd = schema_dir.joinpath("bio-l2b-agb-main-annotation.xsd")
            self.l2l3_common_ann_xsd = schema_dir.joinpath("bio-l2l3-common-annotations.xsd")
            self.l2l3_agb_proc_ann_xsd = schema_dir.joinpath("bio-l2l3-agb-proc-annotations.xsd")
            self.common_xsd = schema_dir.joinpath("bio-common-types.xsd")

            self.schema_files = [
                str(self.l2b_agb_main_ann_xsd),
                str(self.l2l3_common_ann_xsd),
                str(self.l2l3_agb_proc_ann_xsd),
                str(self.common_xsd),
            ]


@dataclass
class BIOMASSL2bAGBProductMeasurement:
    """The L2b AGB product MDS (COG data with metadata)"""

    @dataclass
    class MetadataCOG:
        tile_id_list: list[str]  # One element list in L2B
        basin_id_list: list[str]
        compression: list[int]
        image_description: str
        software: str
        dateTime: str

    latitude_vec: np.ndarray
    longitude_vec: np.ndarray
    data_dict: dict[str, np.ndarray]
    metadata_dict: dict[str, MetadataCOG]


class BIOMASSL2bAGBProduct:
    """_summary_"""

    def __init__(
        self,
        measurement: BIOMASSL2bAGBProductMeasurement,
        main_ads_product: common_annotation_l2l3.ProductL2bL3Type,
        main_ads_raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage,
        main_ads_input_information: common_annotation_l2l3.InputInformationL2BL3ListType,
        main_ads_processing_parameters: main_annotation_l2b_agb.ProcessingParametersAnnotation,
        baseline: int,
        product_doi: str,
        agb_tile_iteration: np.uint8 = np.uint8(1),
        agb_tile_status: agbTileStatus | None = None,
    ) -> None:
        """_summary_"""

        self.measurement = measurement
        self.main_ads_product = main_ads_product
        self.main_ads_raster_image = main_ads_raster_image
        self.main_ads_input_information = main_ads_input_information
        self.baseline = baseline
        self.product_doi = product_doi
        self.agb_tile_iteration = agb_tile_iteration
        self.agb_tile_status = agb_tile_status

        self.product_type = "FP_AGB_L2B"

        self.main_ads_processing_parameters = main_ads_processing_parameters
        self._set_product_name()

        # self.phenomenon_time_start = None
        # self.phenomenon_time_stop = None
        # self.result_time = None
        # self.valid_time_start = None
        # self.valid_time_stop = None

    def _set_product_name(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        self.creation_date = PreciseDateTime.now()
        global_coverage_id_str = encode_product_name_id_value(self.main_ads_product.global_coverage_id, npad=2)

        self.name = "_".join(
            [
                "BIO",
                self.product_type,
                self.main_ads_product.mission_phase_id.value[0],
                "G" + ("__" if self.main_ads_product.mission_phase_id.value[0] == "C" else global_coverage_id_str),
                "T" + f"{self.main_ads_product.tile_id}",
                "B" + f"{self.main_ads_product.basin_id[0][1:4]}",
                ("__" if self.baseline == 0 else f"{self.baseline:02d}"),
                pdt_to_compact_date(self.creation_date),
            ]
        )
