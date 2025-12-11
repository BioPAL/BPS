# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.io import common_annotation_models_l2, main_annotation_models_l2b_fd
from bps.transcoder.utils.production_model_utils import encode_product_name_id_value
from bps.transcoder.utils.time_conversions import pdt_to_compact_date


class BIOMASSL2bFDProductStructure:
    """_summary_"""

    def __init__(self, product_path) -> None:
        """_summary_

        Parameters
        ----------
        product_path : _type_
            _description_
        """

        self.product_path = product_path

        self.product_type = "FP_FD__L2B"

        self.measurement_subfolder = "measurement"
        self.annotation_subfolder = "annotation"
        self.preview_subfolder = "preview"
        self.schema_subfolder = "schema"

        self.mph_file = None
        self.vrt_file = None
        self.fd_file = None
        self.probability_file = None
        self.cfm_file = None
        self.heatmap_file = None
        self.acquisition_id_image_file = None
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
            self.fd_file = glob(os.path.join(self.product_path, self.measurement_subfolder, "*_i_fd.tiff"))[0]

            self.probability_file = glob(
                os.path.join(
                    self.product_path,
                    self.measurement_subfolder,
                    "*_i_probability.tiff",
                )
            )[0]

            self.cfm_file = glob(os.path.join(self.product_path, self.measurement_subfolder, "*_i_cfm.tiff"))[0]
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
            self.mph_file = os.path.join(self.product_path, product_name.lower() + ".xml")
            # - Measurement files
            measurement_file_root = os.path.join(
                self.product_path,
                self.measurement_subfolder,
                product_name.lower()[:-10],
            )
            self.vrt_file = measurement_file_root + "_i.vrt"
            self.fd_file = measurement_file_root + "_i_fd.tiff"
            self.probability_file = measurement_file_root + "_i_probability.tiff"
            self.cfm_file = measurement_file_root + "_i_cfm.tiff"

            # - Annotation files
            annotation_file_root = os.path.join(
                self.product_path, self.annotation_subfolder, product_name.lower()[:-10]
            )
            self.main_annotation_file = annotation_file_root + "_annot.xml"
            self.heatmap_file = annotation_file_root + "_i_heatmap.tiff"
            self.acquisition_id_image_file = annotation_file_root + "_i_acquisition_id_image.tiff"

            # - STAC file
            self.stac_file = os.path.join(self.product_path, product_name.lower()[:-10]) + ".json"
            # - Quick-look files
            preview_file_root = os.path.join(self.product_path, self.preview_subfolder, product_name.lower()[:-10])
            self.quicklook_files = [
                preview_file_root + "_fd_ql.png",
                preview_file_root + "_probability_ql.png",
                preview_file_root + "_cfm_ql.png",
                preview_file_root + "_hmagreeing_ql.png",
                preview_file_root + "_hmcontrib_ql.png",
            ]
            # - Overlay files
            self.overlay_files = [
                preview_file_root + "_fd_map.kml",
                preview_file_root + "_probability_map.kml",
                preview_file_root + "_cfm_map.kml",
                preview_file_root + "_hmagreeing_ql.kml",
                preview_file_root + "_hmcontrib_ql.kml",
            ]

            # - Schema files
            schema_dir = Path(self.product_path, self.schema_subfolder)
            self.l2b_fd_main_ann_xsd = schema_dir.joinpath("bio-l2b-fd-main-annotation.xsd")
            self.l2l3_common_ann_xsd = schema_dir.joinpath("bio-l2l3-common-annotations.xsd")
            self.l2l3_fd_proc_ann_xsd = schema_dir.joinpath("bio-l2l3-fd-proc-annotations.xsd")
            self.common_xsd = schema_dir.joinpath("bio-common-types.xsd")

            self.schema_files = [
                str(self.l2b_fd_main_ann_xsd),
                str(self.l2l3_common_ann_xsd),
                str(self.l2l3_fd_proc_ann_xsd),
                str(self.common_xsd),
            ]


@dataclass
class BIOMASSL2bFDProductMeasurement:
    """The L2b FD product MDS (COG data with metadata)"""

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


@dataclass
class BIOMASSL2bFDMainADSproduct:
    mission: str
    tile_id_list: list[str]
    basin_id_list: list[str]
    product_type: str
    start_time: PreciseDateTime
    stop_time: PreciseDateTime
    radar_carrier_frequency: float
    mission_phase_id: str
    sensor_mode: str
    global_coverage_id: int
    baseline: int


@dataclass
class BIOMASSL2bFDMainADSRasterImage:
    footprint: list[float]
    first_latitude_value: float
    first_longitude_value: float
    latitude_spacing: float
    longitude_spacing: float
    number_of_samples: int
    number_of_lines: int
    projection: str
    datum: common_annotation_models_l2.DatumType
    pixel_representation: common_annotation_models_l2.PixelRepresentationChoiceType
    pixel_type: common_annotation_models_l2.PixelTypeChoiceType
    no_data_value: common_annotation_models_l2.NoDataValueChoiceType


@dataclass
class BIOMASSL2bFDMainADSInputInformation:
    input_information: common_annotation_models_l2.InputInformationL2BL3ListType


@dataclass
class BIOMASSL2bMainADSProcessingParametersFD:
    processor_version: str
    product_generation_time: PreciseDateTime
    compression_options: main_annotation_models_l2b_fd.CompressionOptionsL2B
    minumum_l2a_coverage: float


class BIOMASSL2bFDProduct:
    """_summary_"""

    def __init__(
        self,
        measurement: BIOMASSL2bFDProductMeasurement,
        main_ads_product: BIOMASSL2bFDMainADSproduct,
        main_ads_raster_image: BIOMASSL2bFDMainADSRasterImage,
        main_ads_input_information: BIOMASSL2bFDMainADSInputInformation,
        main_ads_processing_parameters: BIOMASSL2bMainADSProcessingParametersFD,
        product_doi: str,
    ) -> None:
        """_summary_"""

        self.measurement = measurement
        self.main_ads_product = main_ads_product
        self.main_ads_raster_image = main_ads_raster_image
        self.main_ads_input_information = main_ads_input_information
        self.product_doi = product_doi
        self.product_type = "FP_FD__L2B"

        self.main_ads_processing_parameters = main_ads_processing_parameters
        self._set_product_name()

    def _set_product_name(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        self.creation_date = PreciseDateTime.now()
        baseline_id = 0
        if self.main_ads_product.baseline is not None:
            baseline_id = self.main_ads_product.baseline
        global_coverage_id_str = encode_product_name_id_value(self.main_ads_product.global_coverage_id, npad=2)

        self.name = "_".join(
            [
                "BIO",
                self.product_type,
                str(self.main_ads_product.mission_phase_id)[0],
                "G" + ("__" if self.main_ads_product.mission_phase_id[0] == "C" else global_coverage_id_str),
                "T" + f"{self.main_ads_product.tile_id_list[0]}",
                "B" + f"{self.main_ads_product.basin_id_list[0][1:4]}",
                ("__" if baseline_id == 0 else f"{baseline_id:02d}"),
                pdt_to_compact_date(self.creation_date),
            ]
        )
