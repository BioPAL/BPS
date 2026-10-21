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
from bps.common.io.mph import get_mph_path
from bps.transcoder.io.common_annotation_l2l3 import (
    BIOMASSL2aMainADSInputInformation,
    BIOMASSL2aMainADSproduct,
    BIOMASSL2aMainADSRasterImage,
)
from bps.transcoder.io.main_annotation_l2a_fd import BIOMASSL2aMainADSProcessingParametersFD
from bps.transcoder.io.main_annotation_l2a_fh import BIOMASSL2aMainADSProcessingParametersFH
from bps.transcoder.io.main_annotation_l2a_gn import BIOMASSL2aMainADSProcessingParametersGN
from bps.transcoder.io.main_annotation_l2a_tfh import BIOMASSL2aMainADSProcessingParametersTOMOFH
from bps.transcoder.utils.production_model_utils import (
    encode_product_name_id_value,
    translate_com_phase_negative_values,
)
from bps.transcoder.utils.time_conversions import (
    pdt_to_compact_date,
    pdt_to_compact_string,
)


class BIOMASSL2aProductStructure:
    """_summary_"""

    def __init__(self, product_path, product_type: str) -> None:
        """_summary_

        Parameters
        ----------
        product_path : _type_
            _description_
        product_type : _type_
            _description_
        """
        assert product_type in ["FP_FD__L2A", "FP_FH__L2A", "FP_GN__L2A", "FP_TFH_L2A"]

        self.product_path = product_path
        self.product_type = product_type

        self.measurement_subfolder = "measurement"
        self.annotation_subfolder = "annotation"
        self.preview_subfolder = "preview"
        self.schema_subfolder = "schema"

        self.mph_file = None
        self.stac_file = None
        self.measurement_files = None
        self.vrt_file = None
        self.main_annotation_file = None
        self.lut_annotation_file = None
        self.quicklook_files = None
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
            # Set paths of BIOMASS L2a product single files starting from an existing SAR product
            # - MPH file
            self.mph_file = (glob(os.path.join(self.product_path, "*.xml")) or [None])[0]
            # - STAC file
            self.stac_file = (glob(os.path.join(self.product_path, "*.json")) or [None])[0]
            # - Measurement files
            self.measurement_files = glob(os.path.join(self.product_path, self.measurement_subfolder, "*.tiff"))
            self.vrt_file = (glob(os.path.join(self.product_path, self.measurement_subfolder, "*_i.vrt")) or [None])[0]
            # - Annotation files
            self.main_annotation_file = (
                glob(os.path.join(self.product_path, self.annotation_subfolder, "*_annot.xml")) or [None]
            )[0]
            self.lut_annotation_file = (
                glob(os.path.join(self.product_path, self.annotation_subfolder, "*_lut.nc")) or [None]
            )[0]
            # - Quick-look files
            self.quicklook_files = glob(os.path.join(self.product_path, self.preview_subfolder, "*_ql.png")) or [None]
            # - Overlay files
            self.overlay_files = glob(os.path.join(self.product_path, self.preview_subfolder, "*_map.kml")) or [None]
            # - Schema files
            self.schema_files = glob(os.path.join(self.product_path, self.schema_subfolder, "*.xsd"))
        else:
            # Set paths of BIOMASS L1 product single files starting from product name
            product_name = os.path.basename(self.product_path)
            # - MPH file
            self.mph_file = str(get_mph_path(Path(self.product_path)))
            # - STAC file
            self.stac_file = os.path.join(self.product_path, product_name.lower()[:-10] + ".json")
            # - Measurement files
            measurement_file_root = os.path.join(
                self.product_path,
                self.measurement_subfolder,
                product_name.lower()[:-10],
            )
            self.vrt_file = measurement_file_root + "_i.vrt"

            # - Measurement files
            if self.product_type == "FP_FD__L2A":
                self.measurement_files = [
                    measurement_file_root + "_i_fd.tiff",
                    measurement_file_root + "_i_probability.tiff",
                    measurement_file_root + "_i_cfm.tiff",
                ]
            if self.product_type == "FP_FH__L2A":
                self.measurement_files = [
                    measurement_file_root + "_i_fh.tiff",
                    measurement_file_root + "_i_quality.tiff",
                ]
            if self.product_type == "FP_GN__L2A":
                self.measurement_files = [
                    measurement_file_root + "_i_gn.tiff",
                ]
            if self.product_type == "FP_TFH_L2A":
                self.measurement_files = [
                    measurement_file_root + "_i_fh.tiff",
                    measurement_file_root + "_i_quality.tiff",
                ]
            # - Annotation files
            annotation_file_root = os.path.join(
                self.product_path, self.annotation_subfolder, product_name.lower()[:-10]
            )
            self.main_annotation_file = annotation_file_root + "_annot.xml"
            self.lut_annotation_file = annotation_file_root + "_lut.nc"

            # - Quick-look files and Overlay files
            preview_file_root = os.path.join(self.product_path, self.preview_subfolder, product_name.lower()[:-10])
            if self.product_type == "FP_FD__L2A":
                self.quicklook_files = [
                    preview_file_root + "_fd_ql.png",
                    preview_file_root + "_probability_ql.png",
                    preview_file_root + "_cfm_ql.png",
                ]
                self.overlay_files = [
                    preview_file_root + "_fd_map.kml",
                    preview_file_root + "_probability_map.kml",
                    preview_file_root + "_cfm_map.kml",
                ]
            if self.product_type == "FP_FH__L2A":
                self.quicklook_files = [
                    preview_file_root + "_fh_ql.png",
                    preview_file_root + "_fhquality_ql.png",
                ]
                self.overlay_files = [
                    preview_file_root + "_fh_map.kml",
                    preview_file_root + "_fhquality_map.kml",
                ]
            if self.product_type == "FP_GN__L2A":
                self.quicklook_files = [
                    preview_file_root + "_gn_ql.png",
                ]
                self.overlay_files = [
                    preview_file_root + "_gn_map.kml",
                ]
            if self.product_type == "FP_TFH_L2A":
                self.quicklook_files = [
                    preview_file_root + "_fh_ql.png",
                    preview_file_root + "_fhquality_ql.png",
                ]
                self.overlay_files = [
                    preview_file_root + "_fh_map.kml",
                    preview_file_root + "_fhquality_map.kml",
                ]

            # - Schema files
            schema_dir = Path(self.product_path, self.schema_subfolder)
            self.l2l3_common_ann_xsd = schema_dir.joinpath("bio-l2l3-common-annotations.xsd")
            self.common_xsd = schema_dir.joinpath("bio-common-types.xsd")

            if self.product_type == "FP_FD__L2A":
                self.l2a_fd_main_ann_xsd = schema_dir.joinpath("bio-l2a-fd-main-annotation.xsd")
                self.l2l3_fd_proc_ann_xsd = schema_dir.joinpath("bio-l2l3-fd-proc-annotations.xsd")

                self.schema_files = [
                    str(self.l2a_fd_main_ann_xsd),
                    str(self.l2l3_common_ann_xsd),
                    str(self.l2l3_fd_proc_ann_xsd),
                    str(self.common_xsd),
                ]

            if self.product_type == "FP_FH__L2A":
                self.l2a_fh_main_ann_xsd = schema_dir.joinpath("bio-l2a-fh-main-annotation.xsd")
                self.l2l3_fh_proc_ann_xsd = schema_dir.joinpath("bio-l2l3-fh-proc-annotations.xsd")

                self.schema_files = [
                    str(self.l2a_fh_main_ann_xsd),
                    str(self.l2l3_common_ann_xsd),
                    str(self.l2l3_fh_proc_ann_xsd),
                    str(self.common_xsd),
                ]

            if self.product_type == "FP_GN__L2A":
                self.l2a_gn_main_ann_xsd = schema_dir.joinpath("bio-l2a-gn-main-annotation.xsd")
                self.l2l3_agb_proc_ann_xsd = schema_dir.joinpath("bio-l2l3-agb-proc-annotations.xsd")

                self.schema_files = [
                    str(self.l2a_gn_main_ann_xsd),
                    str(self.l2l3_common_ann_xsd),
                    str(self.l2l3_agb_proc_ann_xsd),
                    str(self.common_xsd),
                ]

            if self.product_type == "FP_TFH_L2A":
                self.l2a_tomo_fh_main_ann_xsd = schema_dir.joinpath("bio-l2a-tfh-main-annotation.xsd")
                self.l2l3_tomo_fh_proc_ann_xsd = schema_dir.joinpath("bio-l2l3-tfh-proc-annotations.xsd")

                self.schema_files = [
                    str(self.l2a_tomo_fh_main_ann_xsd),
                    str(self.l2l3_common_ann_xsd),
                    str(self.l2l3_tomo_fh_proc_ann_xsd),
                    str(self.common_xsd),
                ]


@dataclass
class BIOMASSL2aProductMeasurement:
    """The L2a product MDS (COG data with metadata)"""

    @dataclass
    class MetadataCOG:
        swath: str
        tile_id_list: list[str]
        basin_id_list: list[str]
        compression: list[int]
        image_description: str
        software: str
        dateTime: str

    latitude_vec: np.ndarray
    longitude_vec: np.ndarray
    data_dict: dict[str, list[np.ndarray]]
    metadata_dict: dict[str, MetadataCOG]


@dataclass
class BIOMASSL2aLutAdsFD:
    @dataclass
    class LutMetadata:
        first_sample: int
        first_line: int
        samples_interval: int
        lines_interval: int
        pixelType: str
        no_data_value: int | float
        projection: str
        coordinateReferenceSystem: str
        geodeticReferenceFrame: str
        least_significant_digit: int | None = None

    lut_fnf: np.ndarray
    lut_acm: dict  # Dict with keys "layer1", "layer2"..., "layer9"
    lut_number_of_averages: np.ndarray
    # Geometryc luts
    lut_latitude: np.ndarray
    lut_longitude: np.ndarray
    lut_dem: np.ndarray
    lut_incidence_angle: np.ndarray
    lut_elevation_angle: np.ndarray
    lut_range_dem_slope: np.ndarray
    lut_azimuth_dem_slope: np.ndarray
    # Metadata
    lut_fnf_metadata: LutMetadata
    lut_acm_metadata: LutMetadata
    lut_number_of_averages_metadata: LutMetadata
    lut_geometric_metadata: LutMetadata


@dataclass
class BIOMASSL2aLutAdsFH:
    @dataclass
    class LutMetadata:
        first_sample: int
        first_line: int
        samples_interval: int
        lines_interval: int
        pixelType: str
        no_data_value: int | float
        projection: str
        coordinateReferenceSystem: str
        geodeticReferenceFrame: str
        least_significant_digit: int | None = None

    lut_fnf: np.ndarray
    # Geometric luts
    lut_latitude: np.ndarray
    lut_longitude: np.ndarray
    lut_dem: np.ndarray
    lut_incidence_angle: np.ndarray
    lut_elevation_angle: np.ndarray
    lut_range_dem_slope: np.ndarray
    lut_azimuth_dem_slope: np.ndarray
    # Metadata
    lut_fnf_metadata: LutMetadata
    lut_geometric_metadata: LutMetadata


@dataclass
class BIOMASSL2aLutAdsGN:
    @dataclass
    class LutMetadata:
        first_sample: int
        first_line: int
        samples_interval: int
        lines_interval: int
        pixelType: str
        no_data_value: int | float
        projection: str
        coordinateReferenceSystem: str
        geodeticReferenceFrame: str
        least_significant_digit: int | None = None

    lut_fnf: np.ndarray
    lut_local_incidence_angle: np.ndarray
    # Geometric luts
    lut_latitude: np.ndarray
    lut_longitude: np.ndarray
    lut_dem: np.ndarray
    lut_incidence_angle: np.ndarray
    lut_elevation_angle: np.ndarray
    lut_range_dem_slope: np.ndarray
    lut_azimuth_dem_slope: np.ndarray
    # Metadata
    lut_fnf_metadata: LutMetadata
    lut_geometric_metadata: LutMetadata


@dataclass
class BIOMASSL2aLutAdsTOMOFH:
    @dataclass
    class LutMetadata:
        first_sample: int
        first_line: int
        samples_interval: int
        lines_interval: int
        pixelType: str
        no_data_value: int | float
        projection: str
        coordinateReferenceSystem: str
        geodeticReferenceFrame: str
        least_significant_digit: int | None = None

    lut_fnf: np.ndarray
    # Geometryc luts
    lut_latitude: np.ndarray
    lut_longitude: np.ndarray
    lut_dem: np.ndarray
    lut_incidence_angle: np.ndarray
    lut_elevation_angle: np.ndarray
    lut_range_dem_slope: np.ndarray
    lut_azimuth_dem_slope: np.ndarray
    # Metadata
    lut_fnf_metadata: LutMetadata
    lut_geometric_metadata: LutMetadata


class BIOMASSL2aProduct:
    """_summary_"""

    def __init__(
        self,
        measurement: BIOMASSL2aProductMeasurement,
        main_ads_product: BIOMASSL2aMainADSproduct,
        main_ads_raster_image: BIOMASSL2aMainADSRasterImage,
        main_ads_input_information: BIOMASSL2aMainADSInputInformation,
    ) -> None:
        """_summary_"""

        self.measurement = measurement
        self.main_ads_product = main_ads_product
        self.main_ads_raster_image = main_ads_raster_image
        self.main_ads_input_information = main_ads_input_information

        self.product_type = ""  # will be set inside child class BIOMASSL2aProductFD, BIOMASSL2aProductFH, BIOMASSL2aProductGN, BIOMASSL2aProductTOMOFH

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
        major_cycle_id_str = encode_product_name_id_value(self.main_ads_product.major_cycle_id, npad=2)

        self.name = "_".join(
            [
                "BIO",
                self.product_type,
                pdt_to_compact_string(self.main_ads_product.start_time),
                pdt_to_compact_string(self.main_ads_product.stop_time),
                str(self.main_ads_product.mission_phase_id)[0],
                "G" + ("__" if self.main_ads_product.mission_phase_id.value[0] == "C" else global_coverage_id_str),
                "M" + ("__" if self.main_ads_product.mission_phase_id.value[0] == "C" else major_cycle_id_str),
                "C__",
                "T"
                + (
                    "___"
                    if self.main_ads_product.mission_phase_id.value[0] == "C"
                    else f"{translate_com_phase_negative_values(self.main_ads_product.relative_orbit_number):03d}"
                ),
                "F" + ("___" if self.main_ads_product.frame == 0 else f"{self.main_ads_product.frame:03d}"),
                ("__" if baseline_id == 0 else f"{baseline_id:02d}"),
                pdt_to_compact_date(self.creation_date),
            ]
        )


class BIOMASSL2aProductFD(BIOMASSL2aProduct):
    def __init__(
        self,
        measurement: BIOMASSL2aProductMeasurement,
        main_ads_product: BIOMASSL2aMainADSproduct,
        main_ads_raster_image: BIOMASSL2aMainADSRasterImage,
        main_ads_input_information: BIOMASSL2aMainADSInputInformation,
        main_ads_processing_parameters: BIOMASSL2aMainADSProcessingParametersFD,
        lut_ads: BIOMASSL2aLutAdsFD,
        product_doi: str,
    ):
        super().__init__(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
        )

        self.product_type = "FP_FD__L2A"
        self.main_ads_processing_parameters = main_ads_processing_parameters
        self.lut_ads = lut_ads
        self.product_doi = product_doi
        self._set_product_name()


class BIOMASSL2aProductFH(BIOMASSL2aProduct):
    def __init__(
        self,
        measurement: BIOMASSL2aProductMeasurement,
        main_ads_product: BIOMASSL2aMainADSproduct,
        main_ads_raster_image: BIOMASSL2aMainADSRasterImage,
        main_ads_input_information: BIOMASSL2aMainADSInputInformation,
        main_ads_processing_parameters: BIOMASSL2aMainADSProcessingParametersFH,
        lut_ads: BIOMASSL2aLutAdsFH,
        product_doi: str,
    ):
        super().__init__(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
        )

        self.product_type = "FP_FH__L2A"
        self.main_ads_processing_parameters = main_ads_processing_parameters
        self.lut_ads = lut_ads
        self.product_doi = product_doi
        self._set_product_name()


class BIOMASSL2aProductGN(BIOMASSL2aProduct):
    def __init__(
        self,
        measurement: BIOMASSL2aProductMeasurement,
        main_ads_product: BIOMASSL2aMainADSproduct,
        main_ads_raster_image: BIOMASSL2aMainADSRasterImage,
        main_ads_input_information: BIOMASSL2aMainADSInputInformation,
        main_ads_processing_parameters: BIOMASSL2aMainADSProcessingParametersGN,
        lut_ads: BIOMASSL2aLutAdsGN,
        product_doi: str,
    ):
        super().__init__(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
        )

        self.product_type = "FP_GN__L2A"
        self.main_ads_processing_parameters = main_ads_processing_parameters
        self.lut_ads = lut_ads
        self.product_doi = product_doi
        self._set_product_name()


class BIOMASSL2aProductTOMOFH(BIOMASSL2aProduct):
    def __init__(
        self,
        measurement: BIOMASSL2aProductMeasurement,
        main_ads_product: BIOMASSL2aMainADSproduct,
        main_ads_raster_image: BIOMASSL2aMainADSRasterImage,
        main_ads_input_information: BIOMASSL2aMainADSInputInformation,
        main_ads_processing_parameters: BIOMASSL2aMainADSProcessingParametersTOMOFH,
        lut_ads: BIOMASSL2aLutAdsTOMOFH,
        product_doi: str,
    ):
        super().__init__(
            measurement,
            main_ads_product,
            main_ads_raster_image,
            main_ads_input_information,
        )

        self.product_type = "FP_TFH_L2A"
        self.main_ads_processing_parameters = main_ads_processing_parameters
        self.lut_ads = lut_ads
        self.product_doi = product_doi
        self._set_product_name()
