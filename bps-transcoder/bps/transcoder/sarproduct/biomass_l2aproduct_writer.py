# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
import pystac
from bps.common import bps_logger
from bps.common.io import common_types
from bps.common.io.parsing import serialize
from bps.common.l2_joborder_tags import (
    L2A_OUTPUT_PRODUCT_FD,
    L2A_OUTPUT_PRODUCT_FH,
    L2A_OUTPUT_PRODUCT_GN,
    L2A_OUTPUT_PRODUCT_TFH,
)
from bps.transcoder import BPS_PFD_AGB_VERSION, BPS_PFD_FD_VERSION, BPS_PFD_FH_VERSION, BPS_PPD_VERSION
from bps.transcoder.io import (
    common_annotation_models_l2,
    main_annotation_models_l2a_fd,
    main_annotation_models_l2a_fh,
    main_annotation_models_l2a_gn,
    main_annotation_models_l2a_tfh,
    vrt,
)
from bps.transcoder.sarproduct.biomass_l2aproduct import (
    BIOMASSL2aProductFD,
    BIOMASSL2aProductFH,
    BIOMASSL2aProductGN,
    BIOMASSL2aProductStructure,
    BIOMASSL2aProductTOMOFH,
)
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES
from bps.transcoder.sarproduct.overlay import write_overlay_file
from bps.transcoder.utils.gdal_utils import GeotiffConf, GeotiffMetadata, write_geotiff
from bps.transcoder.utils.production_model_utils import (
    encode_mph_id_value,
    translate_com_phase_negative_values,
    translate_frame_id,
    translate_global_coverage_id,
    translate_major_cycle_id,
)
from bps.transcoder.utils.xsd_schema_attacher import copy_biomass_xsd_files
from netCDF4 import Dataset
from scipy.signal import convolve2d

for key, value in MPH_NAMESPACES.items():
    ET.register_namespace(key, value)


COMPRESSION_SCHEMA_MDS_LERC_ZSTD = "LERC_ZSTD"
COMPRESSION_SCHEMA_MDS_ZSTD = "ZSTD"
COMPRESSION_SCHEMA_ADS = "zlib"
COMPRESSION_EXIF_CODES_LERC_ZSTD = [34887, 34926]  #  LERC, ZSTD
FLOAT_NODATA_VALUE = float(-9999.0)
INT_NODATA_VALUE = int(255)
DECIMATION_FACTOR_QUICKLOOKS = 2
AVERAGING_FACTOR_QUICKLOOKS = 2


class BIOMASSL2aProductWriter:
    """_summary_"""

    def __init__(
        self,
        product: (
            BIOMASSL2aProductFD | BIOMASSL2aProductFH | BIOMASSL2aProductGN | BIOMASSL2aProductTOMOFH
        ),  # product to write
        product_path: Path,  # from job order
        processor_name: str,  # from job order
        processor_version: str,  # from job order
        input_stack_acquisitions: list[str],  # from job order
        gcp_list: list,
        aux_pp2_2a_name: str,  # from job order
        fnf_directory_name: str,  # from job order
        footprint_mask_for_quicklooks: np.ndarray,
        aux_fp_fd_l2a_name: (str | None) = None,  # Optional input for FP_FD__L2A, FP_FH__L2A, FP_TFH_L2A processors
    ) -> None:
        """_summary_

        Parameters
        ----------
        product : BIOMASSL2aProduct
            _description_
        product_path : _type_
            _description_
        processor_name : str
            _description_
        processor_version : str
            _description_
        input_stack_acquisitions: List[str]
            _description_
        aux_pp2_2a_name: str
            _description_
        fnf_directory_name: str
            _description_
        footprint_mask_for_quicklooks: np.ndarray
            _description_
        aux_fp_fd_l2a_name: Optional[str]
            _description_
        """
        self.product = product
        self.product_path = product_path
        self.processor_name = processor_name
        self.processor_version = processor_version
        self.input_stack_acquisitions = input_stack_acquisitions
        self.gcp_list = gcp_list
        self.aux_pp2_2a_name = aux_pp2_2a_name
        self.fnf_directory_name = fnf_directory_name
        self.aux_fp_fd_l2a_name = aux_fp_fd_l2a_name
        self.footprint_mask_for_quicklooks = footprint_mask_for_quicklooks

        # output path full and check
        self.product_path = self.product_path.joinpath(self.product.name)
        self._check_product_path()

        # folder structure of the product
        self.product_structure = BIOMASSL2aProductStructure(self.product_path, self.product.product_type)

    def _check_product_path(self):
        """_summary_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        FileExistsError
            _description_
        """
        if self.product_path.exists():
            raise FileExistsError(f"Folder {self.product_path} already exists.")

    def _init_product_structure(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        bps_logger.info(f"Writing output to {self.product_path} :")
        self.product_path.mkdir(parents=True, exist_ok=True)
        self.product_path.joinpath(self.product_structure.measurement_subfolder).mkdir(parents=True, exist_ok=True)
        self.product_path.joinpath(self.product_structure.annotation_subfolder).mkdir(parents=True, exist_ok=True)
        self.product_path.joinpath(self.product_structure.schema_subfolder).mkdir(parents=True, exist_ok=True)

    def _write_mph_file(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        name = self.product.name
        id_counter = 0

        footprint_str = ""
        footprint_closed = (
            self.product.main_ads_input_information.footprint + self.product.main_ads_input_information.footprint[0:2]
        )

        for num in footprint_closed:
            footprint_str = footprint_str + str(num) + " "
        footprint_str = footprint_str[0:-1]

        # Fill MPH file
        id_counter += 1
        xml1 = ET.Element(
            "{" + MPH_NAMESPACES["bio"] + "}EarthObservation",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}phenomenonTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimePeriod",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}beginPosition")
        xml4.text = self.product.main_ads_product.start_time.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.product.main_ads_product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}resultTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimeInstant",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}timePosition")
        xml4.text = self.product.main_ads_product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}validTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimePeriod",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}beginPosition")
        xml4.text = self.product.main_ads_product.start_time.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.product.main_ads_product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}procedure")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["eop"] + "}EarthObservationEquipment",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}platform")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["eop"] + "}Platform")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}shortName")
        xml6.text = "Biomass"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}instrument")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["eop"] + "}Instrument")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}shortName")
        xml6.text = "P-SAR"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}sensor")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["eop"] + "}Sensor")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}sensorType")
        xml6.text = "RADAR"
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}operationalMode",
            attrib={"codeSpace": "urn:esa:eop:Biomass:P-SAR:operationalMode"},
        )
        xml6.text = "SM"
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}swathIdentifier",
            attrib={"codeSpace": "urn:esa:eop:Biomass:P-SAR:swathIdentifier"},
        )
        xml6.text = self.product.main_ads_product.swath
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}acquisitionParameters")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["bio"] + "}Acquisition")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}orbitDirection")
        xml6.text = self.product.main_ads_product.orbit_pass.upper()
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}wrsLongitudeGrid",
            attrib={"codeSpace": "urn:esa:eop:Biomass:relativeOrbits"},
        )
        xml6.text = (
            "0"
            if self.product.main_ads_product.relative_orbit_number <= 0
            else str(self.product.main_ads_product.relative_orbit_number)
        )  # track number
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}wrsLatitudeGrid",
            attrib={"codeSpace": "urn:esa:eop:Biomass:frames"},
        )
        xml6.text = (
            "___" if self.product.main_ads_product.frame <= 0 else str(self.product.main_ads_product.frame)
        )  # slice
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["sar"] + "}antennaLookDirection")
        xml6.text = "LEFT"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}missionPhase")
        mission_phases_dict = {
            "INT": "INTERFEROMETRIC",
            "TOM": "TOMOGRAPHIC",
            "COM": "COMMISSIONING",
        }
        xml6.text = mission_phases_dict[str(self.product.main_ads_product.mission_phase_id)]
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}globalCoverageID")
        xml6.text = encode_mph_id_value(self.product.main_ads_product.global_coverage_id)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}majorCycleID")
        xml6.text = encode_mph_id_value(self.product.main_ads_product.major_cycle_id)

        for tile_id in self.product.main_ads_product.tile_id_list:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}tileID")
            xml6.text = str(tile_id)

        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}basinID")
        xml6.text = str(self.product.main_ads_product.basin_id_list[0])

        xml2 = ET.SubElement(
            xml1,
            "{" + MPH_NAMESPACES["om"] + "}observedProperty",
            attrib={
                "{" + MPH_NAMESPACES["xsi"] + "}nil": "true",
                "nilReason": "inapplicable",
            },
        )

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}featureOfInterest")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["eop"] + "}Footprint",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}multiExtentOf")
        id_counter += 1
        xml5 = ET.SubElement(
            xml4,
            "{" + MPH_NAMESPACES["gml"] + "}MultiSurface",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["gml"] + "}surfaceMember")
        id_counter += 1
        xml7 = ET.SubElement(
            xml6,
            "{" + MPH_NAMESPACES["gml"] + "}Polygon",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml8 = ET.SubElement(xml7, "{" + MPH_NAMESPACES["gml"] + "}exterior")
        xml9 = ET.SubElement(xml8, "{" + MPH_NAMESPACES["gml"] + "}LinearRing")
        xml10 = ET.SubElement(xml9, "{" + MPH_NAMESPACES["gml"] + "}posList")
        xml10.text = footprint_str
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}centerOf")
        id_counter += 1
        xml5 = ET.SubElement(
            xml4,
            "{" + MPH_NAMESPACES["gml"] + "}Point",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["gml"] + "}pos")
        footprint_center = [
            np.mean(
                [
                    self.product.main_ads_input_information.footprint[0],
                    self.product.main_ads_input_information.footprint[3],
                ]
            ),
            np.mean(
                [
                    self.product.main_ads_input_information.footprint[1],
                    self.product.main_ads_input_information.footprint[2],
                ]
            ),
        ]  # NE, SE, SW, NW
        footprint_center_str = str(footprint_center[0]) + " " + str(footprint_center[1])
        xml6.text = str(footprint_center_str).replace(",", "")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}result")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["eop"] + "}EarthObservationResult",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}product")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["bio"] + "}ProductInformation")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}fileName")
        xml7 = ET.SubElement(
            xml6,
            "{" + MPH_NAMESPACES["ows"] + "}ServiceReference",
            attrib={"{" + MPH_NAMESPACES["xlink"] + "}href": self.product.name},
        )
        xml8 = ET.SubElement(xml7, "{" + MPH_NAMESPACES["ows"] + "}RequestMessage")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}version")
        xml6.text = f"{self.product.main_ads_product.baseline:02d}"

        def add_file_to_mph(
            base_element: ET.Element,
            file: Path,
            product_path: Path,
            rds_file: Path | None,
        ):
            """Add product file to mph list"""
            relative_path = file.relative_to(product_path)
            file_size = file.stat().st_size

            product_elem = ET.SubElement(base_element, "{" + MPH_NAMESPACES["eop"] + "}product")
            product_information_elem = ET.SubElement(product_elem, "{" + MPH_NAMESPACES["bio"] + "}ProductInformation")
            file_name_elem = ET.SubElement(product_information_elem, "{" + MPH_NAMESPACES["eop"] + "}fileName")
            service_reference_elem = ET.SubElement(
                file_name_elem,
                "{" + MPH_NAMESPACES["ows"] + "}ServiceReference",
                attrib={"{" + MPH_NAMESPACES["xlink"] + "}href": "./" + str(relative_path)},
            )
            ET.SubElement(service_reference_elem, "{" + MPH_NAMESPACES["ows"] + "}RequestMessage")
            size_elem = ET.SubElement(
                product_information_elem,
                "{" + MPH_NAMESPACES["eop"] + "}size",
                attrib={"uom": "bytes"},
            )
            size_elem.text = str(file_size)
            if rds_file is not None:
                rel_path = Path(rds_file).relative_to(product_path)
                rds_elem = ET.SubElement(product_information_elem, "{" + MPH_NAMESPACES["bio"] + "}rds")
                rds_elem.text = "./" + str(rel_path)

        assert self.product_structure.measurement_files is not None

        files: list[tuple[str | None, Path | None]] = [
            (file, None) for file in self.product_structure.measurement_files
        ]

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            files += [
                (
                    self.product_structure.main_annotation_file,
                    self.product_structure.l2a_fd_main_ann_xsd,
                )
            ]
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FH:
            files += [
                (
                    self.product_structure.main_annotation_file,
                    self.product_structure.l2a_fh_main_ann_xsd,
                )
            ]
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            files += [
                (
                    self.product_structure.main_annotation_file,
                    self.product_structure.l2a_gn_main_ann_xsd,
                )
            ]
        if self.product.product_type == L2A_OUTPUT_PRODUCT_TFH:
            files += [
                (
                    self.product_structure.main_annotation_file,
                    self.product_structure.l2a_tomo_fh_main_ann_xsd,
                )
            ]

        files += (
            [(self.product_structure.lut_annotation_file, None)]
            + [(file, None) for file in self.product_structure.quicklook_files]
            + [(self.product_structure.stac_file, None)]
        )

        for file, xsd_file in files:
            file_path = Path(file) if file is not None else None
            if file_path is not None and file_path.exists():
                add_file_to_mph(xml3, file_path, self.product_path, xsd_file)

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["eop"] + "}metaDataProperty")
        xml3 = ET.SubElement(xml2, "{" + MPH_NAMESPACES["bio"] + "}EarthObservationMetaData")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}identifier")
        xml4.text = self.product.name
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}doi")
        xml4.text = self.product.product_doi
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}acquisitionType")
        xml4.text = "NOMINAL"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}productType")

        if L2A_OUTPUT_PRODUCT_FD in self.product_structure.schema_files[0]:
            xml4.text = L2A_OUTPUT_PRODUCT_FD
        if L2A_OUTPUT_PRODUCT_FH in self.product_structure.schema_files[0]:
            xml4.text = L2A_OUTPUT_PRODUCT_FH
        if L2A_OUTPUT_PRODUCT_GN in self.product_structure.schema_files[0]:
            xml4.text = L2A_OUTPUT_PRODUCT_GN
        if L2A_OUTPUT_PRODUCT_TFH in self.product_structure.schema_files[0]:
            xml4.text = L2A_OUTPUT_PRODUCT_TFH

        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}status")
        xml4.text = "ARCHIVED"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}processing")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["bio"] + "}ProcessingInformation")
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}processingCenter",
            attrib={"codeSpace": "urn:esa:eop:Biomass:facility"},
        )
        xml6.text = "ARESYS"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}processingDate")
        xml6.text = self.product.creation_date.isoformat(timespec="seconds")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}processorName")
        xml6.text = self.processor_name
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}processorVersion")
        xml6.text = self.processor_version
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}processingLevel")
        xml6.text = "other: L2a"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
        xml6.text = self.aux_pp2_2a_name
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
        xml6.text = self.fnf_directory_name
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}processingMode",
            attrib={"codeSpace": "urn:esa:eop:Biomass:P-SAR:processingMode"},
        )
        xml6.text = "OPERATIONAL"

        if self.aux_fp_fd_l2a_name is not None:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}sourceProduct")
            xml6.text = self.aux_fp_fd_l2a_name

        for stack_acquisition_name in self.input_stack_acquisitions:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}sourceProduct")
            xml6.text = stack_acquisition_name

        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
        if L2A_OUTPUT_PRODUCT_FD in self.product_structure.schema_files[0]:
            xml4.text = f"BIOMASS Forest Disturbance Product Format Specification (BPS_FD_PFD) {BPS_PFD_FD_VERSION}"
        if L2A_OUTPUT_PRODUCT_FH in self.product_structure.schema_files[0]:
            xml4.text = f"BIOMASS Forest Height Product Format Specification (BPS_FH_PFD) {BPS_PFD_FH_VERSION}"
        if L2A_OUTPUT_PRODUCT_GN in self.product_structure.schema_files[0]:
            xml4.text = f"BIOMASS Above Ground Biomass Product Format Specification (BPS_AGB_PFD) {BPS_PFD_AGB_VERSION}"
        if L2A_OUTPUT_PRODUCT_TFH in self.product_structure.schema_files[0]:
            xml4.text = "not provided"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
        if L2A_OUTPUT_PRODUCT_TFH in self.product_structure.schema_files[0]:
            xml4.text = "not provided"
        else:
            xml4.text = f"BIOMASS Product Performance Description (BPS_PPD) {BPS_PPD_VERSION}"

        # Write MPH file
        xmlstr = minidom.parseString(ET.tostring(xml1)).toprettyxml(indent="   ")
        assert self.product_structure.mph_file is not None
        with open(file=self.product_structure.mph_file, mode="w", encoding="utf-8") as f:
            f.write(xmlstr)

    def _write_stac_file(self):
        """Write STAC file
        The STAC file will be a .json file, containing a simple Item:
        https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md
        """

        # prepare STAC item content
        # id
        id = self.product_path.name
        # geometry
        footprint = (
            self.product.main_ads_raster_image.footprint
        )  # [sw_lon, sw_lat, nw_lon, nw_lat, ne_lon, ne_lat, se_lon, se_lat]
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [footprint[0], footprint[1]],
                    [footprint[2], footprint[3]],
                    [footprint[4], footprint[5]],
                    [footprint[6], footprint[7]],
                    [footprint[0], footprint[1]],  # copy first entry at the end
                ]
            ],
        }
        # bbox
        lat_min = min(footprint[0::2])
        lat_max = max(footprint[0::2])
        lon_min = min(footprint[1::2])
        lon_max = max(footprint[1::2])
        bbox = [lon_min, lat_min, lon_max, lat_max]
        # properties
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            type_str = "Forest Disturbance"
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FH:
            type_str = "Forest Height"
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            type_str = "Ground Cancelled"
        if self.product.product_type == L2A_OUTPUT_PRODUCT_TFH:
            type_str = "Tomo Forest Height"

        # times
        start_time = self.product.main_ads_product.start_time.isoformat(timespec="milliseconds")
        stop_time = self.product.main_ads_product.stop_time.isoformat(timespec="milliseconds")
        product_generation_time = self.product.main_ads_processing_parameters.product_generation_time.isoformat(
            timespec="milliseconds"
        )

        proj_shape = list(next(iter(self.product.measurement.data_dict.values()))[0].shape)
        properties = {
            "platform": "BIOMASS",
            "instrument": "p-sar",
            "processing:software": "bps",
            "processing:software_version": self.product.main_ads_processing_parameters.processor_version,
            "processing:level": "L2A",
            "processing:product_type": type_str,
            "tiles_id": self.product.main_ads_product.tile_id_list,
            "start_datetime": start_time,
            "end_datetime": stop_time,
            "global_cycle": id[50:52],
            "created": product_generation_time,
            "proj:shape": proj_shape,
        }

        product_root = Path(self.product_structure.measurement_files[0]).parent.parent
        measurement_file_root = product_root.joinpath(self.product_structure.measurement_subfolder)
        preview_file_root = product_root.joinpath(self.product_structure.preview_subfolder)

        # link.set_owner(owner="")
        # assets
        assets_dict = {}

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            assets_dict["fd"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_fd.tiff").relative_to(
                        product_root
                    )
                ),
                title="Forest Disturbance",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["probability"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_probability.tiff").relative_to(
                        product_root
                    )
                ),
                title="Probability of Change",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["cfm"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_cfm.tiff").relative_to(
                        product_root
                    )
                ),
                title="Computed Forest Mask",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["quicklook_fd"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_fd_ql.png").relative_to(product_root)
                ),
                title="QuickLook Forest Disturbance",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
            assets_dict["quicklook_probability"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_probability_ql.png").relative_to(
                        product_root
                    )
                ),
                title="QuickLook Probability of Change",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
            assets_dict["quicklook_cfm"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_cfm_ql.png").relative_to(
                        product_root
                    )
                ),
                title="QuickLook Computed Forest Mask",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FH:
            assets_dict["fh"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_fh.tiff").relative_to(
                        product_root
                    )
                ),
                title="Forest Height",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["quality"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_quality.tiff").relative_to(
                        product_root
                    )
                ),
                title="Forest Height Quality",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["quicklook_fh"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_fh_ql.png").relative_to(product_root)
                ),
                title="QuickLook Forest Height",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
            assets_dict["quicklook_quality"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_fhquality_ql.png").relative_to(
                        product_root
                    )
                ),
                title="QuickLook Forest Height Quality",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            assets_dict["gn"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_gn.tiff").relative_to(
                        product_root
                    )
                ),
                title="Ground Cancelled",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["quicklook_gn"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_gn_ql.png").relative_to(product_root)
                ),
                title="QuickLook Ground Cancelled",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_TFH:
            assets_dict["fh"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_fh.tiff").relative_to(
                        product_root
                    )
                ),
                title="Forest Height",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["quality"] = pystac.asset.Asset(
                href="./"
                + str(
                    measurement_file_root.joinpath(self.product.name.lower()[:-10] + "_i_quality.tiff").relative_to(
                        product_root
                    )
                ),
                title="Forest Height Quality",
                description=None,
                media_type=None,
                roles=["measurement", "data"],
                extra_fields={"type": "tiff"},
            )
            assets_dict["quicklook_fh"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_fh_ql.png").relative_to(product_root)
                ),
                title="QuickLook Forest Height",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )
            assets_dict["quicklook_quality"] = pystac.asset.Asset(
                href="./"
                + str(
                    preview_file_root.joinpath(self.product.name.lower()[:-10] + "_fhquality_ql.png").relative_to(
                        product_root
                    )
                ),
                title="QuickLook Forest Height Quality",
                description=None,
                media_type=None,
                roles=["preview", "image"],
                extra_fields={"type": "png"},
            )

        assets_dict["mph"] = pystac.asset.Asset(
            href="./" + str(Path(self.product_structure.mph_file).relative_to(product_root)),
            title="Main Product Header",
            description=None,
            media_type=None,
            roles=["metadata"],
            extra_fields={"type": "xml"},
        )
        assets_dict["main_ads"] = pystac.asset.Asset(
            href="./" + str(Path(self.product_structure.main_annotation_file).relative_to(product_root)),
            title="Main Annotation Data Set",
            description=None,
            media_type=None,
            roles=["annotation", "metadata"],
            extra_fields={"type": "xml"},
        )
        assets_dict["lut_ads"] = pystac.asset.Asset(
            href="./" + str(Path(self.product_structure.lut_annotation_file).relative_to(product_root)),
            title="Look-Up Tables Annotation Data Set",
            description=None,
            media_type=None,
            roles=["annotation", "lut"],
            extra_fields={"type": "netCDF"},
        )

        # create the STAC item, with pystac library
        item = pystac.Item(
            id,
            geometry,
            bbox,
            collection=None,
            datetime=None,
            properties=properties,
        )

        # Add assets to STAC item
        for key, asset in assets_dict.items():
            item.add_asset(
                key,
                asset,
            )

        # save to disc the single item STAC object
        item.save_object(dest_href=self.product_structure.stac_file)

    def _write_measurement_files_core(
        self,
        data_list,
        cog_metadata,
        file_path,
        compression_options,
        nodata_value,
        mds_block_size,
    ):
        """Write one of the MDS measurement folder tiff file.
           This function is called one time for each MDS to be written.

        Parameters
        ----------
        data_list : list containing MDS data matrices
            it contains, depending on the product type
                single element list in case of FD or FH products (one of the MDS for each core writer call)
                three element list in case of GN (three polarizations)
                GN three polarizations matrices
        file_path : Path
            Path where the measirement file file will be written
        compression_options : Dict
            Dictionary containing two keys:
                compressionFactor: int
                MAX_Z_ERROR: float (this should be zero in caseo of binary MDS as FD or CFM)
            _description_
        nodata_value : float, int
            Pixel value in case of invalid data: flot for float MDS, int for binary MDS

        Returns
        -------
        _type_
            _description_
        """

        if len(cog_metadata.compression) == 2:
            compression_name = COMPRESSION_SCHEMA_MDS_LERC_ZSTD
        elif cog_metadata.compression[0] == COMPRESSION_EXIF_CODES_LERC_ZSTD[1]:
            compression_name = COMPRESSION_SCHEMA_MDS_ZSTD
        else:
            raise ValueError(f"Compression schema not recognized: {cog_metadata.compression}")

        max_z_error = compression_options.max_z_error if hasattr(compression_options, "max_z_error") else 0

        geotiff_conf = GeotiffConf(
            compression_schema=compression_name,
            max_z_error=max_z_error,
            zstd_level=compression_options.compression_factor,
            nodata_value=nodata_value,
            block_size=mds_block_size,
            overview_levels=[2, 4],
            epsg_code=4326,
        )

        matrix_representation = "SYMMETRIZED_SCATTERING" if len(data_list) > 1 else None
        polarizations = ["HH", "VH", "VV"] if len(data_list) > 1 else None
        additional_metadata = {
            "tileID": cog_metadata.tile_id_list,
            "basinID": cog_metadata.basin_id_list,
        }

        geotiff_metadata = GeotiffMetadata(
            creation_date=cog_metadata.dateTime,
            swath=cog_metadata.swath,
            software=f"BPS-{self.processor_name}-{self.processor_version}",
            matrix_representation=matrix_representation,
            description=cog_metadata.image_description,
            polarizations=polarizations,
            additional_metadata=additional_metadata,
        )

        # add geotransform [longitude start, longitude step, 0, latitude start, 0, latitude step]
        geotransform = [
            self.product.measurement.longitude_vec[0],
            self.product.measurement.longitude_vec[1] - self.product.measurement.longitude_vec[0],
            0,
            self.product.measurement.latitude_vec[0],
            0,
            self.product.measurement.latitude_vec[1] - self.product.measurement.latitude_vec[0],
        ]

        write_geotiff(
            file_path,
            data_list,
            nodata_mask=None,
            ecef_gcp_list=None,
            geotiff_metadata=geotiff_metadata,
            geotiff_conf=geotiff_conf,
            geotransform=geotransform,
        )

    def _write_measurement_files(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # Create measurement file
        assert self.product_structure.measurement_files is not None

        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            for key in self.product.measurement.data_dict.keys():
                if key not in "gn":
                    raise ValueError("Not valid dictionary key '{}' in processed data dictionary.".format(key))

            # write a single COG file containing three layers (three polarizations)
            file_path = self.product_structure.measurement_files[0]

            bps_logger.info(
                f"    Output product GN, COG compression factor: {self.product.main_ads_processing_parameters.compression_options.mds.gn.compression_factor}"
            )
            bps_logger.info(
                f"    Output product GN, COG MAX Z ERROR: {self.product.main_ads_processing_parameters.compression_options.mds.gn.max_z_error}"
            )

            self._write_measurement_files_core(
                self.product.measurement.data_dict["gn"],
                self.product.measurement.metadata_dict["gn"],  # all metadata are equal
                file_path,
                self.product.main_ads_processing_parameters.compression_options.mds.gn,
                FLOAT_NODATA_VALUE,
                self.product.main_ads_processing_parameters.compression_options.mds_block_size,
            )

        else:
            for key in self.product.measurement.data_dict.keys():
                if key not in [
                    "fd",
                    "cfm",
                    "probability_ofchange",
                    "fh",
                    "quality",
                ]:
                    raise ValueError("Not valid dictionary key '{}' in processed data dictionary.".format(key))

                data_list = self.product.measurement.data_dict[key]

                # dont trust dict and list ordering, search correct data for current key:
                if key == "fd":
                    path_string_to_search = "_i_fd.tiff"
                elif key == "cfm":
                    path_string_to_search = "_i_cfm.tiff"
                elif key == "probability_ofchange":
                    path_string_to_search = "_i_probability.tiff"
                elif key == "fh":
                    path_string_to_search = "_i_fh.tiff"
                elif key == "quality":
                    path_string_to_search = "_i_quality.tiff"
                else:
                    raise ValueError(f"Key {key} not recognized in product measurement data dictionary")
                cog_metadata = self.product.measurement.metadata_dict[key]

                for file_path in self.product_structure.measurement_files:
                    if path_string_to_search in file_path:
                        break

                no_data_value = (
                    FLOAT_NODATA_VALUE
                    if key in ["probability_ofchange", "fh", "quality"]
                    else INT_NODATA_VALUE  # ["fd", "cfm"]
                )

                key_r = str(key).replace("_", " ")

                if hasattr(
                    self.product.main_ads_processing_parameters.compression_options.mds,
                    "tfh",
                ):
                    comp_opt = getattr(
                        self.product.main_ads_processing_parameters.compression_options.mds,
                        "tfh",
                    )
                else:
                    comp_opt = getattr(
                        self.product.main_ads_processing_parameters.compression_options.mds,
                        key,
                    )

                string_to_add = ""
                if (key == "fd" and np.all(self.product.measurement.data_dict[key] == INT_NODATA_VALUE)) or (
                    key == "probability_ofchange"
                    and np.all(self.product.measurement.data_dict[key] == FLOAT_NODATA_VALUE)
                ):
                    string_to_add = " (all no data values)"
                bps_logger.info(
                    f"    Output product {key_r}, COG compression factor: {comp_opt.compression_factor}{string_to_add}"
                )
                if hasattr(comp_opt, "max_z_error"):
                    bps_logger.info(
                        f"    Output product {key_r}, COG MAX Z ERROR: {comp_opt.max_z_error}{string_to_add}"
                    )

                self._write_measurement_files_core(
                    data_list,  # first input should be a dictionary
                    cog_metadata,
                    file_path,
                    comp_opt,
                    no_data_value,
                    self.product.main_ads_processing_parameters.compression_options.mds_block_size,
                )

    def __write_vrt_file(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # Get configuration parameters

        # Fill vrt model
        srs_list = [
            vrt.Srstype(
                value=""" GEOGCRS["WGS 84",
                          DATUM["World Geodetic System 1984",
                              ELLIPSOID["WGS 84",6378137,298.257223563,
                                  LENGTHUNIT["metre",1]]],
                          PRIMEM["Greenwich",0,
                              ANGLEUNIT["degree",0.0174532925199433]],
                          CS[ellipsoidal,2],
                              AXIS["geodetic latitude (Lat)",north,
                                  ORDER[1],
                                  ANGLEUNIT["degree",0.0174532925199433]],
                              AXIS["geodetic longitude (Lon)",east,
                                  ORDER[2],
                                  ANGLEUNIT["degree",0.0174532925199433]],
                          ID["EPSG",4326]] """,
                data_axis_to_srsaxis_mapping="1,2",
            )
        ]

        vrtraster_band_list = []
        if L2A_OUTPUT_PRODUCT_GN in self.product_structure.schema_files[0]:
            raster_xsize = self.product.measurement.data_dict["gn"][0].shape[1]
            raster_ysize = self.product.measurement.data_dict["gn"][0].shape[0]
            number_of_bands = 3
            descriptions = ["H/H", "V/H", "V/V"]

            data_type = vrt.DataTypeType.FLOAT32.value
            no_data_value = FLOAT_NODATA_VALUE

            pixel_function_type_list = ["real"]
            pixel_function_arguments_list = []

            for band in range(number_of_bands):
                description = descriptions[band]

                sub_class = vrt.VrtrasterBandSubTypeType.VRTDERIVED_RASTER_BAND.value

                description_list = [description]
                source_transfer_type_list = [vrt.DataTypeType.FLOAT32.value]
                no_data_value_element_list = [no_data_value]
                color_interp_list = [vrt.ColorInterpType.GRAY.value]

                simple_source_list = []
                file_name = Path(self.product_structure.measurement_files[0])
                if file_name.exists():
                    source_filename = [vrt.SourceFilenameType(value=file_name.name, relative_to_vrt=1)]
                    source_band = [str(band + 1)]
                    source_properties = [vrt.SourcePropertiesType(raster_xsize=raster_xsize, raster_ysize=raster_ysize)]
                    simple_source = vrt.SimpleSourceType(
                        source_filename=source_filename,
                        source_band=source_band,
                        source_properties=source_properties,
                    )
                    simple_source_list.append(simple_source)

                vrtraster_band = vrt.VrtrasterBandType(
                    description=description_list,
                    pixel_function_type=pixel_function_type_list,
                    pixel_function_arguments=pixel_function_arguments_list,
                    source_transfer_type=source_transfer_type_list,
                    no_data_value_element=no_data_value_element_list,
                    color_interp=color_interp_list,
                    simple_source=simple_source_list,
                    data_type=data_type,
                    band=band,
                    sub_class=sub_class,
                )
                vrtraster_band_list.append(vrtraster_band)

        else:
            if L2A_OUTPUT_PRODUCT_FD in self.product_structure.schema_files[0]:
                raster_xsize = self.product.measurement.data_dict["fd"][0].shape[1]
                raster_ysize = self.product.measurement.data_dict["fd"][0].shape[0]
            if L2A_OUTPUT_PRODUCT_FH in self.product_structure.schema_files[0]:
                raster_xsize = self.product.measurement.data_dict["fh"][0].shape[1]
                raster_ysize = self.product.measurement.data_dict["fh"][0].shape[0]

            for band, key in enumerate(self.product.measurement.data_dict.keys()):
                data_type = vrt.DataTypeType.FLOAT32.value
                no_data_value = FLOAT_NODATA_VALUE
                if key in ["fd", "cfm"]:
                    data_type = vrt.DataTypeType.BYTE.value
                    no_data_value = INT_NODATA_VALUE
                if key in ["probability_ofchange"]:
                    key = "probability"

                description = key

                pixel_function_type_list = ["real"]
                pixel_function_arguments_list = []

                sub_class = vrt.VrtrasterBandSubTypeType.VRTDERIVED_RASTER_BAND.value

                description_list = [description]
                source_transfer_type_list = [vrt.DataTypeType.FLOAT32.value]
                no_data_value_element_list = [no_data_value]
                color_interp_list = [vrt.ColorInterpType.GRAY.value]

                simple_source_list = []
                for name in self.product_structure.measurement_files:
                    if key + ".tiff" in name:
                        file_name = Path(name)
                        break
                if file_name.exists():
                    source_filename = [vrt.SourceFilenameType(value=file_name.name, relative_to_vrt=1)]
                    source_band = "1"
                    source_properties = [vrt.SourcePropertiesType(raster_xsize=raster_xsize, raster_ysize=raster_ysize)]
                    simple_source = vrt.SimpleSourceType(
                        source_filename=source_filename,
                        source_band=source_band,
                        source_properties=source_properties,
                    )
                    simple_source_list.append(simple_source)

                vrtraster_band = vrt.VrtrasterBandType(
                    description=description_list,
                    pixel_function_type=pixel_function_type_list,
                    pixel_function_arguments=pixel_function_arguments_list,
                    source_transfer_type=source_transfer_type_list,
                    no_data_value_element=no_data_value_element_list,
                    color_interp=color_interp_list,
                    simple_source=simple_source_list,
                    data_type=data_type,
                    band=band,
                    sub_class=sub_class,
                )
                vrtraster_band_list.append(vrtraster_band)

        vrt_model = vrt.Vrtdataset(
            srs=srs_list,
            gcplist=[],
            vrtraster_band=vrtraster_band_list,
            raster_xsize=raster_xsize,
            raster_ysize=raster_ysize,
        )

        # Write vrt file
        vrt_text = serialize(vrt_model)
        vrt_path = Path(self.product_structure.vrt_file)
        vrt_path.write_text(vrt_text, encoding="utf-8")

    def _write_main_annotation_file(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # Fill main annotation model

        # - product
        mission = common_annotation_models_l2.MissionType(self.product.main_ads_product.mission)
        tile_id_list = common_annotation_models_l2.StringListType(self.product.main_ads_product.tile_id_list)
        basin_id_list = common_annotation_models_l2.StringListType(self.product.main_ads_product.basin_id_list)

        assert self.product_structure.schema_files is not None
        if L2A_OUTPUT_PRODUCT_FD in self.product_structure.schema_files[0]:
            product_type = common_annotation_models_l2.ProductType.FD_L2_A
        if L2A_OUTPUT_PRODUCT_FH in self.product_structure.schema_files[0]:
            product_type = common_annotation_models_l2.ProductType.FH_L2_A
        if L2A_OUTPUT_PRODUCT_GN in self.product_structure.schema_files[0]:
            product_type = common_annotation_models_l2.ProductType.GN_L2_A
        if L2A_OUTPUT_PRODUCT_TFH in self.product_structure.schema_files[0]:
            product_type = common_annotation_models_l2.ProductType.TFH_L2_A

        start_time = self.product.main_ads_product.start_time.isoformat(timespec="microseconds")[:-1]
        stop_time = self.product.main_ads_product.stop_time.isoformat(timespec="microseconds")[:-1]
        radar_carrier_frequency = common_annotation_models_l2.DoubleWithUnit(
            value=self.product.main_ads_product.radar_carrier_frequency,
            units=common_types.UomType.HZ,
        )
        mission_phase_id = common_annotation_models_l2.MissionPhaseIdtype(
            self.product.main_ads_product.mission_phase_id
        )
        sensor_mode = common_annotation_models_l2.SensorModeType(self.product.main_ads_product.sensor_mode)
        global_coverage_id = translate_global_coverage_id(self.product.main_ads_product.global_coverage_id)
        swath = common_annotation_models_l2.SwathType(self.product.main_ads_product.swath)
        major_cycle_id = translate_major_cycle_id(self.product.main_ads_product.major_cycle_id)
        absolute_orbit_number = common_annotation_models_l2.IntegerListType(
            [
                translate_com_phase_negative_values(absolute_orbit_number)
                for absolute_orbit_number in self.product.main_ads_product.absolute_orbit_number
            ]
        )
        relative_orbit_number = translate_com_phase_negative_values(self.product.main_ads_product.relative_orbit_number)
        orbit_pass = common_annotation_models_l2.OrbitPassType(self.product.main_ads_product.orbit_pass)
        datatake_id = common_annotation_models_l2.IntegerListType(self.product.main_ads_product.datatake_id)
        frame = translate_frame_id(self.product.main_ads_product.frame)
        platform_heading = common_annotation_models_l2.DoubleWithUnit(
            value=self.product.main_ads_product.platform_heading,
            units=common_types.UomType.DEG,
        )
        forest_coverage_percentage = self.product.main_ads_product.forest_coverage_percentage

        if self.product.main_ads_product.selected_reference_image is not None:
            selected_reference_image = common_annotation_models_l2.SelectedReferenceImageType(
                value=self.product.main_ads_product.selected_reference_image,
                acquisition_id=self.product.main_ads_product.acquisition_id_reference_image,
            )
        else:
            selected_reference_image = None

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            product = main_annotation_models_l2a_fd.ProductL2AType(
                mission,
                tile_id_list,
                basin_id_list,
                product_type,
                start_time,
                stop_time,
                radar_carrier_frequency,
                mission_phase_id,
                sensor_mode,
                global_coverage_id,
                swath,
                major_cycle_id,
                absolute_orbit_number,
                relative_orbit_number,
                orbit_pass,
                datatake_id,
                frame,
                platform_heading,
                forest_coverage_percentage,
                selected_reference_image,
            )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FH:
            product = main_annotation_models_l2a_fh.ProductL2AType(
                mission,
                tile_id_list,
                basin_id_list,
                product_type,
                start_time,
                stop_time,
                radar_carrier_frequency,
                mission_phase_id,
                sensor_mode,
                global_coverage_id,
                swath,
                major_cycle_id,
                absolute_orbit_number,
                relative_orbit_number,
                orbit_pass,
                datatake_id,
                frame,
                platform_heading,
                forest_coverage_percentage,
                selected_reference_image,
            )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            product = main_annotation_models_l2a_gn.ProductL2AType(
                mission,
                tile_id_list,
                basin_id_list,
                product_type,
                start_time,
                stop_time,
                radar_carrier_frequency,
                mission_phase_id,
                sensor_mode,
                global_coverage_id,
                swath,
                major_cycle_id,
                absolute_orbit_number,
                relative_orbit_number,
                orbit_pass,
                datatake_id,
                frame,
                platform_heading,
                forest_coverage_percentage,
                selected_reference_image,
            )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_TFH:
            product = main_annotation_models_l2a_tfh.ProductL2AType(
                mission,
                tile_id_list,
                basin_id_list,
                product_type,
                start_time,
                stop_time,
                radar_carrier_frequency,
                mission_phase_id,
                sensor_mode,
                global_coverage_id,
                swath,
                major_cycle_id,
                absolute_orbit_number,
                relative_orbit_number,
                orbit_pass,
                datatake_id,
                frame,
                platform_heading,
                forest_coverage_percentage,
                selected_reference_image,
            )

        footprint = ""
        for num in self.product.main_ads_input_information.footprint:
            footprint = footprint + str(num) + " "
        footprint = footprint[0:-1]
        footprint = common_annotation_models_l2.FloatArrayWithUnits(
            value=footprint,
            count=len(self.product.main_ads_input_information.footprint),
            units=common_types.UomType.DEG,
        )

        first_latitude_value = common_annotation_models_l2.FloatWithUnit(
            value=float(self.product.main_ads_raster_image.first_latitude_value),
            units=common_types.UomType.DEG,
        )
        first_longitude_value = common_annotation_models_l2.FloatWithUnit(
            value=float(self.product.main_ads_raster_image.first_longitude_value),
            units=common_types.UomType.DEG,
        )
        latitude_spacing = common_annotation_models_l2.FloatWithUnit(
            value=float(self.product.main_ads_raster_image.latitude_spacing),
            units=common_types.UomType.DEG,
        )
        longitude_spacing = common_annotation_models_l2.FloatWithUnit(
            value=float(self.product.main_ads_raster_image.longitude_spacing),
            units=common_types.UomType.DEG,
        )
        number_of_samples = self.product.main_ads_raster_image.number_of_samples
        number_of_lines = self.product.main_ads_raster_image.number_of_lines
        projection = common_annotation_models_l2.ProjectionType(self.product.main_ads_raster_image.projection)
        datum = self.product.main_ads_raster_image.datum
        pixel_representation = self.product.main_ads_raster_image.pixel_representation
        pixel_type = self.product.main_ads_raster_image.pixel_type
        no_data_value = self.product.main_ads_raster_image.no_data_value
        raster_image = common_annotation_models_l2.RasterImageType(
            footprint,
            first_latitude_value,
            first_longitude_value,
            latitude_spacing,
            longitude_spacing,
            number_of_samples,
            number_of_lines,
            projection,
            datum,
            pixel_representation,
            pixel_type,
            no_data_value,
        )

        # # - inputInformation
        polarisation_list = self.product.main_ads_input_information.polarisation_list

        list_acq = []
        for acquisition_type in self.product.main_ads_input_information.acquisition_list.acquisition:
            list_acq.append(
                common_annotation_models_l2.AcquisitionType(
                    acquisition_type.folder_name,
                    acquisition_type.sta_quality,
                    acquisition_type.reference_image,
                    average_wavenumber=(
                        float(acquisition_type.average_wavenumber)
                        if acquisition_type.average_wavenumber is not None
                        else None
                    ),
                )
            )
        acquisition_list = common_annotation_models_l2.AcquisitionListType(
            acquisition=list_acq,
            count=len(list_acq),
        )

        input_information = common_annotation_models_l2.InputInformationL2AType(
            common_types.ProductType.STA,
            self.product.main_ads_input_information.overall_products_quality_index,
            self.product.main_ads_input_information.nominal_stack,
            polarisation_list,
            common_types.ProjectionType.SLANT_RANGE,
            footprint,
            self.product.main_ads_input_information.vertical_wavenumbers,
            self.product.main_ads_input_information.height_of_ambiguity,
            acquisition_list,  # acquisition list
        )

        # # - processing parameters
        subsetting_rule = self.product.main_ads_processing_parameters.general_configuration.subsetting_rule
        general_configuration_parameters = common_annotation_models_l2.GeneralConfigurationParametersType(
            self.product.main_ads_processing_parameters.general_configuration.apply_calibration_screen,
            self.product.main_ads_processing_parameters.general_configuration.forest_coverage_threshold,
            self.product.main_ads_processing_parameters.general_configuration.forest_mask_interpolation_threshold,
            subsetting_rule,
        )

        processor_version = self.product.main_ads_processing_parameters.processor_version
        PRODUCT_GENERATION_TIME_MS = self.product.main_ads_processing_parameters.product_generation_time.isoformat(
            timespec="microseconds"
        )[:-1]

        compression_options = self.product.main_ads_processing_parameters.compression_options

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            emphasized_forest_height = common_annotation_models_l2.FloatWithUnit(
                value=self.product.main_ads_processing_parameters.emphasized_forest_height,
                units="m",
            )
            operational_mode = main_annotation_models_l2a_fd.OperationalModeType(
                self.product.main_ads_processing_parameters.operational_mode
            )
            disable_ground_cancellation_flag = str(
                self.product.main_ads_processing_parameters.disable_ground_cancellation_flag
            ).lower()
            significance_level = float(self.product.main_ads_processing_parameters.significance_level)

            product_resolution = common_annotation_models_l2.FloatWithUnit(
                value=float(self.product.main_ads_processing_parameters.product_resolution),
                units="m",
            )
            numerical_determinant_limit = float(self.product.main_ads_processing_parameters.numerical_determinant_limit)

            upsampling_factor = self.product.main_ads_processing_parameters.upsampling_factor

            processing_parameters = main_annotation_models_l2a_fd.ProcessingParametersL2AType(
                processor_version,
                PRODUCT_GENERATION_TIME_MS,
                general_configuration_parameters,
                emphasized_forest_height,
                operational_mode,
                (
                    self.product.main_ads_processing_parameters.images_pair_selection
                    if operational_mode == common_annotation_models_l2.OperationalModeType.INSAR_PAIR
                    else None
                ),
                disable_ground_cancellation_flag,
                significance_level,
                product_resolution,
                numerical_determinant_limit,
                upsampling_factor,
                compression_options,
            )

            # - annotationLUT
            layer = [
                main_annotation_models_l2a_fd.LayerType.FNF,
                main_annotation_models_l2a_fd.LayerType.ACM,
                main_annotation_models_l2a_fd.LayerType.NUMBER_OF_AVERAGES,
            ]
            annotation_lut = common_annotation_models_l2.LayerListType(layer, count=len(layer))

            main_annotation_model = main_annotation_models_l2a_fd.MainAnnotation(
                product,
                raster_image,
                input_information,
                processing_parameters,
                annotation_lut,
            )

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FH:
            vertical_reflectivity_option = main_annotation_models_l2a_fh.VerticalProfileOptionType(
                self.product.main_ads_processing_parameters.vertical_reflectivity_option
            )
            vertical_reflectivity_default_profile = main_annotation_models_l2a_fh.VerticalReflectivityProfileType(
                [
                    float(value)
                    for value in self.product.main_ads_processing_parameters.vertical_reflectivity_default_profile
                ],
                count=len(self.product.main_ads_processing_parameters.vertical_reflectivity_default_profile),
            )
            model_inversion = main_annotation_models_l2a_fh.ModelInversionType(
                self.product.main_ads_processing_parameters.model_inversion
            )
            spectral_decorrelation_compensation_flag = str(
                self.product.main_ads_processing_parameters.spectral_decorrelation_compensation_flag
            ).lower()
            snr_decorrelation_compensation_flag = str(
                self.product.main_ads_processing_parameters.snr_decorrelation_compensation_flag
            ).lower()
            correct_terrain_slopes_flag = str(
                self.product.main_ads_processing_parameters.correct_terrain_slopes_flag
            ).lower()
            normalised_height_estimation_range = (
                self.product.main_ads_processing_parameters.normalised_height_estimation_range
            )
            normalised_wavenumber_estimation_range = (
                self.product.main_ads_processing_parameters.normalised_wavenumber_estimation_range
            )
            ground_to_volume_ratio_range = self.product.main_ads_processing_parameters.ground_to_volume_ratio_range
            temporal_decorrelation_estimation_range = (
                self.product.main_ads_processing_parameters.temporal_decorrelation_estimation_range
            )
            temporal_decorrelation_ground_to_volume_ratio = (
                self.product.main_ads_processing_parameters.temporal_decorrelation_ground_to_volume_ratio
            )
            residual_decorrelation = self.product.main_ads_processing_parameters.residual_decorrelation

            product_resolution = common_annotation_models_l2.FloatWithUnit(
                value=self.product.main_ads_processing_parameters.product_resolution,
                units="m",
            )
            uncertainty_valid_values_limits = (
                self.product.main_ads_processing_parameters.uncertainty_valid_values_limits
            )

            vertical_wavenumber_valid_values_limits = (
                self.product.main_ads_processing_parameters.vertical_wavenumber_valid_values_limits
            )

            lower_height_limit = common_annotation_models_l2.FloatWithUnit(
                value=self.product.main_ads_processing_parameters.lower_height_limit,
                units=common_types.UomType.M,
            )
            upsampling_factor = self.product.main_ads_processing_parameters.upsampling_factor

            processing_parameters = main_annotation_models_l2a_fh.ProcessingParametersL2AType(
                processor_version,
                PRODUCT_GENERATION_TIME_MS,
                general_configuration_parameters,
                vertical_reflectivity_option,
                vertical_reflectivity_default_profile,
                model_inversion,
                spectral_decorrelation_compensation_flag,
                snr_decorrelation_compensation_flag,
                correct_terrain_slopes_flag,
                normalised_height_estimation_range,
                normalised_wavenumber_estimation_range,
                ground_to_volume_ratio_range,
                temporal_decorrelation_estimation_range,
                temporal_decorrelation_ground_to_volume_ratio,
                residual_decorrelation,
                product_resolution,
                uncertainty_valid_values_limits,
                vertical_wavenumber_valid_values_limits,
                lower_height_limit,
                upsampling_factor,
                compression_options,
            )

            # - annotationLUT
            layer = [
                main_annotation_models_l2a_fh.LayerType.FNF,
            ]
            annotation_lut = common_annotation_models_l2.LayerListType(layer, count=len(layer))

            main_annotation_model = main_annotation_models_l2a_fh.MainAnnotation(
                product,
                raster_image,
                input_information,
                processing_parameters,
                annotation_lut,
            )

        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            emphasized_forest_height = common_annotation_models_l2.FloatWithUnit(
                value=self.product.main_ads_processing_parameters.emphasized_forest_height,
                units="m",
            )
            operational_mode = main_annotation_models_l2a_gn.OperationalModeType(
                self.product.main_ads_processing_parameters.operational_mode
            )
            compute_gn_power_flag = str(self.product.main_ads_processing_parameters.compute_gn_power_flag).lower()
            radiometric_calibration_flag = str(
                self.product.main_ads_processing_parameters.radiometric_calibration_flag
            ).lower()
            disable_ground_cancellation_flag = str(
                self.product.main_ads_processing_parameters.disable_ground_cancellation_flag
            ).lower()
            product_resolution = common_annotation_models_l2.FloatWithUnit(
                value=self.product.main_ads_processing_parameters.product_resolution,
                units="m",
            )
            upsampling_factor = self.product.main_ads_processing_parameters.upsampling_factor

            processing_parameters = main_annotation_models_l2a_gn.ProcessingParametersL2AType(
                processor_version,
                PRODUCT_GENERATION_TIME_MS,
                general_configuration_parameters,
                emphasized_forest_height,
                operational_mode,
                compute_gn_power_flag,
                radiometric_calibration_flag,
                (
                    self.product.main_ads_processing_parameters.images_pair_selection
                    if operational_mode == common_annotation_models_l2.OperationalModeType.INSAR_PAIR
                    else None
                ),
                disable_ground_cancellation_flag,
                product_resolution,
                upsampling_factor,
                compression_options,
            )

            # - annotationLUT
            layer = [
                main_annotation_models_l2a_gn.LayerType.FNF,
                main_annotation_models_l2a_gn.LayerType.INCIDENCE_ANGLE_DEG,
            ]
            annotation_lut = common_annotation_models_l2.LayerListType(layer, count=int(2))

            main_annotation_model = main_annotation_models_l2a_gn.MainAnnotation(
                product,
                raster_image,
                input_information,
                processing_parameters,
                annotation_lut,
            )

        if self.product.product_type == L2A_OUTPUT_PRODUCT_TFH:
            enable_super_resolution = str(self.product.main_ads_processing_parameters.enable_super_resolution).lower()

            product_resolution = common_annotation_models_l2.FloatWithUnit(
                value=self.product.main_ads_processing_parameters.product_resolution,
                units="m",
            )

            processing_parameters = main_annotation_models_l2a_tfh.ProcessingParametersL2AType(
                processor_version,
                PRODUCT_GENERATION_TIME_MS,
                general_configuration_parameters,
                enable_super_resolution,
                product_resolution,
                self.product.main_ads_processing_parameters.regularization_noise_factor,
                self.product.main_ads_processing_parameters.power_threshold,
                self.product.main_ads_processing_parameters.median_factor,
                self.product.main_ads_processing_parameters.estimation_valid_values_limits,
                compression_options,
            )

            # - annotationLUT
            layer = [
                main_annotation_models_l2a_fh.LayerType.FNF,
            ]
            annotation_lut = common_annotation_models_l2.LayerListType(layer, count=len(layer))

            main_annotation_model = main_annotation_models_l2a_tfh.MainAnnotation(
                product,
                raster_image,
                input_information,
                processing_parameters,
                annotation_lut,
            )

        # Write main annotation file
        assert self.product_structure.main_annotation_file is not None

        main_annotation_text = serialize(main_annotation_model)
        main_annotation_path = Path(self.product_structure.main_annotation_file)
        main_annotation_path.write_text(main_annotation_text, encoding="utf-8")

    def _write_lut_annotation_file(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        # Opening a file, creating a new Dataset
        ncfile = None
        # mode='r' is the default.
        # mode='a' opens an existing file and allows for appending (does not clobber existing data)
        # format can be one of NETCDF3_CLASSIC, NETCDF3_64BIT, NETCDF4_CLASSIC or NETCDF4 (default). NETCDF4_CLASSIC uses HDF5 for the underlying storage layer (as does NETCDF4) but enforces the classic netCDF 3 data model so data can be read with older clients.
        ncfile = Dataset(
            self.product_structure.lut_annotation_file,
            mode="w",
            format="NETCDF4",
            clobber=True,
        )

        # ATTRIBUTES: global, common for all the LUT elements
        ncfile.mission = common_types.MissionType.BIOMASS.value
        ncfile.tileID = json.dumps(self.product.main_ads_product.tile_id_list)
        ncfile.basinID = json.dumps(self.product.main_ads_product.basin_id_list)
        ncfile.productType = self.product.product_type

        ncfile.startTime = self.product.main_ads_product.start_time.isoformat(timespec="milliseconds")
        ncfile.stopTime = self.product.main_ads_product.stop_time.isoformat(timespec="milliseconds")
        ncfile.radarCarrierFrequency = np.float32(self.product.main_ads_product.radar_carrier_frequency)
        ncfile.missionPhaseID = self.product.main_ads_product.mission_phase_id
        ncfile.sensorMode = common_types.SensorModeType.MEASUREMENT.value
        ncfile.globalCoverageID = np.uint16(
            translate_global_coverage_id(self.product.main_ads_product.global_coverage_id)
        )
        ncfile.swath = self.product.main_ads_product.swath
        ncfile.majorCycleID = np.uint16(translate_major_cycle_id(self.product.main_ads_product.major_cycle_id))

        ncfile.absoluteOrbitNumber = [
            np.uint32(translate_com_phase_negative_values(absolute_orbit_number))
            for absolute_orbit_number in self.product.main_ads_product.absolute_orbit_number
        ]
        ncfile.relativeOrbitNumber = np.uint16(
            translate_com_phase_negative_values(self.product.main_ads_product.relative_orbit_number)
        )
        ncfile.orbitPass = self.product.main_ads_product.orbit_pass
        ncfile.dataTakeID = json.dumps([str(dt_id) for dt_id in self.product.main_ads_product.datatake_id])
        ncfile.frame = np.uint16(translate_frame_id(self.product.main_ads_product.frame))
        ncfile.platform_heading = np.float32(self.product.main_ads_product.platform_heading)
        ncfile.forest_coverage_percentage = np.float32(self.product.main_ads_product.forest_coverage_percentage)
        if self.product.main_ads_product.selected_reference_image is not None:
            ncfile.selected_reference_image = np.uint16(self.product.main_ads_product.selected_reference_image)
            ncfile.acquisition_id_reference_image = self.product.main_ads_product.acquisition_id_reference_image

        # DIMENSIONS:
        ncfile.createDimension("Latitude", len(self.product.measurement.latitude_vec))
        ncfile.createDimension("Longitude", len(self.product.measurement.longitude_vec))
        ncfile.createDimension("scalar", 1)
        ncfile.createDimension("string", 1)

        # 3) VARIABLES
        # 3/1: Coordinate Variables
        lat_var = ncfile.createVariable("Latitude", np.float32, ("Latitude",))
        lat_var.units = "deg"
        lat_var.description = "LUT data latitude axis"
        lon_var = ncfile.createVariable("Longitude", np.float32, ("Longitude",))
        lon_var.units = "deg"
        lon_var.description = "LUT data longitude axis"

        # 3/2: Variables:
        # le inserisco in gruppi per scelta (non obbligatorio)
        # oltre ai dati, qui ci vanno anche tutte le cose che nei nostri documenti abbiamo messo in DIMENSIONS

        fnf_group = ncfile.createGroup("FNF")
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            acm_group = ncfile.createGroup("ACM")
            acm_group.description = """ACM represents the covariance history of all previous cycles (see also numberOfAverages variable): it is obtained by averaging the covariance computed with FD algorithm at the given Global cycle with the previous ACM in input to the processor. At first Global cycle it is the current covariance. It consists of six images of which three real and three complex, stored in nine NetCDF layers as three layers for the three real images and three complex images, each split in the amplitude value layer and the phase value (radiants) layer."""
            noa_group = ncfile.createGroup("numberOfAverages")
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            inc_group = ncfile.createGroup("localIncidenceAngle")
        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            groups_list = [fnf_group, acm_group, noa_group]
        if self.product.product_type in [
            L2A_OUTPUT_PRODUCT_FH,
            L2A_OUTPUT_PRODUCT_TFH,
        ]:
            groups_list = [fnf_group]
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            groups_list = [fnf_group, inc_group]

        compression_options = self.product.main_ads_processing_parameters.compression_options

        fnf_mask_var = fnf_group.createVariable(
            "FNF",
            np.uint8,
            ("Latitude", "Longitude"),
            chunksizes=(
                compression_options.ads_block_size,
                compression_options.ads_block_size,
            ),  # chunk sizes
            compression=COMPRESSION_SCHEMA_ADS,
            complevel=compression_options.ads.fnf.compression_factor,
            fill_value=INT_NODATA_VALUE,
        )
        fnf_mask_var.description = "Forest/Non-Forest mask: this is an external image, defined outside of BPS from aggregated C3S LCM, representing the initial two-dimensional mask, here defined in the same latitude-longitude based on DGG projection used for L2a images and cropped to have a coverage containing the L2a product image boundaries. It defines, for each geographical pixel, the initial presence or absence of forest."
        bps_logger.info(
            f"    Output product LUT ADS FNF compression factor: {compression_options.ads.fnf.compression_factor}"
        )

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            noa_var = noa_group.createVariable(
                "numberOfAverages",
                np.uint8,
                ("Latitude", "Longitude"),
                chunksizes=(
                    compression_options.ads_block_size,
                    compression_options.ads_block_size,
                ),  # chunk sizes
                compression=COMPRESSION_SCHEMA_ADS,
                complevel=compression_options.ads.number_of_averages.compression_factor,
                fill_value=INT_NODATA_VALUE,
            )
            noa_var.description = "It represents, for the ACM, the number of covariances averaged so far at given Global cycle, pixel wise (see also ACM variable). At first Global cycle it is zero for each pixel."
            bps_logger.info(
                f"    Output product LUT ADS Number of Averages compression factor: {compression_options.ads.number_of_averages.compression_factor}"
            )
            acm_layers_list = []

            # Those are the 9 matrix_out elements (for 6 images)
            # lut layer   >> [ layer1,  layer2,  layer3,   layer4,  layer5,   layer6,  layer7,  layer8,   layer9]
            # image index >> [ 1        abs(2)   phase(2)  abs(3)   phase(3)  4        abs(5)   phase(5)  6     ]
            # image type  >> [ HH-HH,   HH-XP,   HH-XP,    HH-VV,   HH-VV,    XP-XP,   XP-VV,   XP-VV,    VV-VV ]

            images_indices_for_description = [1, 2, 2, 3, 3, 4, 5, 5, 6]
            images_description = [
                "real, HH-HH",  # layer1, image 1
                "amplitude value of complex image, HH-XP",  # layer2, image 2
                "phase value of complex image [rad], HH-XP",  # layer3, image 2
                "amplitude value of complex image, HH-VV",  # layer4, image 3
                "phase value of complex image [rad], HH-VV",  # layer5, image 3
                "real, XP-XP",  # layer6, image 4
                "amplitude value of complex image, XP-VV",  # layer7, image 5
                "phase value of complex image [rad], XP-VV",  # layer8, image 5
                "real, VV-VV",  # layer9, image 6
            ]
            for list_idx, image_idx_first, image_description in zip(
                np.arange(9), images_indices_for_description, images_description
            ):
                layer_idx = list_idx + 1
                acm_layers_list.append(
                    acm_group.createVariable(
                        "layer{}".format(layer_idx),
                        np.float32,
                        ("Latitude", "Longitude"),
                        chunksizes=(
                            compression_options.ads_block_size,
                            compression_options.ads_block_size,
                        ),  # chunk sizes
                        compression=COMPRESSION_SCHEMA_ADS,
                        complevel=compression_options.ads.acm.compression_factor,
                        least_significant_digit=(
                            compression_options.ads.acm.least_significant_digit
                            if compression_options.ads.acm.least_significant_digit
                            else None
                        ),
                        fill_value=FLOAT_NODATA_VALUE,
                    )
                )
                acm_layers_list[list_idx].description = "layer {}: ACM image {} of 6, {}".format(
                    layer_idx, image_idx_first, image_description
                )
                if "phase" in image_description:
                    acm_layers_list[list_idx].units = "rad"

            bps_logger.info(
                f"    Output product LUT ADS ACM compression factor: {compression_options.ads.acm.compression_factor}"
            )
            if compression_options.ads.acm.least_significant_digit:
                bps_logger.info(
                    f"    Output product LUT ADS ACM  least significant digit: {compression_options.ads.acm.least_significant_digit}"
                )
        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            incidence_var = inc_group.createVariable(
                "localIncidenceAngle",
                np.float32,
                ("Latitude", "Longitude"),
                chunksizes=(
                    compression_options.ads_block_size,
                    compression_options.ads_block_size,
                ),  # chunk sizes
                compression=COMPRESSION_SCHEMA_ADS,
                complevel=compression_options.ads.local_incidence_angle.compression_factor,
                least_significant_digit=(
                    compression_options.ads.local_incidence_angle.least_significant_digit
                    if compression_options.ads.local_incidence_angle.least_significant_digit
                    else None
                ),
                fill_value=FLOAT_NODATA_VALUE,
            )
            incidence_var.description = """Local incidence angle image in radiants, which takes into account terrain local slope and defined in the latitude-longitude based on DGG projection, cropped to have a  coverage containing the L2a Ground cancelled backscatter image product boundaries."""
            incidence_var.units = "rad"
            bps_logger.info(
                f"    Output product LUT ADS Incidence Angle compression factor: {compression_options.ads.local_incidence_angle.compression_factor}"
            )
            if compression_options.ads.local_incidence_angle.least_significant_digit:
                bps_logger.info(
                    f"    Output product LUT ADS Incidence Angle least significant digit: {compression_options.ads.local_incidence_angle.least_significant_digit}"
                )

        for group in groups_list:
            if group.name == "FNF":
                md = self.product.lut_ads.lut_fnf_metadata
            if group.name == "numberOfAverages":
                md = self.product.lut_ads.lut_number_of_averages_metadata
            if group.name == "ACM":
                md = self.product.lut_ads.lut_acm_metadata
            if group.name == "localIncidenceAngle":
                md = self.product.lut_ads.lut_local_incidence_angle_metadata

            firstSample = group.createVariable("firstSample", np.uint32, dimensions=("scalar"))
            firstSample.description = "Product image MDS sample the first LUT sample corresponds to"
            firstSample[:] = np.uint32(md.first_sample)

            firstLine = group.createVariable("firstLine", np.uint32, dimensions=("scalar"))
            firstLine.description = "Product image MDS line the first LUT line corresponds to"
            firstLine[:] = np.uint32(md.first_line)

            samplesInterval = group.createVariable("samplesInterval", np.uint32, dimensions=("scalar"))
            samplesInterval.description = "Number of product image MDS samples one LUT sample corresponds to"
            samplesInterval[:] = np.uint32(md.samples_interval)

            linesInterval = group.createVariable("linesInterval", np.uint32, dimensions=("scalar"))
            linesInterval.description = "Number of product image MDS lines one LUT line corresponds to"
            linesInterval[:] = np.uint32(md.lines_interval)

            pixelType = group.createVariable("pixelType", str, dimensions=("string"))
            pixelType.description = "Data type of output LUT pixels"
            pixelType[:] = np.array([md.pixelType], dtype="object")  # This is the way to store strings in NetCdf

            if md.no_data_value == INT_NODATA_VALUE:
                noDataValue = group.createVariable("noDataValue", np.uint8, dimensions=("scalar"))
            else:
                noDataValue = group.createVariable("noDataValue", np.float32, dimensions=("scalar"))
            noDataValue.description = "Pixel value in case of LUT invalid data"
            noDataValue[:] = md.no_data_value

            projection = group.createVariable("projection", str, dimensions=("string"))
            projection.description = "LUT projection, is the latitude-longitude based on DGG"
            projection[:] = np.array([md.projection], dtype="object")

            coordinateReferenceSystem = group.createVariable("coordinateReferenceSystem", str, dimensions=("string"))
            coordinateReferenceSystem.description = "Coordinate reference systems, as per WKT representation"
            coordinateReferenceSystem[:] = np.array([md.coordinateReferenceSystem], dtype="object")

            geodeticReferenceFrame = group.createVariable("geodeticReferenceFrame", str, dimensions=("string"))
            geodeticReferenceFrame.description = "Geodetic reference frame"
            geodeticReferenceFrame[:] = np.array([md.geodeticReferenceFrame], dtype="object")

            if group.name == "ACM":
                if type(md.least_significant_digit) is str:
                    least_significant_digit = group.createVariable(
                        "least_significant_digit", str, dimensions=("string")
                    )
                    least_significant_digit[:] = np.array([md.least_significant_digit], dtype="object")
                else:
                    least_significant_digit = group.createVariable(
                        "least_significant_digit", np.uint8, dimensions=("scalar")
                    )
                    least_significant_digit[:] = np.uint8(md.least_significant_digit)
                least_significant_digit.description = "Power of ten of the smallest decimal place in the data that is a reliable value, adopted to the data during ZLIB compression. Can be disabled with special value [string] “none”."

        # WRITE DATA ON VARIABLES

        # Write the Coordinate Variables
        lat_var[:] = self.product.measurement.latitude_vec
        lon_var[:] = self.product.measurement.longitude_vec
        # Write the main matrices variables

        # Write FNF
        fnf_mask_var[:, :] = self.product.lut_ads.lut_fnf

        if self.product.product_type == L2A_OUTPUT_PRODUCT_FD:
            # Write Number of Averages
            noa_var[:, :] = self.product.lut_ads.lut_number_of_averages

            # Write each layer of ACM
            for idx, acm_layer in enumerate(acm_layers_list):
                acm_layer[:, :] = self.product.lut_ads.lut_acm[idx, :, :]

        if self.product.product_type == L2A_OUTPUT_PRODUCT_GN:
            # Write Local Incidence Angle
            incidence_var[:, :] = self.product.lut_ads.lut_local_incidence_angle

        # Close the Dataset and examine the contents with ncdump.
        ncfile.close()

    def _write_quicklook_files(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        dirname = Path(self.product_structure.quicklook_files[0]).parent
        if not dirname.exists():
            dirname.mkdir(parents=True, exist_ok=True)

        if "gn" in self.product.measurement.data_dict.keys():
            # For GN, genwerate an RGB image using the three polarizations

            # Assign input data (absolute values in linear scale) to RGB channels
            (data_lines, data_samples) = self.product.measurement.data_dict["gn"][0].shape
            rgb = np.zeros((data_lines, data_samples, 3), "float")

            mask = self.product.measurement.data_dict["gn"][0] == FLOAT_NODATA_VALUE

            rgb[:, :, 0] = np.abs(
                np.abs(self.product.measurement.data_dict["gn"][0])
                - np.abs(self.product.measurement.data_dict["gn"][2])
            )  # Red channel: ||HH|-|VV|| (double bounce)
            rgb[:, :, 1] = np.abs(
                self.product.measurement.data_dict["gn"][1]
            )  # Green channel: |VH| (volume scattering)
            rgb[:, :, 2] = np.abs(
                np.abs(self.product.measurement.data_dict["gn"][0])
                + np.abs(self.product.measurement.data_dict["gn"][2])
            )  # Blue channel: |HH|+|VV| (surface scattering)

            # set no data values to zeros, for the quick look
            rgb[:, :, 0][mask] = 0.0
            rgb[:, :, 1][mask] = 0.0
            rgb[:, :, 2][mask] = 0.0

            # If required, multilook input data
            if AVERAGING_FACTOR_QUICKLOOKS > 1:
                rgb_decimated = np.zeros(
                    (
                        int(np.ceil(data_lines / DECIMATION_FACTOR_QUICKLOOKS)),
                        int(np.ceil(data_samples / DECIMATION_FACTOR_QUICKLOOKS)),
                        3,
                    ),
                    "float",
                )
                boxcar = np.ones((AVERAGING_FACTOR_QUICKLOOKS, AVERAGING_FACTOR_QUICKLOOKS))
                for channel in range(3):
                    rgb_channel = np.sqrt(
                        convolve2d(rgb[:, :, channel] ** 2, boxcar, "same")
                        / (AVERAGING_FACTOR_QUICKLOOKS * AVERAGING_FACTOR_QUICKLOOKS)
                    )
                    rgb_decimated[:, :, channel] = rgb_channel[
                        ::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS
                    ]
                rgb = rgb_decimated

            # Scale RGB channels to be in the [0:255] range
            for channel in range(3):
                rgb_channel = rgb[:, :, channel]
                max_value = np.percentile(rgb_channel, 99)
                rgb_channel[rgb_channel > max_value] = max_value
                rgb_channel = rgb_channel / max_value * 255
                rgb[:, :, channel] = rgb_channel

            rgb = rgb.astype("uint8")

            # make transparent (alpha = 0) out of the footprint
            alpha = np.where(self.footprint_mask_for_quicklooks, 255, 0).astype(np.uint8)

            # Insert Alpha channel to the image RGB

            imm_bgra = np.dstack([cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR), alpha])

            cv2.imwrite(self.product_structure.quicklook_files[0], imm_bgra)

        else:
            # For all but not GN, generate a simple image (not RGB)

            for key in self.product.measurement.data_dict.keys():
                (data_lines, data_samples) = self.product.measurement.data_dict[key][0].shape

                imm_to_save = self.product.measurement.data_dict[key][0]

                # set no data values to zeros, for the quick look
                if key in ["fd", "cfm"]:
                    imm_to_save[imm_to_save == INT_NODATA_VALUE] = 0
                else:
                    imm_to_save[imm_to_save == FLOAT_NODATA_VALUE] = 0.0
                if key == "probability_ofchange":
                    name_to_search = "_probability_ql"
                else:
                    name_to_search = "_" + key + "_ql"

                # If required, multilook input data
                if AVERAGING_FACTOR_QUICKLOOKS > 1:
                    boxcar = np.ones((AVERAGING_FACTOR_QUICKLOOKS, AVERAGING_FACTOR_QUICKLOOKS))

                    imm_to_save = np.sqrt(
                        convolve2d(imm_to_save**2, boxcar, "same")
                        / (AVERAGING_FACTOR_QUICKLOOKS * AVERAGING_FACTOR_QUICKLOOKS)
                    )[::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS]

                if key in ["fd", "cfm"]:
                    imm_to_save = imm_to_save.astype(np.uint8)
                else:
                    # Scale float values to be integers in the [0:255] range
                    imm_to_save = (255.0 * imm_to_save / np.max(imm_to_save)).astype(np.uint8)

                fname = None
                for fname in self.product_structure.quicklook_files:
                    if name_to_search in fname:
                        break
                assert fname is not None

                # Write quicklook file
                # Convert from 2D GRAY image a 4D BGRA (replica of image in three RGB channels)
                imm_bgra = cv2.cvtColor(imm_to_save, cv2.COLOR_GRAY2BGRA)

                # make transparent (alpha = 0) out of the footprint
                alpha = np.where(self.footprint_mask_for_quicklooks, 255, 0).astype(np.uint8)

                # Insert Alpha channel to the image BRGA
                imm_bgra[:, :, 3] = alpha

                cv2.imwrite(fname, imm_bgra)

    def _write_overlay_files(self):
        # The quick look footprint is larger than the product footprint (geocoded and placed in the latitude-longitude grid, with NaNs where needed)
        lon_min = min(self.product.main_ads_raster_image.footprint[0::2])
        lon_max = max(self.product.main_ads_raster_image.footprint[0::2])
        lat_min = min(self.product.main_ads_raster_image.footprint[1::2])
        lat_max = max(self.product.main_ads_raster_image.footprint[1::2])

        # Clock wise quick look footprint, starting from SW (is the pixel 0,0)
        footprint = [
            (lon_min, lat_min),  # SW → bottom-left PNG
            (lon_min, lat_max),  # NW → top-left PNG
            (lon_max, lat_max),  # NE → top-right PNG
            (lon_max, lat_min),  # SE → bottom-right PNG
        ]

        for overlay_file, quicklook_file in zip(
            self.product_structure.overlay_files, self.product_structure.quicklook_files
        ):
            write_overlay_file(
                Path(overlay_file),
                Path(quicklook_file),
                self.product.name,
                footprint,
                "L2a Product Overlay ADS",
            )

    def _write_schema_files(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        assert self.product_structure.schema_files is not None
        names = [Path(s).name for s in self.product_structure.schema_files]
        copy_biomass_xsd_files(Path(self.product_structure.schema_files[0]).parent, names)

    def write(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        bps_logger.info("Writing BIOMASS L2a product")

        # Initialize product structure on disk
        self._init_product_structure()

        # Write STAC file
        bps_logger.debug("..STAC file")
        self._write_stac_file()

        # Write measurement files
        bps_logger.debug("..measurement files")
        self._write_measurement_files()
        # self.__write_vrt_file()

        # Write annotation files
        bps_logger.debug("..main annotation file")
        self._write_main_annotation_file()
        bps_logger.debug("..LUT annotation file")
        self._write_lut_annotation_file()

        # Write quick-look file
        bps_logger.debug("..quick-look files")
        self._write_quicklook_files()

        # Write overlay files.
        bps_logger.debug("Writing overlay files")
        self._write_overlay_files()

        # Write schema files
        bps_logger.debug("..schema files")
        self._write_schema_files()

        # Write MPH file
        bps_logger.debug("..MPH file")
        self._write_mph_file()

        bps_logger.debug("..done")
