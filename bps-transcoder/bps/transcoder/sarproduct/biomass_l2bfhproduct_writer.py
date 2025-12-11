# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
import pystac
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common_types
from bps.common.io.parsing import serialize
from bps.transcoder import BPS_PFD_FH_VERSION, BPS_PPD_VERSION
from bps.transcoder.io import (
    common_annotation_models_l2,
    main_annotation_models_l2b_fh,
    vrt,
)
from bps.transcoder.sarproduct.biomass_l2bfhproduct import (
    BIOMASSL2bFHProduct,
    BIOMASSL2bFHProductStructure,
)
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES
from bps.transcoder.sarproduct.overlay import write_overlay_file
from bps.transcoder.utils.gdal_utils import GeotiffConf, GeotiffMetadata, write_geotiff
from bps.transcoder.utils.production_model_utils import (
    encode_mph_id_value,
    translate_global_coverage_id,
)
from bps.transcoder.utils.xsd_schema_attacher import copy_biomass_xsd_files
from scipy.signal import convolve2d

for key, value in MPH_NAMESPACES.items():
    ET.register_namespace(key, value)

COMPRESSION_SCHEMA_MDS_LERC_ZSTD = "LERC_ZSTD"
COMPRESSION_SCHEMA_MDS_ZSTD = "ZSTD"
COMPRESSION_EXIF_CODES_LERC_ZSTD = [34887, 34926]  #  LERC, ZSTD
FLOAT_NODATA_VALUE = float(-9999.0)
INT_NODATA_VALUE = int(255)
L2B_OUTPUT_PRODUCT_FH = "FP_FH__L2B"
DECIMATION_FACTOR_QUICKLOOKS = 2
AVERAGING_FACTOR_QUICKLOOKS = 2


class BIOMASSL2bFHProductWriter:
    """_summary_"""

    def __init__(
        self,
        product: BIOMASSL2bFHProduct,  # product to write
        product_path: Path,  # from job order
        processor_name: str,  # from job order
        processor_version: str,  # from job order
        footprint: list[float],  # to be computed by the processor for each L2b tile
        input_l2a_products_names: list[str],
        gcp_list: list,
        aux_pp2_fd_name: str,
        start_time_l2a: PreciseDateTime,
        stop_time_l2a: PreciseDateTime,
        footprint_mask_for_quicklooks: np.ndarray,
        input_l2b_fd_product_name: str | None = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        product : BIOMASSL2bFHProduct
            _description_
        product_path : _type_
            _description_
        processor_name : str
            _description_
        processor_version : str
            _description_
        input_l2a_products_names: List[str]
            _description_
        aux_pp2_fd_name: str
            _description_
        input_l2b_fd_product_name: Optional[str] = None
            _description_
        start_time_l2a: PreciseDateTime
            _description_
        stop_time_l2a: PreciseDateTime
            _description_
        footprint_mask_for_quicklooks
            _description_
        """
        self.product = product
        self.product_path = product_path
        self.processor_name = processor_name
        self.processor_version = processor_version
        self.footprint = footprint
        self.input_l2a_products_names = input_l2a_products_names
        self.gcp_list = gcp_list
        self.aux_pp2_fd_name = aux_pp2_fd_name
        self.input_l2b_fd_product_name = input_l2b_fd_product_name
        self.start_time_l2a = start_time_l2a
        self.stop_time_l2a = stop_time_l2a
        self.footprint_mask_for_quicklooks = footprint_mask_for_quicklooks

        # output path full and check
        self.product_path = self.product_path.joinpath(self.product.name)
        self._check_product_path()

        # folder structure of the product
        self.product_structure = BIOMASSL2bFHProductStructure(self.product_path)

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

        footprint_closed = self.footprint + self.footprint[0:2]
        footprint_str = ""
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
        xml4.text = self.start_time_l2a.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.stop_time_l2a.isoformat(timespec="milliseconds")
        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}resultTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimeInstant",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}timePosition")
        xml4.text = self.stop_time_l2a.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}validTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimePeriod",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}beginPosition")
        xml4.text = self.start_time_l2a.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.stop_time_l2a.isoformat(timespec="milliseconds")
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
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}acquisitionParameters")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["bio"] + "}Acquisition")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}missionPhase")
        mission_phases_dict = {
            "INT": "INTERFEROMETRIC",
            "TOM": "TOMOGRAPHIC",
            "COM": "COMMISSIONING",
        }
        xml6.text = mission_phases_dict[str(self.product.main_ads_product.mission_phase_id)]
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}globalCoverageID")
        xml6.text = encode_mph_id_value(self.product.main_ads_product.global_coverage_id)
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
                    self.footprint[0],
                    self.footprint[3],
                ]
            ),
            np.mean(
                [
                    self.footprint[1],
                    self.footprint[2],
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

        files = (
            [
                (self.product_structure.fh_file, None),
                (
                    self.product_structure.fh_quality_file,
                    None,
                ),
                (
                    self.product_structure.bps_fnf_file,
                    None,
                ),
                (
                    self.product_structure.heatmap_file,
                    None,
                ),
                (
                    self.product_structure.acquisition_id_image_file,
                    None,
                ),
                (
                    self.product_structure.main_annotation_file,
                    self.product_structure.l2b_fh_main_ann_xsd,
                ),
            ]
            + [(self.product_structure.stac_file, None)]
            + [(file, None) for file in self.product_structure.quicklook_files]
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
        xml4.text = L2B_OUTPUT_PRODUCT_FH
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
        xml6.text = "other: L2b"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
        xml6.text = self.aux_pp2_fd_name
        if self.input_l2b_fd_product_name:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
            xml6.text = self.input_l2b_fd_product_name
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}processingMode",
            attrib={"codeSpace": "urn:esa:eop:Biomass:P-SAR:processingMode"},
        )
        xml6.text = "OPERATIONAL"
        for input_l2a_product_name in self.input_l2a_products_names:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}sourceProduct")
            xml6.text = input_l2a_product_name

        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
        xml4.text = f"BIOMASS Forest Height Product Format Specification (BPS_FH_PFD) {BPS_PFD_FH_VERSION}"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
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
        self.product.main_ads_input_information.input_information.l2a_inputs

        # times
        start_time = self.product.main_ads_product.start_time.isoformat(timespec="milliseconds")
        stop_time = self.product.main_ads_product.stop_time.isoformat(timespec="milliseconds")
        product_generation_time = self.product.main_ads_processing_parameters.product_generation_time.isoformat(
            timespec="milliseconds"
        )

        properties = {
            "platform": "BIOMASS",
            "instrument": "p-sar",
            "processing:software": "bps",
            "processing:software_version": self.product.main_ads_processing_parameters.processor_version,
            "processing:level": "L2B",
            "processing:product_type": "Forest Height",
            "tile_id": self.product.main_ads_product.tile_id_list[0],
            "start_datetime": start_time,
            "end_datetime": stop_time,
            "global_cycle": id[18:20],
            "created": product_generation_time,
            "proj:shape": list(self.product.measurement.data_dict["fh"].shape),
        }

        product_root = Path(self.product_structure.product_path)
        preview_file_root = product_root.joinpath(self.product_structure.preview_subfolder)

        # assets
        assets_dict = {}
        preview_file_root = str(
            self.product_path.joinpath(
                self.product_structure.preview_subfolder,
                self.product.name.lower()[:-10],
            )
        )
        assets_dict["fh"] = pystac.asset.Asset(
            href="./" + str(Path(self.product_structure.fh_file).relative_to(product_root)),
            title="Forest Height",
            description=None,
            media_type=None,
            roles=["measurement", "data"],
            extra_fields={"type": "tiff"},
        )
        assets_dict["quality"] = pystac.asset.Asset(
            href="./" + str(Path(self.product_structure.fh_quality_file).relative_to(product_root)),
            title="Forest Height Quality",
            description=None,
            media_type=None,
            roles=["measurement", "data"],
            extra_fields={"type": "tiff"},
        )
        assets_dict["heatmap"] = pystac.asset.Asset(
            href="./" + str(Path(self.product_structure.heatmap_file).relative_to(product_root)),
            title="Heat Map",
            description=None,
            media_type=None,
            roles=["annotation", "data"],
            extra_fields={"type": "tiff"},
        )
        assets_dict["quicklook_fh"] = pystac.asset.Asset(
            href="./" + str(Path(preview_file_root + "_fh_ql.png").relative_to(product_root)),
            title="QuickLook Forest Height",
            description=None,
            media_type=None,
            roles=["preview", "image"],
            extra_fields={"type": "png"},
        )
        assets_dict["quicklook_quality"] = pystac.asset.Asset(
            href="./"
            + str(
                Path(preview_file_root + self.product.name.lower()[:-10] + "_fh_quality_ql.png").relative_to(
                    product_root
                )
            ),
            title="Forest Height Quality",
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

        # create the STAC item, with pystac library
        item = pystac.Item(
            id,
            geometry,
            bbox,
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
        data,
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
        data : MDS data matrices
        file_path : Path
            Path where the measirement file file will be written
        compression_options : Dict
            Dictionary containing two keys:
                compressionFactor: int
                MAX_Z_ERROR: float (this should be zero in caseo of binary MDS as FH or FNF)
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

        additional_metadata = {
            "tileID": cog_metadata.tile_id_list,
            "basinID": cog_metadata.basin_id_list,
        }

        geotiff_metadata = GeotiffMetadata(
            creation_date=cog_metadata.dateTime,
            swath=None,
            software=f"BPS-{self.processor_name}-{self.processor_version}",
            matrix_representation=None,
            description=cog_metadata.image_description,
            polarizations=None,
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

        if len(data.shape) == 3 and data.shape[2] > 1:
            # heat map, acquisition image id...
            band_number = data.shape[2]
        elif len(data.shape) == 3:
            # heat map, acquisition image id, special case when only one l2a input is present
            band_number = data.shape[2]
            data = np.squeeze(data)
        else:
            band_number = 1

        # Write bands
        if band_number == 1:
            data_to_write = [data]
        else:
            # its the heat_map
            data_to_write = []
            for band_idx in range(band_number):
                data_to_write.append(data[:, :, band_idx])

        write_geotiff(
            file_path,
            data_to_write,
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
        assert self.product_structure.fh_file is not None
        assert self.product_structure.fh_quality_file is not None
        assert self.product_structure.bps_fnf_file is not None
        assert self.product_structure.heatmap_file is not None

        for key in self.product.measurement.data_dict.keys():
            if key in ["dgg_latitude_axis", "dgg_longitude_axis"]:
                continue

            if key not in [
                "fh",
                "quality",
                "bps_fnf",
                "heat_map",
                "acquisition_id_image",
            ]:
                raise ValueError("Not valid dictionary key '{}' in processed data dictionary.".format(key))

            data = self.product.measurement.data_dict[key]

            # dont trust dict and list ordering, search correct data for current key:
            key_r = str(key).replace("_", " ")
            cog_metadata = self.product.measurement.metadata_dict[key]
            no_data_value = FLOAT_NODATA_VALUE

            if key == "fh":
                file_path = self.product_structure.fh_file
            elif key == "quality":
                file_path = self.product_structure.fh_quality_file
            elif key == "bps_fnf":
                file_path = self.product_structure.bps_fnf_file
                no_data_value = INT_NODATA_VALUE
            elif key == "heat_map":
                file_path = self.product_structure.heatmap_file
            elif key == "acquisition_id_image":
                file_path = self.product_structure.acquisition_id_image_file
            else:
                raise ValueError(f"Key {key} not recognized in product measurement data dictionary")

            comp_opt = getattr(
                self.product.main_ads_processing_parameters.compression_options.mds,
                key,
            )
            bps_logger.info(f"    Output product {key_r}, COG compression factor: {comp_opt.compression_factor}")
            if hasattr(comp_opt, "max_z_error"):
                bps_logger.info(f"    Output product {key_r}, COG MAX Z ERROR: {comp_opt.max_z_error}")

            self._write_measurement_files_core(
                data,
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
        raster_xsize = self.product.measurement.data_dict["fh"].shape[1]
        raster_ysize = self.product.measurement.data_dict["fh"].shape[0]

        for band, (file_name, description) in enumerate(
            zip(
                [
                    self.product_structure.fh_file,
                    self.product_structure.fh_quality_file,
                ],
                ["fh", "quality"],
            )
        ):
            file_name = Path(file_name)
            data_type = vrt.DataTypeType.FLOAT32.value
            no_data_value = FLOAT_NODATA_VALUE

            pixel_function_type_list = ["real"]
            pixel_function_arguments_list = []

            sub_class = vrt.VrtrasterBandSubTypeType.VRTDERIVED_RASTER_BAND.value

            description_list = [description]
            source_transfer_type_list = [vrt.DataTypeType.FLOAT32.value]
            no_data_value_element_list = [no_data_value]
            color_interp_list = [vrt.ColorInterpType.GRAY.value]

            simple_source_list = []

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
        tile_id = common_annotation_models_l2.StringListType(self.product.main_ads_product.tile_id_list)
        basin_id_list = common_annotation_models_l2.StringListType(self.product.main_ads_product.basin_id_list)
        product_type = common_annotation_models_l2.ProductType.FH_L2_B

        start_time = self.start_time_l2a.isoformat(timespec="microseconds")[:-1]

        stop_time = self.stop_time_l2a.isoformat(timespec="microseconds")[:-1]

        radar_carrier_frequency = common_annotation_models_l2.DoubleWithUnit(
            value=self.product.main_ads_product.radar_carrier_frequency,
            units=common_types.UomType.HZ,
        )
        mission_phase_id = common_annotation_models_l2.MissionPhaseIdtype(
            self.product.main_ads_product.mission_phase_id
        )
        sensor_mode = common_annotation_models_l2.SensorModeType(self.product.main_ads_product.sensor_mode)
        global_coverage_id = translate_global_coverage_id(self.product.main_ads_product.global_coverage_id)
        product = main_annotation_models_l2b_fh.ProductL2BL3Type(
            mission,
            tile_id.id[0],
            basin_id_list,
            product_type,
            start_time,
            stop_time,
            radar_carrier_frequency,
            mission_phase_id,
            sensor_mode,
            global_coverage_id,
        )

        dgg_tile_footprint = ""
        for num in self.footprint:
            dgg_tile_footprint = dgg_tile_footprint + str(num) + " "
        dgg_tile_footprint = dgg_tile_footprint[0:-1]
        dgg_tile_footprint = common_annotation_models_l2.FloatArrayWithUnits(
            value=dgg_tile_footprint,
            count=len(self.footprint),
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
            dgg_tile_footprint,
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
        input_information_l2b = common_annotation_models_l2.InputInformationL2BL3ListType(
            self.product.main_ads_input_information.input_information.l2a_inputs,
            count=len(self.product.main_ads_input_information.input_information.l2a_inputs),
        )
        # # - processing parameters
        processor_version = self.product.main_ads_processing_parameters.processor_version
        PRODUCT_GENERATION_TIME_MS = self.product.main_ads_processing_parameters.product_generation_time.isoformat(
            timespec="microseconds"
        )[:-1]

        compression_options_fh = self.product.main_ads_processing_parameters.compression_options

        minimum_l2acoverage = self.product.main_ads_processing_parameters.minumum_l2a_coverage
        assert minimum_l2acoverage is not None

        forest_masking_flag = str(self.product.main_ads_processing_parameters.forest_masking_flag).lower()

        bps_fnf_type = self.product.main_ads_processing_parameters.bps_fnf

        roll_off_factor_azimuth = self.product.main_ads_processing_parameters.roll_off_factor_azimuth
        roll_off_factor_range = self.product.main_ads_processing_parameters.roll_off_factor_range

        processing_parameters = main_annotation_models_l2b_fh.ProcessingParametersL2BType(
            processor_version,
            PRODUCT_GENERATION_TIME_MS,
            minimum_l2acoverage,
            forest_masking_flag,
            bps_fnf_type,
            roll_off_factor_azimuth,
            roll_off_factor_range,
            compression_options_fh,
        )

        main_annotation_model = main_annotation_models_l2b_fh.MainAnnotation(
            product,
            raster_image,
            input_information_l2b,
            processing_parameters,
        )

        # Write main annotation file
        assert self.product_structure.main_annotation_file is not None

        main_annotation_text = serialize(main_annotation_model)
        main_annotation_path = Path(self.product_structure.main_annotation_file)
        main_annotation_path.write_text(main_annotation_text, encoding="utf-8")

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

        for key in self.product.measurement.data_dict.keys():
            if key not in [
                "fh",
                "quality",
                "bps_fnf",
                "heat_map",
            ]:
                continue

            imm_to_save = self.product.measurement.data_dict[key]
            name_to_search = "_" + key + "_ql"

            if key == "bps_fnf":
                imm_to_save[imm_to_save == INT_NODATA_VALUE] = 0
                imm_to_save = np.sum(imm_to_save, axis=2)
                imm_to_save[imm_to_save > 1] = 1
            else:
                imm_to_save[imm_to_save == FLOAT_NODATA_VALUE] = 0.0
                if key == "quality":
                    name_to_search = "_fhquality_ql"

            # If required, multilook input data
            if AVERAGING_FACTOR_QUICKLOOKS > 1:
                boxcar = np.ones((AVERAGING_FACTOR_QUICKLOOKS, AVERAGING_FACTOR_QUICKLOOKS))

                imm_to_save = np.sqrt(
                    convolve2d(imm_to_save**2, boxcar, "same")
                    / (AVERAGING_FACTOR_QUICKLOOKS * AVERAGING_FACTOR_QUICKLOOKS)
                )[::DECIMATION_FACTOR_QUICKLOOKS, ::DECIMATION_FACTOR_QUICKLOOKS]

            fname = None
            for fname in self.product_structure.quicklook_files:
                if name_to_search in fname:
                    break
            assert fname is not None

            # Write quicklook file
            # Scale float values to be integers in the [0:255] range
            imm_to_save = (255.0 * imm_to_save / np.max(imm_to_save)).astype(np.uint8)

            # Convert from 2D GRAY image a 4D BGRA (replica of image in three RGB channels)
            imm_bgra = cv2.cvtColor(imm_to_save, cv2.COLOR_GRAY2BGRA)

            # make transparent (alpha = 0) out of the footprint
            alpha = np.where(self.footprint_mask_for_quicklooks, 255, 0).astype(np.uint8)

            # Insert Alpha channel to the image BRGA
            imm_bgra[:, :, 3] = alpha

            cv2.imwrite(fname, imm_bgra)

    def _write_overlay_files(self):
        # NE, SE, SW, NW
        # The quick look footprint is larger than the product footprint (geocoded and placed in the latitude-longitude grid, with NaNs where needed)
        # For L2B, main_ads_raster_image already contains the DGG 1x1 deg tile footprint
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
                "L2b FH Product Overlay ADS",
                self.product.main_ads_product.start_time.isoformat().replace("-", "").replace(":", "")[0:15],
                self.product.main_ads_product.stop_time.isoformat().replace("-", "").replace(":", "")[0:15],
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
        bps_logger.info("Writing BIOMASS L2b FH product")

        # Initialize product structure on disk
        self._init_product_structure()

        # Write STAC file (TBD)
        bps_logger.debug("..STAC file")
        self._write_stac_file()

        # Write measurement files
        bps_logger.debug("..measurement files")
        self._write_measurement_files()
        # self.__write_vrt_file()

        # Write annotation files
        bps_logger.debug("..main annotation file")
        self._write_main_annotation_file()

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
