# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

import copy
import itertools
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
from arepytools.constants import LIGHT_SPEED
from arepytools.math import genericpoly
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io import common, translate_common
from bps.common.io.parsing import serialize
from bps.transcoder import BPS_PFD_L1_VERSION, BPS_PPD_VERSION
from bps.transcoder.io import (
    common_annotation_l1,
    main_annotation_l1ab,
    translate_main_annotation_l1ab,
)
from bps.transcoder.io.preprocessor_report import BAQCompressionLevel
from bps.transcoder.sarproduct.biomass_l1product import BIOMASSL1Product
from bps.transcoder.sarproduct.generic_product import GenericProduct
from bps.transcoder.sarproduct.l1.product_content import L1ProductContent
from bps.transcoder.sarproduct.l1.quality_index import L1QualityIndex
from bps.transcoder.sarproduct.l1_annotations import (
    CombinedDCStatistics,
    build_list_of_combined_dc_stats_from_annotations_and_poly,
    fill_coordinate_conversion_type,
    fill_empty_coordinate_conversion_type,
    fill_tx_pulse,
    get_coefficients_from_poly,
    get_list_of_dc_poly,
    translate_rfi_freq_stats_to_model,
    translate_rfi_time_stats,
)
from bps.transcoder.sarproduct.l1_lut_writer import ProductLUTID, write_lut_file
from bps.transcoder.sarproduct.l1_pre_proc_report_to_annotations import InternalCalibrationParameters, RawDataAnalysis
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES
from bps.transcoder.sarproduct.navigation_files_utils import replace_template_string
from bps.transcoder.sarproduct.nodata_mask import create_no_data_mask
from bps.transcoder.sarproduct.overlay import write_overlay_file
from bps.transcoder.sarproduct.vrt import VRTInfo, write_vrt_file
from bps.transcoder.utils.constants import AVERAGE_GROUND_VELOCITY
from bps.transcoder.utils.gdal_utils import GeotiffConf, GeotiffMetadata, write_geotiff
from bps.transcoder.utils.production_model_utils import encode_mph_id_value
from bps.transcoder.utils.rxgain_table import get_rx_gain_from_parameters_code
from bps.transcoder.utils.xsd_schema_attacher import copy_biomass_xsd_files

CURRENT_AZIMUTH_TIME = PreciseDateTime.now()
TAI_UTC = 37

for key, value in MPH_NAMESPACES.items():
    ET.register_namespace(key, value)


def convert_baq_compression_level(
    level: BAQCompressionLevel,
) -> common.DataFormatModeType:
    if level == BAQCompressionLevel.BAQ_4_BIT:
        return common.DataFormatModeType.BAQ_4_BIT
    elif level == BAQCompressionLevel.BAQ_5_BIT:
        return common.DataFormatModeType.BAQ_5_BIT
    elif level == BAQCompressionLevel.BAQ_6_BIT:
        return common.DataFormatModeType.BAQ_6_BIT
    elif level == BAQCompressionLevel.BYPASS:
        return common.DataFormatModeType.BYPASS
    else:
        raise ValueError("Invalid BAQCompressionLevel")


class BIOMASSL1ProductWriter:
    """_summary_"""

    def __init__(
        self,
        product: BIOMASSL1Product,
        product_path,
        product_lut: dict[ProductLUTID, GenericProduct | None] | None = None,
        processor_name: str = "L1_P",
        processor_version: str = "1.0.0",
        gdal_num_threads: int = 1,
    ) -> None:
        """_summary_

        Parameters
        ----------
        product : BIOMASSL1Product
            _description_
        product_path : _type_
            _description_
        product_lut : dict
            _description_
        processor_name : str
            _description_
        processor_version : str
            _description_
        """
        self.product = product
        self.product_path = Path(product_path).joinpath(self.product.name)
        if os.path.exists(self.product_path):
            raise FileExistsError(f"Folder {self.product_path} already exists.")

        self.product_lut = product_lut if product_lut else {}
        self.processor_name = processor_name
        self.processor_version = processor_version

        data_shape = min(d.shape for d in self.product.data_list)
        self.product.data_list = [d[: data_shape[0], : data_shape[1]] for d in self.product.data_list]

        self.content = L1ProductContent.from_name(self.product_path.name)
        self.gdal_num_threads = gdal_num_threads

    def _write_mph_file(self):
        """Write mph file"""

        def build_relative_href_path(path: str | Path):
            return f"./{Path(path).relative_to(self.product_path)}"

        name = self.product.name
        id_counter = 0

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
        xml4.text = self.product.start_time.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}resultTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimeInstant",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}timePosition")
        xml4.text = self.product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}validTime")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimePeriod",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}beginPosition")
        xml4.text = self.product.start_time.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.product.stop_time.isoformat(timespec="milliseconds")

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
            attrib={"codeSpace": "urn:esa:eop:P-SAR:operationalMode"},
        )
        xml6.text = "SM"
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}swathIdentifier",
            attrib={"codeSpace": "urn:esa:eop:P-SAR:swathIdentifier"},
        )
        xml6.text = self.product.swath_list[0]
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}acquisitionParameters")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["bio"] + "}Acquisition")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}orbitNumber")
        xml6.text = str(self.product.orbit_number)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}lastOrbitNumber")
        xml6.text = str(self.product.orbit_number)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}orbitDirection")
        xml6.text = self.product.orbit_direction
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}wrsLongitudeGrid",
            attrib={"codeSpace": "urn:esa:eop:Biomass:relativeOrbits"},
        )
        xml6.text = str(self.product.track_number)
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}wrsLatitudeGrid",
            attrib={"codeSpace": "urn:esa:eop:Biomass:frames"},
        )
        xml6.text = "___" if self.product.frame_number == 0 else str(self.product.frame_number)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}ascendingNodeDate")
        xml6.text = self.product.anx_time.isoformat(timespec="milliseconds")
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}startTimeFromAscendingNode",
            attrib={"uom": "ms"},
        )
        xml6.text = str(int((self.product.start_time - self.product.anx_time) * 1e3))
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}completionTimeFromAscendingNode",
            attrib={"uom": "ms"},
        )
        xml6.text = str(int((self.product.stop_time - self.product.anx_time) * 1e3))
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["sar"] + "}polarisationMode")
        xml6.text = "Q"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["sar"] + "}polarisationChannels")
        xml6.text = ", ".join(self.product.polarization_list).replace("/", "")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["sar"] + "}antennaLookDirection")
        xml6.text = "LEFT"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}missionPhase")
        xml6.text = self.product.mission_phase_id
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}instrumentConfID")
        xml6.text = str(self.product.instrument_configuration_id)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}dataTakeID")
        xml6.text = str(self.product.datatake_id)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}orbitDriftFlag")
        xml6.text = "false"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}globalCoverageID")
        xml6.text = encode_mph_id_value(self.product.global_coverage_id)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}majorCycleID")
        xml6.text = encode_mph_id_value(self.product.major_cycle_id)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}repeatCycleID")
        xml6.text = encode_mph_id_value(self.product.repeat_cycle_id)

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
        footprint = [self.product.footprint[i] for i in [0, 1, 2, 3, 0]]
        xml10.text = " ".join(f"{f[0]:.6f} {f[1]:.6f}" for f in footprint)
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}centerOf")
        id_counter += 1
        xml5 = ET.SubElement(
            xml4,
            "{" + MPH_NAMESPACES["gml"] + "}Point",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["gml"] + "}pos")
        footprint = self.product.footprint
        xml6.text = f"{(np.mean([f[0] for f in footprint])):.6f} {(np.mean([f[1] for f in footprint])):.6f}"

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}result")
        id_counter += 1
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["eop"] + "}EarthObservationResult",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(id_counter)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}browse")
        xml5 = ET.SubElement(xml4, "{" + MPH_NAMESPACES["eop"] + "}BrowseInformation")
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}type")
        xml6.text = "QUICKLOOK"
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}referenceSystemIdentifier",
            attrib={"codeSpace": "urn:esa:eop:crs"},
        )
        xml6.text = "EPSG:4326"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}fileName")
        xml7 = ET.SubElement(
            xml6,
            "{" + MPH_NAMESPACES["ows"] + "}ServiceReference",
            attrib={
                "{" + MPH_NAMESPACES["xlink"] + "}href": build_relative_href_path(
                    self.product_path.joinpath(self.content.quicklook)
                )
            },
        )
        xml8 = ET.SubElement(xml7, "{" + MPH_NAMESPACES["ows"] + "}RequestMessage")
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
        xml6.text = f"{self.product.baseline_id:02d}"

        def add_file_to_mph(
            base_element: ET.Element,
            file: Path,
            rds_file: Path | None,
        ):
            """Add product file to mph list"""
            file_size = file.stat().st_size

            product_elem = ET.SubElement(base_element, "{" + MPH_NAMESPACES["eop"] + "}product")
            product_information_elem = ET.SubElement(product_elem, "{" + MPH_NAMESPACES["bio"] + "}ProductInformation")
            file_name_elem = ET.SubElement(product_information_elem, "{" + MPH_NAMESPACES["eop"] + "}fileName")
            service_reference_elem = ET.SubElement(
                file_name_elem,
                "{" + MPH_NAMESPACES["ows"] + "}ServiceReference",
                attrib={"{" + MPH_NAMESPACES["xlink"] + "}href": build_relative_href_path(file)},
            )
            ET.SubElement(service_reference_elem, "{" + MPH_NAMESPACES["ows"] + "}RequestMessage")
            size_elem = ET.SubElement(
                product_information_elem,
                "{" + MPH_NAMESPACES["eop"] + "}size",
                attrib={"uom": "bytes"},
            )
            size_elem.text = str(file_size)
            if rds_file is not None:
                rds_elem = ET.SubElement(product_information_elem, "{" + MPH_NAMESPACES["bio"] + "}rds")
                rds_elem.text = build_relative_href_path(rds_file)

        for file, xsd_file in self.content.files_and_validators_if_any:
            file_path = self.product_path.joinpath(file)
            xsd_file_path = self.product_path.joinpath(xsd_file) if xsd_file is not None else None
            if file_path.exists():
                add_file_to_mph(xml3, file_path, xsd_file_path)

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["eop"] + "}metaDataProperty")
        xml3 = ET.SubElement(xml2, "{" + MPH_NAMESPACES["bio"] + "}EarthObservationMetaData")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}identifier")
        xml4.text = self.product.name
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}doi")
        xml4.text = self.product.doi
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}acquisitionType")
        xml4.text = "NOMINAL"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}productType")
        xml4.text = self.product.name[4:14]
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
        xml6.text = "other: " + ("L1a" if self.product.type == "SCS" else "L1b")
        for auxiliary_file in self.product.source_auxiliary_names:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
            xml6.text = auxiliary_file
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}processingMode",
            attrib={"codeSpace": "urn:esa:eop:Biomass:P-SAR:processingMode"},
        )
        xml6.text = "OPERATIONAL"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}sourceProduct")
        xml6.text = self.product.source_name
        if self.product.source_monitoring_name is not None:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}sourceProduct")
            xml6.text = self.product.source_monitoring_name
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}TAI-UTC")
        xml4.text = str(TAI_UTC)
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}isIncomplete")
        xml4.text = "true" if self.product.frame_status == "INCOMPLETE" else "false"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}isPartial")
        xml4.text = "true" if self.product.frame_status == "PARTIAL" else "false"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}isMerged")
        xml4.text = "true" if self.product.frame_status == "MERGED" else "false"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
        xml4.text = f"BIOMASS L1a/b/c Products Format Specification (BPS_L1_PFD) {BPS_PFD_L1_VERSION}"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
        xml4.text = f"BIOMASS Product Performance Description (BPS_PPD) {BPS_PPD_VERSION}"

        # Write MPH file
        xmlstr = minidom.parseString(ET.tostring(xml1)).toprettyxml(indent="   ")
        with open(
            file=self.product_path.joinpath(self.content.mph_file),
            mode="w",
            encoding="utf-8",
        ) as f:
            f.write(xmlstr)

    def _write_measurement_files(self):
        """Write raster files and vrt file"""

        nodata_mask = create_no_data_mask(
            data_shape=self.product.data_list[0].shape,
            product_type=self.product.type,
            processing_parameters=self.product.processing_parameters,
            swath=self.product.swath_list[0],
        )
        assert nodata_mask.shape == self.product.data_list[0].shape

        geotiff_conf_abs = GeotiffConf(
            compression_schema=self.product.sar_image_parameters.abs_compression_method.value,
            max_z_error=self.product.sar_image_parameters.abs_max_z_error,
            nodata_value=self.product.sar_image_parameters.no_pixel_value,
            block_size=self.product.sar_image_parameters.block_size,
            overview_resampling="NEAREST",
            gdal_num_threads=self.gdal_num_threads,
        )

        geotiff_conf_phase = copy.copy(geotiff_conf_abs)
        geotiff_conf_phase.compression_schema = self.product.sar_image_parameters.phase_compression_method.value
        geotiff_conf_phase.max_z_error = self.product.sar_image_parameters.phase_max_z_error

        geotiff_metadata = GeotiffMetadata(
            polarizations=[pol.replace("/", "") for pol in self.product.polarization_list],
            creation_date=CURRENT_AZIMUTH_TIME.isoformat(timespec="microseconds")[:-1],
            swath=self.product.swath_list[0],
            software=f"BPS-{self.processor_name}-{self.processor_version}",
            description="BIOMASS L1",
            matrix_representation="SCATTERING",
        )

        abs_data = [np.abs(d) for d in self.product.data_list]
        assert self.content.abs_raster is not None
        abs_file = self.product_path.joinpath(self.content.abs_raster)
        write_geotiff(
            output_geotiff_file=abs_file,
            data_list=abs_data,
            nodata_mask=nodata_mask,
            ecef_gcp_list=self.product.gcp_list,
            geotiff_metadata=geotiff_metadata,
            geotiff_conf=geotiff_conf_abs,
        )

        if self.product.type == "SCS":
            phase_data = [np.angle(d) for d in self.product.data_list]
            assert self.content.phase_raster is not None
            phase_file = self.product_path.joinpath(self.content.phase_raster)
            write_geotiff(
                output_geotiff_file=phase_file,
                data_list=phase_data,
                nodata_mask=nodata_mask,
                ecef_gcp_list=self.product.gcp_list,
                geotiff_metadata=geotiff_metadata,
                geotiff_conf=geotiff_conf_phase,
            )

            vrt_info = VRTInfo(
                raster_xsize=self.product.data_list[0].shape[1],
                raster_ysize=self.product.data_list[0].shape[0],
                ground_corner_points_ecef=self.product.gcp_list,
                abs_measurement_file=abs_file,
                phase_measurement_file=phase_file,
                geotiff_metadata=geotiff_metadata,
                abs_geotiff_conf=geotiff_conf_abs,
                phase_geotiff_conf=geotiff_conf_phase,
            )

            assert self.content.vrt is not None
            write_vrt_file(self.product_path.joinpath(self.content.vrt), vrt_info)

    def _write_main_annotation_file(self, added_luts):
        """Write main annotation"""

        polarisation_dict = {
            "H/H": common.PolarisationType.HH,
            "H/V": common.PolarisationType.HV,
            "V/H": common.PolarisationType.VH,
            "V/V": common.PolarisationType.VV,
        }
        mission_phase_id_dict = {
            "COMMISSIONING": common.MissionPhaseIdtype.COM,
            "INTERFEROMETRIC": common.MissionPhaseIdtype.INT,
            "TOMOGRAPHIC": common.MissionPhaseIdtype.TOM,
        }
        sensor_mode_dict = {
            "MEASUREMENT": common.SensorModeType.MEASUREMENT,
            "RX ONLY": common.SensorModeType.RX_ONLY,
            "EXTERNAL CALIBRATION": common.SensorModeType.EXTERNAL_CALIBRATION,
        }
        orbit_pass_dict = {
            "ASCENDING": common.OrbitPassType.ASCENDING,
            "DESCENDING": common.OrbitPassType.DESCENDING,
        }
        product_composition_dict = {
            "NOMINAL": common.ProductCompositionType.NOMINAL,
            "MERGED": common.ProductCompositionType.MERGED,
            "PARTIAL": common.ProductCompositionType.PARTIAL,
            "INCOMPLETE": common.ProductCompositionType.INCOMPLETE,
            "NOT_FRAMED": common.ProductCompositionType.NOT_FRAMED,
        }

        acquisition_raster_info = self.product.acquisition_raster_info or self.product.raster_info_list[0]

        # - acquisitionInformation
        assert isinstance(self.product.start_time, PreciseDateTime)
        assert isinstance(self.product.stop_time, PreciseDateTime)

        assert self.product.mission_phase_id is not None
        assert self.product.global_coverage_id is not None
        assert self.product.major_cycle_id is not None
        assert self.product.repeat_cycle_id is not None
        assert self.product.orbit_number is not None
        assert self.product.track_number is not None
        assert self.product.orbit_direction is not None
        assert self.product.datatake_id is not None
        assert self.product.type is not None
        acquisition_information = common_annotation_l1.AcquisitionInformationType(
            mission=common.MissionType.BIOMASS,
            swath=common.SwathType[self.product.swath_list[0]],
            product_type=common.ProductType[self.product.type],
            polarisation_list=[polarisation_dict[p] for p in self.product.polarization_list],
            start_time=self.product.start_time,
            stop_time=self.product.stop_time,
            mission_phase_id=mission_phase_id_dict[self.product.mission_phase_id],
            drift_phase_flag=True if self.product.repeat_cycle_id == -2 else False,
            sensor_mode=sensor_mode_dict[self.product.sensor_mode],
            global_coverage_id=self.product.global_coverage_id,
            major_cycle_id=self.product.major_cycle_id,
            repeat_cycle_id=self.product.repeat_cycle_id,
            absolute_orbit_number=self.product.orbit_number,
            relative_orbit_number=self.product.track_number,
            orbit_pass=orbit_pass_dict[self.product.orbit_direction],
            platform_heading=self.product.platform_heading,
            data_take_id=self.product.datatake_id,
            frame=self.product.frame_number,
            product_composition=product_composition_dict[self.product.frame_status],
        )

        # - sarImage
        raster_info_ref = self.product.raster_info_list[0]
        if self.product.type == "DGM":
            ground_to_slant_ref = self.product.ground_to_slant_list[0]
            g2s = genericpoly.create_sorted_poly_list(ground_to_slant_ref)

            first_sample_slant_range_time: float = g2s.evaluate(
                (raster_info_ref.lines_start, raster_info_ref.samples_start)
            )

            last_sample_slant_range_time: float = g2s.evaluate(
                (
                    raster_info_ref.lines_start,
                    raster_info_ref.samples_start + (raster_info_ref.samples - 1) * raster_info_ref.samples_step,
                )
            )
            range_pixel_spacing = raster_info_ref.samples_step
            range_time_interval = raster_info_ref.samples_step / (LIGHT_SPEED / 2)
        else:
            assert isinstance(raster_info_ref.samples_start, float)
            first_sample_slant_range_time = raster_info_ref.samples_start

            last_sample_slant_range_time = (
                raster_info_ref.samples_start + (raster_info_ref.samples - 1) * raster_info_ref.samples_step
            )
            range_pixel_spacing = raster_info_ref.samples_step * LIGHT_SPEED / 2
            range_time_interval = raster_info_ref.samples_step

        projection_dict = {
            "SCS": common.ProjectionType.SLANT_RANGE,
            "DGM": common.ProjectionType.GROUND_RANGE,
        }

        representation_dict = {
            "SCS": common.PixelRepresentationType.ABS_PHASE,
            "DGM": common.PixelRepresentationType.ABS,
        }

        coordinate_reference_system = (
            "GEOGCRS["
            "WGS 84"
            ",DATUM["
            "World Geodetic System 1984"
            ",ELLIPSOID["
            "WGS 84"
            ",6378137,298.257223563,LENGTHUNIT["
            "metre"
            ",1]]],PRIMEM["
            "Greenwich"
            ",0,ANGLEUNIT["
            "degree"
            ",0.0174532925199433]],CS[ellipsoidal,2],AXIS["
            "geodetic latitude (Lat)"
            ",north,ORDER[1],ANGLEUNIT["
            "degree"
            ",0.0174532925199433]],AXIS["
            "geodetic longitude (Lon)"
            ",east,ORDER[2],ANGLEUNIT["
            "degree"
            ",0.0174532925199433]],ID["
            "EPSG"
            ",4326]]"
        )
        datum = common.DatumType(coordinate_reference_system, common.GeodeticReferenceFrameType.WGS84)

        coordinate_conversion = []
        if self.product.type == "DGM":
            ground_to_slant_ref = self.product.ground_to_slant_list[0]
            slant_to_ground_ref = self.product.slant_to_ground_list[0]

            if ground_to_slant_ref.get_number_of_poly() == 0 and slant_to_ground_ref.get_number_of_poly() == 0:
                coordinate_conversion.append(fill_empty_coordinate_conversion_type(self.product.start_time))
            else:
                for index in range(ground_to_slant_ref.get_number_of_poly()):
                    g2s_poly = ground_to_slant_ref.get_poly(index)
                    s2g_poly = (
                        slant_to_ground_ref.get_poly(index) if slant_to_ground_ref.get_number_of_poly() > 0 else None
                    )
                    coordinate_conversion.append(fill_coordinate_conversion_type(g2s_poly, s2g_poly))

        assert isinstance(raster_info_ref.lines_start, PreciseDateTime)
        sar_image = common_annotation_l1.SarImageType(
            first_sample_slant_range_time=first_sample_slant_range_time,
            last_sample_slant_range_time=last_sample_slant_range_time,
            first_line_azimuth_time=raster_info_ref.lines_start,
            last_line_azimuth_time=raster_info_ref.lines_start
            + (raster_info_ref.lines - 1) * raster_info_ref.lines_step,
            range_time_interval=range_time_interval,
            azimuth_time_interval=raster_info_ref.lines_step,
            range_pixel_spacing=range_pixel_spacing,
            azimuth_pixel_spacing=raster_info_ref.lines_step * AVERAGE_GROUND_VELOCITY,
            number_of_samples=raster_info_ref.samples,
            number_of_lines=raster_info_ref.lines,
            projection=projection_dict[self.product.type],
            range_coordinate_conversion=coordinate_conversion,
            datum=datum,
            footprint=np.reshape(self.product.footprint, (8,)).tolist(),
            pixel_representation=representation_dict[self.product.type],
            pixel_type=translate_common.translate_pixel_type_type(self.product.sar_image_parameters.pixel_type),
            pixel_quantity=common.PixelQuantityType.BETA_NOUGHT,
            no_data_value=self.product.sar_image_parameters.no_pixel_value,
        )

        # - instrumentParameters
        first_line_sensing_time_list = {
            polarisation_dict[p]: acquisition_raster_info.lines_start_date for p in self.product.polarization_list
        }
        last_line_sensing_time_list = {
            polarisation_dict[p]: acquisition_raster_info.lines_start_date
            + acquisition_raster_info.lines_step * (acquisition_raster_info.lines - 1)
            for p in self.product.polarization_list
        }
        number_of_input_samples = acquisition_raster_info.samples
        number_of_input_lines = acquisition_raster_info.lines
        if self.product.acquisition_timeline is not None:
            swp_list = self.product.acquisition_timeline.swp_list
            swl_list = self.product.acquisition_timeline.swl_list
            prf_list = self.product.acquisition_timeline.prf_list
        else:
            swp_list = [(self.product.start_time, 0.0)]
            swl_list = [(self.product.start_time, 0.0)]
            prf_list = [(self.product.start_time, 0.0)]

        rank = self.product.swath_info_list[0].rank
        tx_pulse_list = [fill_tx_pulse(self.product.pulse_list[0], self.product.start_time)]
        assert self.product.instrument_configuration_id is not None
        instrument_configuration_id = self.product.instrument_configuration_id
        radar_carrier_frequency = self.product.dataset_info[0].fc_hz
        assert radar_carrier_frequency is not None

        interleaved_calibration_flag = True

        if self.product.preproc_report is not None:
            rx_gain_list = {
                common.PolarisationType.H: get_rx_gain_from_parameters_code(
                    self.product.preproc_report.gain_param_code_h
                ),
                common.PolarisationType.V: get_rx_gain_from_parameters_code(
                    self.product.preproc_report.gain_param_code_v
                ),
            }
            preamble_flag = self.product.preproc_report.noise_preamble_present
            postamble_flag = self.product.preproc_report.noise_postamble_present
            data_format = common_annotation_l1.DataFormatType(
                echo_format=convert_baq_compression_level(self.product.preproc_report.echo_baq_compression),
                calibration_format=convert_baq_compression_level(
                    self.product.preproc_report.calibration_baq_compression
                ),
                noise_format=convert_baq_compression_level(self.product.preproc_report.noise_baq_compression),
                mean_bit_rate=self.product.source_bit_rate,
            )
        else:
            rx_gain_list = {common.PolarisationType.H: 1.0, common.PolarisationType.V: 1.0}
            preamble_flag = True
            postamble_flag = True
            data_format = common_annotation_l1.DataFormatType(
                echo_format=common.DataFormatModeType.BAQ_4_BIT,
                calibration_format=common.DataFormatModeType.BYPASS,
                noise_format=common.DataFormatModeType.BYPASS,
                mean_bit_rate=self.product.source_bit_rate,
            )
        instrument_parameters = common_annotation_l1.InstrumentParametersType(
            first_line_sensing_time_list,
            last_line_sensing_time_list,
            number_of_input_samples,
            number_of_input_lines,
            swp_list,
            swl_list,
            prf_list,
            rank,
            tx_pulse_list,
            instrument_configuration_id,
            radar_carrier_frequency,
            rx_gain_list,
            preamble_flag,
            postamble_flag,
            interleaved_calibration_flag,
            data_format,
        )

        # - rawDataAnalysis
        raw_data_analysis_values = (
            RawDataAnalysis.from_report(self.product.preproc_report)
            if self.product.preproc_report is not None
            else RawDataAnalysis.from_polarization_list(self.product.polarization_list)
        )
        raw_data_analysis = raw_data_analysis_values.to_annotation()

        # - processingParameters
        processing_parameters = common_annotation_l1.ProcessingParameters(
            processor_version=self.processor_version,
            product_generation_time=CURRENT_AZIMUTH_TIME,
            processing_mode=common.ProcessingModeType.NOMINAL,
            orbit_source=common.OrbitAttitudeSourceType.AUXILIARY,
            attitude_source=common.OrbitAttitudeSourceType.AUXILIARY,
            raw_data_correction_flag=(self.product.processing_parameters.raw_data_correction_flag),
            rfi_detection_flag=self.product.processing_parameters.rfi_detection_flag,
            rfi_correction_flag=self.product.processing_parameters.rfi_correction_flag,
            rfi_mitigation_method=self.product.processing_parameters.rfi_mitigation_method,
            rfi_mask=self.product.processing_parameters.rfi_mask,
            rfi_mask_generation_method=(self.product.processing_parameters.rfi_mask_generation_method),
            rfi_fm_mitigation_method=self.product.processing_parameters.rfi_fm_mitigation_method,
            rfi_fm_chirp_source=self.product.processing_parameters.rfi_fm_chirp_source,
            internal_calibration_estimation_flag=(
                self.product.processing_parameters.internal_calibration_estimation_flag
            ),
            internal_calibration_correction_flag=(
                self.product.processing_parameters.internal_calibration_correction_flag
            ),
            range_reference_function_source=(self.product.processing_parameters.range_reference_function_source),
            range_compression_method=(self.product.processing_parameters.range_compression_method),
            extended_swath_processing_flag=self.product.processing_parameters.extended_swath_processing,
            dc_method=self.product.processing_parameters.dc_method,
            dc_value=self.product.processing_parameters.dc_value,  # Hz
            antenna_pattern_correction1_flag=(self.product.processing_parameters.antenna_pattern_correction1_flag),
            antenna_pattern_correction2_flag=(self.product.processing_parameters.antenna_pattern_correction2_flag),
            antenna_cross_talk_correction_flag=(self.product.processing_parameters.antenna_cross_talk_correction_flag),
            range_processing_parameters=common_annotation_l1.SpectrumProcessingParametersType(
                window_type=self.product.processing_parameters.range_window_type[self.product.swath_list[0]],
                window_coefficient=(
                    self.product.processing_parameters.range_window_coefficient[self.product.swath_list[0]]
                ),
                total_bandwidth=self.product.sampling_constants_list[0].frg_hz,  # Hz
                processing_bandwidth=self.product.sampling_constants_list[0].brg_hz,  # Hz
                look_bandwidth=self.product.sampling_constants_list[0].brg_hz,  # Hz
                number_of_looks=1,
                look_overlap=0,  # Hz
            ),
            azimuth_processing_parameters=common_annotation_l1.SpectrumProcessingParametersType(
                window_type=self.product.processing_parameters.azimuth_window_type[self.product.swath_list[0]],
                window_coefficient=(
                    self.product.processing_parameters.azimuth_window_coefficient[self.product.swath_list[0]]
                ),
                total_bandwidth=self.product.sampling_constants_list[0].faz_hz,  # Hz
                processing_bandwidth=self.product.sampling_constants_list[0].baz_hz,  # Hz
                look_bandwidth=self.product.sampling_constants_list[0].baz_hz,  # Hz
                number_of_looks=1,
                look_overlap=0,
            ),  # Hz
            bistatic_delay_correction_flag=(self.product.processing_parameters.bistatic_delay_correction_flag),
            bistatic_delay_correction_method=(self.product.processing_parameters.bistatic_delay_correction_method),
            range_spreading_loss_compensation_flag=(
                self.product.processing_parameters.range_spreading_loss_compensation_flag
            ),
            reference_range=self.product.processing_parameters.reference_range,  # M
            processing_gain_list=[
                (1, common.PolarisationType(p.replace("/", ""))) for p in self.product.polarization_list
            ],
            polarimetric_correction_flag=(self.product.processing_parameters.polarimetric_correction_flag),
            ionosphere_height_defocusing_flag=(self.product.processing_parameters.ionosphere_height_defocusing_flag),
            ionosphere_height_estimation_method=(
                self.product.processing_parameters.ionosphere_height_estimation_method
            ),
            faraday_rotation_correction_flag=(self.product.processing_parameters.faraday_rotation_correction_flag),
            ionospheric_phase_screen_correction_flag=(
                self.product.processing_parameters.ionospheric_phase_screen_correction_flag
            ),
            group_delay_correction_flag=(self.product.processing_parameters.group_delay_correction_flag),
            autofocus_flag=self.product.processing_parameters.autofocus_flag,
            autofocus_method=self.product.processing_parameters.autofocus_method,
            detection_flag=(False if self.product.type == "SCS" else self.product.processing_parameters.detection_flag),
            thermal_denoising_flag=(
                False if self.product.type == "SCS" else self.product.processing_parameters.thermal_denoising_flag
            ),
            ground_projection_flag=(
                False if self.product.type == "SCS" else self.product.processing_parameters.ground_projection_flag
            ),
            noise_gain_list=[(1, common.PolarisationType(p.replace("/", ""))) for p in self.product.polarization_list],
        )

        # - internalCalibration
        internal_calibration_parameters_list = (
            InternalCalibrationParameters.from_report(
                self.product.preproc_report, self.product.polarization_list, self.product.quality_parameters
            )
            if self.product.preproc_report is not None
            else InternalCalibrationParameters.from_polarization_list(
                self.product.polarization_list, [self.product.start_time, self.product.stop_time]
            )
        ).to_annotations()

        if self.product.preproc_report is not None:
            noise_list = {}

            for polarisation in self.product.polarization_list:
                polarisation_enum = polarisation_dict[polarisation]
                rx_pol = polarisation_enum.value[1]
                preamble = (
                    self.product.preproc_report.noise_preamble_h
                    if rx_pol == "H"
                    else self.product.preproc_report.noise_preamble_v
                )
                postamble = (
                    self.product.preproc_report.noise_postamble_h
                    if rx_pol == "H"
                    else self.product.preproc_report.noise_postamble_v
                )
                noise_sequence = [
                    common_annotation_l1.NoiseSequenceType(
                        azimuth_time=sequence.time,
                        noise_power_correction_factor=sequence.average_noise,
                        number_of_noise_lines=sequence.num_lines,
                    )
                    for sequence in [preamble, postamble]
                    if sequence.num_lines > 0
                ]
                noise_list[polarisation_enum] = noise_sequence
        else:
            noise_sequence = [
                common_annotation_l1.NoiseSequenceType(
                    azimuth_time=azimuth_time,
                    noise_power_correction_factor=1,
                    number_of_noise_lines=0,
                )
                for azimuth_time in [self.product.start_time, self.product.stop_time]
            ]
            noise_list = {polarisation_dict[p]: noise_sequence for p in self.product.polarization_list}

        internal_calibration = common_annotation_l1.InternalCalibrationType(
            internal_calibration_parameters_used=(self.product.processing_parameters.internal_calibration_source),
            range_reference_function_used=(self.product.processing_parameters.range_reference_function_source),
            noise_parameters_used=(self.product.processing_parameters.noise_parameters_source),
            internal_calibration_parameters_list=internal_calibration_parameters_list,
            noise_list=noise_list,
        )

        # - rfiMitigation
        rfi_mitigation = common_annotation_l1.RfiMitigationType(
            rfi_tmreport_list=[],
            rfi_isolated_fmreport_list=[],
            rfi_persistent_fmreport_list=[],
        )
        if self.product.rfi_masks_statistics is not None:
            if self.product.rfi_masks_statistics.time_stats:
                rfi_mitigation.rfi_tmreport_list = translate_rfi_time_stats(
                    self.product.rfi_masks_statistics.time_stats
                )
            if self.product.rfi_masks_statistics.freq_stats:
                (
                    rfi_mitigation.rfi_isolated_fmreport_list,
                    rfi_mitigation.rfi_persistent_fmreport_list,
                ) = translate_rfi_freq_stats_to_model(self.product.rfi_masks_statistics.freq_stats)

        # - dopplerParameters
        def _build_empty_stats(number_of_poly: int):
            combined_stats_list = [CombinedDCStatistics([], [], 0.0)] * number_of_poly
            combined_dcrmserror_above_threshold_list = [False] * number_of_poly
            return combined_stats_list, combined_dcrmserror_above_threshold_list

        geometry_dcpolynomial_list = None
        combined_dcpolynomial_list = None
        combined_stats_list = None
        combined_dcrmserror_above_threshold_list = None
        if self.product.processing_parameters.dc_method == common.DcMethodType.GEOMETRY:
            geometry_dcpolynomial_list = self.product.dc_vector_list[0]
            combined_stats_list, combined_dcrmserror_above_threshold_list = _build_empty_stats(
                geometry_dcpolynomial_list.get_number_of_poly()
            )
        elif self.product.processing_parameters.dc_method in (
            common.DcMethodType.COMBINED,
            common.DcMethodType.FIXED,
        ):
            combined_dcpolynomial_list = self.product.dc_vector_list[0]
            if self.product.dc_annotations and combined_dcpolynomial_list.get_number_of_poly() > 0:
                reference_pol = list(self.product.dc_annotations.geometric_dc.keys())[0]
                geometry_dcpolynomial_list = self.product.dc_annotations.geometric_dc.get(reference_pol)

                combined_stats_list = build_list_of_combined_dc_stats_from_annotations_and_poly(
                    dc_annotations=self.product.dc_annotations,
                    combined_dc=combined_dcpolynomial_list,
                )
                combined_dcrmserror_above_threshold_list = [
                    combined_stats.rmse > self.product.quality_parameters.dc_rmserror_threshold
                    for combined_stats in combined_stats_list
                ]
            else:
                combined_stats_list, combined_dcrmserror_above_threshold_list = _build_empty_stats(
                    combined_dcpolynomial_list.get_number_of_poly()
                )
        else:
            raise RuntimeError(f"Unknown DC estimation method: {self.product.processing_parameters.dc_method}")

        dc_estimate: list[common_annotation_l1.DcEstimateType] = []

        geometric_polys = get_list_of_dc_poly(geometry_dcpolynomial_list)
        combined_polys = get_list_of_dc_poly(combined_dcpolynomial_list)

        for (
            geo_poly,
            combined_poly,
            combined_stats,
            combined_dcrmserror_above_threshold,
        ) in itertools.zip_longest(
            geometric_polys,
            combined_polys,
            combined_stats_list,
            combined_dcrmserror_above_threshold_list,
            fillvalue=None,
        ):
            reference_poly = (
                geo_poly
                if combined_poly is None
                or (
                    self.product.processing_parameters.dc_method == common.DcMethodType.GEOMETRY
                    and geo_poly is not None
                )
                else combined_poly
            )
            assert reference_poly is not None

            assert isinstance(reference_poly.t_ref_az, PreciseDateTime)
            assert reference_poly.t_ref_rg is not None

            dc_estimate.append(
                common_annotation_l1.DcEstimateType(
                    azimuth_time=reference_poly.t_ref_az,
                    t0=reference_poly.t_ref_rg,
                    geometry_dcpolynomial=get_coefficients_from_poly(geo_poly),
                    combined_dcpolynomial=get_coefficients_from_poly(combined_poly),
                    combined_dcslant_range_times=(
                        combined_stats.slant_range_times if combined_stats is not None else []
                    ),
                    combined_dcvalues=(combined_stats.values if combined_stats is not None else []),
                    combined_dcrmserror=(combined_stats.rmse if combined_stats is not None else 0.0),
                    combined_dcrmserror_above_threshold=combined_dcrmserror_above_threshold or False,
                )
            )

        if len(dc_estimate) == 0:
            dc_estimate.append(
                common_annotation_l1.DcEstimateType(
                    azimuth_time=self.product.start_time,
                    t0=0.0,
                    geometry_dcpolynomial=[],
                    combined_dcpolynomial=[],
                    combined_dcslant_range_times=[],
                    combined_dcvalues=[],
                    combined_dcrmserror=0.0,
                    combined_dcrmserror_above_threshold=False,
                )
            )

        dr_vector_list_ref = self.product.dr_vector_list[0]
        fm_rate_estimate = []
        if dr_vector_list_ref.get_number_of_poly() == 0:
            fm_rate_estimate.append(
                common.SlantRangePolynomialType(azimuth_time=self.product.start_time, t0=0.0, polynomial=[])
            )
        else:
            for index in range(dr_vector_list_ref.get_number_of_poly()):
                dr_poly = dr_vector_list_ref.get_poly(index)
                coefficients = np.concatenate((dr_poly.coefficients[0:2], dr_poly.coefficients[4:]))

                fm_rate_estimate.append(
                    common.SlantRangePolynomialType(dr_poly.t_ref_az, dr_poly.t_ref_rg, coefficients.tolist())
                )
        doppler_parameters = common_annotation_l1.DopplerParametersType(
            dc_estimate_list=dc_estimate, fm_rate_estimate_list=fm_rate_estimate
        )

        # - radiometricCalibration
        radiometric_calibration = common_annotation_l1.RadiometricCalibrationType(
            absolute_calibration_constant_list=self.product.processing_parameters.absolute_calibration_constants
        )

        # - polarimetricDistortion
        polarimetric_distortion = self.product.processing_parameters.polarimetric_distortion

        # - ionosphereCorrection
        if self.product.iono_cal_report is not None:
            ionospheric_annotation = self.product.iono_cal_report.ionosphere_correction
        else:
            ionospheric_annotation = common_annotation_l1.IonosphereCorrection(
                ionosphere_height_used=-1.0,
                ionosphere_height_estimated=-1.0,
                ionosphere_height_estimation_method_selected=(common.IonosphereHeightEstimationMethodType.NA),
                ionosphere_height_estimation_latitude_value=float(np.mean(self.product.footprint[::2])),
                ionosphere_height_estimation_flag=False,
                ionosphere_height_estimation_method_used=(common.IonosphereHeightEstimationMethodType.NA),
                gaussian_filter_computation_flag=False,
                faraday_rotation_correction_applied=False,
                autofocus_shifts_applied=False,
            )

        # - geometry
        requested_height_model = common.HeightModelType(
            value=self.product.processing_parameters.requested_height_model,
            version=self.product.processing_parameters.requested_height_model_version,
        )

        assert isinstance(self.product.start_time, PreciseDateTime)
        geometry = common_annotation_l1.GeometryType(
            height_model=requested_height_model,
            height_model_used_flag=self.product.processing_parameters.requested_height_model_used,
            roll_bias=0.0,
        )

        # - quality
        missing_ispfraction = raw_data_analysis.error_counters.num_isp_missing / number_of_input_lines
        max_ispgap = raw_data_analysis.error_counters.num_isp_missing
        invalid_raw_data_samples = {}
        qp = self.product.quality_parameters
        raw_mean_min = qp.raw_mean_expected - qp.raw_mean_threshold
        raw_mean_max = qp.raw_mean_expected + qp.raw_mean_threshold
        raw_std_min = qp.raw_std_expected - qp.raw_std_threshold
        raw_std_max = qp.raw_std_expected + qp.raw_std_threshold
        for stats in raw_data_analysis_values.raw_data_statistics_list:
            invalid_raw_data_samples[stats.polarization] = (
                0.0
                if (
                    raw_mean_min < stats.bias_i < raw_mean_max
                    and raw_mean_min < stats.bias_q < raw_mean_max
                    and raw_std_min < stats.std_dev_i < raw_std_max
                    and raw_std_min < stats.std_dev_q < raw_std_max
                )
                else 1.0
            )
        rfi_tmfraction = {}
        if rfi_mitigation.rfi_tmreport_list:
            for rtm in rfi_mitigation.rfi_tmreport_list:
                rfi_tmfraction[rtm.polarisation] = rtm.avg_percentage_affected_samples
        else:
            for _, p in polarisation_dict.items():
                rfi_tmfraction[p] = 0.0
        rfi_fmfraction = {}
        if rfi_mitigation.rfi_persistent_fmreport_list:
            for rfm in rfi_mitigation.rfi_persistent_fmreport_list:
                rfi_fmfraction[rfm.polarisation] = rfm.max_percentage_affected_bw / 100
        else:
            for _, p in polarisation_dict.items():
                rfi_fmfraction[p] = 0.0
        invalid_drift_fraction = 0.0
        invalid_replica_fraction = {}
        for pol, icp in internal_calibration_parameters_list.items():
            invalid_replica_fraction[pol] = (
                sum([not icps.reconstructed_replica_valid_flag for icps in icp]) / len(icp) if icp else 0.0
            )
        invalid_dcestimates_fraction = sum(combined_dcrmserror_above_threshold_list) / len(
            combined_dcrmserror_above_threshold_list
        )
        residual_ionospheric_phase_screen_std = (
            float(np.std(self.product_lut[ProductLUTID.PHASE_SCREEN_BB].data_list[0]))
            if self.product_lut.get(ProductLUTID.PHASE_SCREEN_BB) is not None
            else 0.0
        )
        invalid_blocks_percentage = 0
        invalid_blocks_percentage_threshold = 0

        iri_model_used = (
            ionospheric_annotation.ionosphere_height_estimation_method_used
            == common.IonosphereHeightEstimationMethodType.MODEL
        )
        iri_model_requested = (
            self.product.processing_parameters.ionosphere_height_estimation_method
            == common.IonosphereHeightEstimationMethodType.MODEL
        )

        if self.product.iono_cal_report is not None:
            geomagnetic_equator_fallback_activated = (
                not self.product.iono_cal_report.constant_sign_geomagnetic_field
                and not self.product.iono_cal_report.phase_screen_correction_applied
                and not self.product.iono_cal_report.group_delay_correction_applied
            )
        else:
            geomagnetic_equator_fallback_activated = False

        quality_index = L1QualityIndex(
            raw_data_stats_out_of_boundaries=any(v > 0 for _, v in invalid_raw_data_samples.items()),
            int_cal_sequence_above_threshold=invalid_drift_fraction > 0
            or any(v > 0 for _, v in invalid_replica_fraction.items()),
            dc_fallback_activated=self.product.dc_fallback_activated,
            dc_rmse_above_threshold=invalid_dcestimates_fraction > 0,
            ionosphere_height_estimation_failure=not ionospheric_annotation.ionosphere_height_estimation_flag,
            iri_model_used_as_fallback=iri_model_used and not iri_model_requested,
            gaussian_filter_size_out_of_boundaries=not ionospheric_annotation.gaussian_filter_computation_flag,
            inconsistent_phasescreen_and_rangeshifts_luts=False,
            geomagnetic_equator_fallback_activated=geomagnetic_equator_fallback_activated,
            number_of_failing_estimations_above_threshold=not ionospheric_annotation.autofocus_shifts_applied,
        )
        overall_product_quality_index = quality_index.encode()

        quality_parameters = [
            common_annotation_l1.QualityParametersType(
                missing_ispfraction=missing_ispfraction,
                max_ispgap=max_ispgap,
                max_ispgap_threshold=self.product.quality_parameters.max_isp_gap,
                invalid_raw_data_samples=invalid_raw_data_samples[p],
                raw_mean_expected=self.product.quality_parameters.raw_mean_expected,
                raw_mean_threshold=self.product.quality_parameters.raw_mean_threshold,
                raw_std_expected=self.product.quality_parameters.raw_std_expected,
                raw_std_threshold=self.product.quality_parameters.raw_std_threshold,
                rfi_tmfraction=rfi_tmfraction[polarisation_dict[p]],
                max_rfitmpercentage=self.product.quality_parameters.max_rfi_tm_percentage,
                rfi_fmfraction=rfi_fmfraction[polarisation_dict[p]],
                max_rfifmpercentage=self.product.quality_parameters.max_rfi_fm_percentage,
                invalid_drift_fraction=invalid_drift_fraction,
                max_invalid_drift_fraction=self.product.quality_parameters.max_invalid_drift_fraction,
                invalid_replica_fraction=invalid_replica_fraction[polarisation_dict[p]],
                invalid_dcestimates_fraction=invalid_dcestimates_fraction,
                dc_rmserror_threshold=self.product.quality_parameters.dc_rmserror_threshold,
                residual_ionospheric_phase_screen_std=residual_ionospheric_phase_screen_std,
                invalid_blocks_percentage=invalid_blocks_percentage,
                invalid_blocks_percentage_threshold=invalid_blocks_percentage_threshold,
                polarisation=polarisation_dict[p],
            )
            for p in self.product.polarization_list
        ]

        quality = common_annotation_l1.QualityType(overall_product_quality_index, quality_parameters)

        # - annotationLUT
        translation_table = {
            "rfiTimeMaskHH": common.LayerType.RFI_TIME_DOMAIN_MASK_HH,
            "rfiTimeMaskHV": common.LayerType.RFI_TIME_DOMAIN_MASK_HV,
            "rfiTimeMaskVH": common.LayerType.RFI_TIME_DOMAIN_MASK_VH,
            "rfiTimeMaskVV": common.LayerType.RFI_TIME_DOMAIN_MASK_VV,
            "rfiFreqMaskHH": common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HH,
            "rfiFreqMaskHV": common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_HV,
            "rfiFreqMaskVH": common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VH,
            "rfiFreqMaskVV": common.LayerType.RFI_FREQUENCY_DOMAIN_MASK_VV,
            "phaseScreen": common.LayerType.PHASE_SCREEN_RAD,
            "faradayRotationPlane": common.LayerType.FARADAY_ROTATION_PLANE_RAD,
            "faradayRotation": common.LayerType.FARADAY_ROTATION_RAD,
            "faradayRotationStd": common.LayerType.FARADAY_ROTATION_STD_RAD,
            "tec": common.LayerType.TEC_TECU,
            "rangeShifts": common.LayerType.RANGE_SHIFTS_M,
            "azimuthShifts": common.LayerType.AZIMUTH_SHIFTS_M,
            "autofocusPhaseScreen": common.LayerType.AUTOFOCUS_PHASE_SCREEN_RAD,
            "autofocusPhaseScreenStd": common.LayerType.AUTOFOCUS_PHASE_SCREEN_STD_RAD,
            "denoisingHH": common.LayerType.DENOISING_MAP_HH,
            "denoisingHV": common.LayerType.DENOISING_MAP_HV,
            "denoisingVH": common.LayerType.DENOISING_MAP_VH,
            "denoisingVV": common.LayerType.DENOISING_MAP_VV,
            "latitude": common.LayerType.LATITUDE_DEG,
            "longitude": common.LayerType.LONGITUDE_DEG,
            "height": common.LayerType.HEIGHT_M,
            "incidenceAngle": common.LayerType.INCIDENCE_ANGLE_DEG,
            "elevationAngle": common.LayerType.ELEVATION_ANGLE_DEG,
            "terrainSlope": common.LayerType.TERRAIN_SLOPE_DEG,
            "sigmaNought": common.LayerType.SIGMA_NOUGHT_LUT,
            "gammaNought": common.LayerType.GAMMA_NOUGHT_LUT,
        }
        layers = [translation_table[lut] for lut in added_luts]

        annotation_l1ab = main_annotation_l1ab.MainAnnotationL1ab(
            acquisition_information=acquisition_information,
            sar_image=sar_image,
            instrument_parameters=instrument_parameters,
            raw_data_analysis=raw_data_analysis,
            processing_parameters=processing_parameters,
            internal_calibration=internal_calibration,
            rfi_mitigation=rfi_mitigation,
            doppler_parameters=doppler_parameters,
            radiometric_calibration=radiometric_calibration,
            polarimetric_distortion=polarimetric_distortion,
            ionosphere_correction=ionospheric_annotation,
            geometry=geometry,
            quality=quality,
            annotation_lut=layers,
        )
        main_annotation_model = translate_main_annotation_l1ab.translate_main_annotation_l1ab_to_model(annotation_l1ab)

        # Write main annotation file
        main_annotation_text = serialize(main_annotation_model)
        main_annotation_path = self.product_path.joinpath(self.content.main_annotation)
        main_annotation_path.write_text(main_annotation_text, encoding="utf-8")

    def write(
        self,
        orbit_xml_template_string: str,
        attitude_xml_template_string: str,
        quicklook_file: Path | None = None,
    ):
        """Write product to disk"""
        bps_logger.info(f"Writing BIOMASS L1 product: {self.product.name}")
        creation_date = PreciseDateTime.now().isoformat(timespec="seconds")[:-1]

        for folder in self.content.folders:
            self.product_path.joinpath(folder).mkdir(parents=True, exist_ok=True)

        # Write measurement files
        if not self.product.is_monitoring:
            bps_logger.debug("..measurement files")
            self._write_measurement_files()

        # Write annotation files
        bps_logger.debug("..annotation files")
        added_luts = write_lut_file(
            self.product_path.joinpath(self.content.lut),
            self.product,
            self.product_lut,
        )
        self._write_main_annotation_file(added_luts)

        # Write orbit and attitude files
        bps_logger.debug("..orbit and attitude files")

        # Navigation files
        # Navigation files - orbit
        orbit_file_path = self.product_path.joinpath(self.content.orbit)
        orbit_xml_string = replace_template_string(orbit_xml_template_string, creation_date, orbit_file_path.stem)
        orbit_file_path.write_text(orbit_xml_string, encoding="utf-8")

        # Navigation files - attitude
        attitude_file_path = self.product_path.joinpath(self.content.attitude)
        attitude_xml_string = replace_template_string(
            attitude_xml_template_string, creation_date, attitude_file_path.stem
        )
        attitude_file_path.write_text(attitude_xml_string, encoding="utf-8")

        # Write quick-look file
        if quicklook_file is not None:
            bps_logger.debug("..quick-look file")
            shutil.copyfile(quicklook_file, self.product_path.joinpath(self.content.quicklook))

        # Write overlay file
        bps_logger.debug("..overlay file")
        write_overlay_file(
            self.product_path.joinpath(self.content.overlay),
            self.product_path.joinpath(self.content.quicklook),
            self.product.name,
            self.product.footprint,
            "L1 Product Overlay ADS",
        )

        # Write schema files
        bps_logger.debug("..schema files")
        schema_files_names = [xsd_file.name for xsd_file in self.content.xsd_schema_files]
        copy_biomass_xsd_files(
            self.product_path.joinpath(self.content.schema_folder),
            schema_files_names,
        )

        # Write MPH file
        bps_logger.debug("..MPH file")
        self._write_mph_file()

        bps_logger.info("..done")
