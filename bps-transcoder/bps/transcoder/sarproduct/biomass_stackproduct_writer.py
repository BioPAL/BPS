# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Writer for the BIOMASS L1c Products
-----------------------------------
"""

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generic
from xml.dom import minidom

import numpy as np
import numpy.typing as npt
from arepytools.io.metadata import RasterInfo
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io.common import UomType
from bps.common.io.parsing import parse, serialize
from bps.common.io.translate_common import translate_float_array_with_units_to_model
from bps.transcoder.io import common_annotation_models_l1 as main_annotation_models
from bps.transcoder.io import main_annotation_models_l1c
from bps.transcoder.sarproduct.biomass_stackproduct import (
    LUT_LAYERS,
    BIOMASSStackProduct,
    InvalidBIOMASSStackProductError,
)
from bps.transcoder.sarproduct.footprint_utils import (
    is_null_footprint,
    serialize_footprint,
)
from bps.transcoder.sarproduct.l1.product_content import L1ProductContent
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES
from bps.transcoder.sarproduct.overlay import write_overlay_file
from bps.transcoder.sarproduct.sta.product_content import BIOMASSStackProductStructure
from bps.transcoder.sarproduct.sta.stack_unique_identifier import StackUniqueID
from bps.transcoder.sarproduct.vrt import VRTInfo, write_vrt_file
from bps.transcoder.utils.gdal_utils import GeotiffConf, GeotiffMetadata, write_geotiff
from bps.transcoder.utils.polarization_conversions import translate_polarization_tag
from bps.transcoder.utils.production_model_utils import (
    encode_mph_id_value,
    translate_global_coverage_id,
    translate_major_cycle_id,
    translate_repeat_cycle_id,
)
from bps.transcoder.utils.quicklook_utils import (
    QuickLookConf,
    compute_quicklook_from_pol_data,
    write_quicklook_to_file,
)
from bps.transcoder.utils.time_conversions import (
    no_zulu_isoformat,
    round_precise_datetime,
)
from bps.transcoder.utils.xsd_schema_attacher import copy_biomass_xsd_files
from netCDF4 import Dataset as NetCDF4Dataset
from netCDF4 import Group as NetCDF4Group

# Default compression level for L1c products.
ZLIB_L1C_COMPLEVEL = 2

# Field to use in case of missing field from L1a product.
MISSING_FROM_INPUT_PRODUCT = "N/A"

for key, value in MPH_NAMESPACES.items():
    ET.register_namespace(key, value)

# Mapping between LUT names and official LUT layers.
STACK_LUT_LAYER_MAP = {
    # Radiometry LUTs.
    "gammaNought": main_annotation_models.LayerType.GAMMA_NOUGHT_LUT,
    "sigmaNought": main_annotation_models.LayerType.SIGMA_NOUGHT_LUT,
    # Denoising LUTs.
    "denoisingHH": main_annotation_models.LayerType.DENOISING_MAP_HH,
    "denoisingHV": main_annotation_models.LayerType.DENOISING_MAP_HV,
    "denoisingXX": main_annotation_models.LayerType.DENOISING_MAP_XX,
    "denoisingVH": main_annotation_models.LayerType.DENOISING_MAP_VH,
    "denoisingVV": main_annotation_models.LayerType.DENOISING_MAP_VV,
    # Geometry LUTs.
    "latitude": main_annotation_models.LayerType.LATITUDE_DEG,
    "longitude": main_annotation_models.LayerType.LONGITUDE_DEG,
    "height": main_annotation_models.LayerType.HEIGHT_M,
    "incidenceAngle": main_annotation_models.LayerType.INCIDENCE_ANGLE_DEG,
    "elevationAngle": main_annotation_models.LayerType.ELEVATION_ANGLE_DEG,
    "terrainSlope": main_annotation_models.LayerType.TERRAIN_SLOPE_DEG,
    # Coregistration LUTs.
    "azimuthCoregistrationShifts": main_annotation_models.LayerType.AZIMUTH_COREGISTRATION_SHIFTS_M,
    "azimuthOrbitCoregistrationShifts": main_annotation_models.LayerType.AZIMUTH_ORBIT_COREGISTRATION_SHIFTS_M,
    "rangeCoregistrationShifts": main_annotation_models.LayerType.RANGE_COREGISTRATION_SHIFTS_M,
    "rangeOrbitCoregistrationShifts": main_annotation_models.LayerType.RANGE_ORBIT_COREGISTRATION_SHIFTS_M,
    "coregistrationShiftsQuality": main_annotation_models.LayerType.COREGISTRATION_SHIFTS_QUALITY,
    "waveNumbers": main_annotation_models.LayerType.WAVENUMBERS_RAD_M,
    "flatteningPhaseScreen": main_annotation_models.LayerType.FLATTENING_PHASE_SCREEN_RAD,
    # SKP LUTs.
    "skpCalibrationPhaseScreen": main_annotation_models.LayerType.SKP_CALIBRATION_PHASE_SCREEN_RAD,
    "skpCalibrationPhaseScreenQuality": main_annotation_models.LayerType.SKP_CALIBRATION_PHASE_SCREEN_QUALITY,
}


class BIOMASSStackProductWriter:
    """Write a BIOMASS L1c product (Stack product)."""

    def __init__(
        self,
        *,
        product: BIOMASSStackProduct,
        output_dir: Path,
        source_product1_path: Path,
        source_product2_path: Path,
        file_name_aux_pps: str,
        file_name_fnf: str,
        source_product_names: tuple[str, ...],
        product_lut: dict,
        lut_axes_primary: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]],
        ql_conf: QuickLookConf,
        stack_nodata_mask: npt.NDArray[float] | None = None,
        stack_id: StackUniqueID | None = None,
        gdal_num_threads: int = 1,
        processor_name: str = "",
        processor_version: str = "",
    ) -> None:
        """
        Instantiate a writer object.

        Parameters
        ----------
        product: BIOMASSStackProduct
            The BIOMASS stack product that will be written.

        output_dir: Path
            The output directory where the product will be stored.

        source_product1_path: Path
            Source path to the current product.

        source_product2_path: Path
            Source path of the primary product.

        file_name_aux_pps: str
            Name of the AUX-PPS file.

        file_name_fnf: str
            Name of the AUX-FNF file.

        source_product_names: tuple[str, ...]
            Names of all L1a source products.

        product_lut: dict = {}
            Look-up table associated to the current product.

        lut_axes_primary: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]] [UTC], [s]
            The absolute axes of the primary LUT.

        ql_conf: QuickLookConf
            Configuration for exporting the quick-looks.

        stack_nodata_mask: npt.NDArray[float]
            A mask that specifies which pixels have no common stack data.

        stack_id: StackUniqueID | None
            The unique stack identifier.

        gdal_num_threads: int = 1
            Number of threads used to write the GeoTIFF data.

        processor_name: str = ""
            The BIOMASS stack processor name.

        processor_version: str = ""
            The BIOMASS stack processor version.

        """
        self.product = product
        self.source_product1_path = source_product1_path
        self.source_product2_path = source_product2_path
        self.source_product_names = source_product_names
        self.file_name_aux_pps = file_name_aux_pps
        self.file_name_fnf = file_name_fnf
        self.product_lut = product_lut
        self.lut_axes_primary = lut_axes_primary
        self.ql_conf = ql_conf
        self.stack_nodata_mask = stack_nodata_mask
        self.stack_id = stack_id
        self.gdal_num_threads = gdal_num_threads
        self.processor_name = processor_name
        self.processor_version = processor_version

        self.product_structure = BIOMASSStackProductStructure(
            Path(output_dir) / self.product.name,
            is_monitoring=self.product.is_monitoring,
            exists_ok=False,
        )
        self.source_product1_content = L1ProductContent.from_name(self.source_product1_path.name)
        self.source_product2_content = L1ProductContent.from_name(self.source_product2_path.name)
        self.source_mph_tai = None
        self.source_mph_ascending_node_date = None
        self.source_mph_start_time_from_ascending_node = None
        self.source_mph_completion_time_from_ascending_node = None
        self.source_mph_ref_docs = None

    def write(self):
        """Write the product to disk."""
        bps_logger.debug(
            "Writing BIOMASS Stack %s product",
            "monitoring" if self.product.is_monitoring else "standard",
        )

        # First validate the content of the product.
        self.__raise_if_inconsistent_l1c_product()

        # Initialize product structure on disk.
        self.product_structure.mkdirs(self.product.is_monitoring, parents=True)

        # Fill some values from L1a MPH.
        self.__set_values_from_source_mph()

        # Write measurement files.
        if not self.product.is_monitoring:
            bps_logger.debug("Writing measurement files")
            self._write_measurement_files()

        # Write annotation files.
        bps_logger.debug("Writing annotation files")
        self.__write_main_annotation1_file()
        self.__write_main_annotation2_file()
        self.__write_lut_annotation2_file()

        # Write orbit and attitude files.
        bps_logger.debug("Writing orbit and attitude files")
        self.__write_orbit1_file()
        self.__write_attitude1_file()
        self.__write_orbit2_file()
        self.__write_attitude2_file()

        # Write quick-look file.
        bps_logger.debug("Writing quick-look file")
        self.__write_quicklook_file()

        # Write overlay file.
        bps_logger.debug("Writing overlay file")
        write_overlay_file(
            self.product_structure.overlay_file,
            self.product_structure.quicklook_file,
            self.product.name,
            self.product.stack_footprint,
            "L1 Product Overlay ADS",
        )

        # Write schema files
        bps_logger.debug("Writing schema files")
        self.__write_schema_files()

        # Write MPH file.
        bps_logger.debug("Writing MPH file")
        self.__write_mph_file()

    def __set_values_from_source_mph(self):
        """
        Takes some fields from L1a MPH, to be written in L1c MPH.
        """
        tree = ET.parse(self.source_product2_path / self.source_product2_content.mph_file)
        root = tree.getroot()

        meta_data_property = root.find("eop:metaDataProperty", MPH_NAMESPACES)
        earth_observation_meta_data = meta_data_property.find("bio:EarthObservationMetaData", MPH_NAMESPACES)
        self.source_mph_tai = earth_observation_meta_data.find("bio:TAI-UTC", MPH_NAMESPACES).text
        self.source_mph_ref_docs = earth_observation_meta_data.findall("bio:refDoc", MPH_NAMESPACES)

        procedure = root.find("om:procedure", MPH_NAMESPACES)
        eo_equipment = procedure.find("eop:EarthObservationEquipment", MPH_NAMESPACES)
        acquisition_parameters = eo_equipment.find("eop:acquisitionParameters", MPH_NAMESPACES)
        acquisition = acquisition_parameters.find("bio:Acquisition", MPH_NAMESPACES)
        self.source_mph_ascending_node_date = acquisition.find("eop:ascendingNodeDate", MPH_NAMESPACES).text
        self.source_mph_start_time_from_ascending_node = acquisition.find(
            "eop:startTimeFromAscendingNode", MPH_NAMESPACES
        ).text
        self.source_mph_completion_time_from_ascending_node = acquisition.find(
            "eop:completionTimeFromAscendingNode", MPH_NAMESPACES
        ).text

    def __write_mph_file(self):
        """Write the MPH file."""
        # Cache product path and name,
        product_path = self.product_structure.product_path
        name = self.product.name

        # Fill MPH file.
        xml1 = ET.Element(
            "{" + MPH_NAMESPACES["bio"] + "}EarthObservation",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(1)},
        )

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}phenomenonTime")
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimePeriod",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(2)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}beginPosition")
        xml4.text = self.product.start_time.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}resultTime")
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimeInstant",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(3)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}timePosition")
        xml4.text = self.product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}validTime")
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["gml"] + "}TimePeriod",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(4)},
        )
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}beginPosition")
        xml4.text = self.product.start_time.isoformat(timespec="milliseconds")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["gml"] + "}endPosition")
        xml4.text = self.product.stop_time.isoformat(timespec="milliseconds")

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}procedure")
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["eop"] + "}EarthObservationEquipment",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(5)},
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
        xml6.text = "___" if self.product.frame_number <= 0 else str(self.product.frame_number)
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}ascendingNodeDate")
        xml6.text = self.source_mph_ascending_node_date
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}startTimeFromAscendingNode",
            attrib={"uom": "ms"},
        )
        xml6.text = self.source_mph_start_time_from_ascending_node
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}completionTimeFromAscendingNode",
            attrib={"uom": "ms"},
        )
        xml6.text = self.source_mph_completion_time_from_ascending_node
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["sar"] + "}polarisationMode")
        xml6.text = "Q"
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["sar"] + "}polarisationChannels")
        xml6.text = "HH, HV, VH, VV"  # NOTE: this is not accurate, awating input from ESA on compliance.
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
        if self.stack_id is not None:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}stackID")
            xml6.text = self.stack_id.to_id()

        xml2 = ET.SubElement(
            xml1,
            "{" + MPH_NAMESPACES["om"] + "}observedProperty",
            attrib={
                "{" + MPH_NAMESPACES["xsi"] + "}nil": "true",
                "nilReason": "inapplicable",
            },
        )

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}featureOfInterest")
        # Leave featureOfInterest field empty if there is no footprint.
        if is_null_footprint(self.product.stack_footprint):
            xml2.text = ""
        else:
            xml3 = ET.SubElement(
                xml2,
                "{" + MPH_NAMESPACES["eop"] + "}Footprint",
                attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(6)},
            )
            xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}multiExtentOf")
            xml5 = ET.SubElement(
                xml4,
                "{" + MPH_NAMESPACES["gml"] + "}MultiSurface",
                attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(7)},
            )
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["gml"] + "}surfaceMember")
            xml7 = ET.SubElement(
                xml6,
                "{" + MPH_NAMESPACES["gml"] + "}Polygon",
                attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(8)},
            )
            xml8 = ET.SubElement(xml7, "{" + MPH_NAMESPACES["gml"] + "}exterior")
            xml9 = ET.SubElement(xml8, "{" + MPH_NAMESPACES["gml"] + "}LinearRing")
            xml10 = ET.SubElement(xml9, "{" + MPH_NAMESPACES["gml"] + "}posList")
            # As per MPH convention, the footprint must be written closed.
            xml10.text = _closed_footprint(self.product.stack_footprint)
            xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}centerOf")
            xml5 = ET.SubElement(
                xml4,
                "{" + MPH_NAMESPACES["gml"] + "}Point",
                attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(9)},
            )
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["gml"] + "}pos")
            center = np.mean(self.product.stack_footprint, axis=0)
            xml6.text = f"{center[0]:.6f} {center[1]:.6f}"

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["om"] + "}result")
        xml3 = ET.SubElement(
            xml2,
            "{" + MPH_NAMESPACES["eop"] + "}EarthObservationResult",
            attrib={"{" + MPH_NAMESPACES["gml"] + "}id": name + "_" + str(10)},
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
                "{" + MPH_NAMESPACES["xlink"] + "}href": _relative_href_path(
                    self.product_structure.quicklook_file,
                    root_dir=product_path,
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

        files = []
        if not self.product.is_monitoring:
            files += [(f, None) for f in self.product_structure.measurement_files.values()]
            files += [
                (
                    self.product_structure.vrt_file,
                    self.product_structure.schema_files["l1_vrt_xsd"],
                ),
            ]
        files += [
            (
                self.product_structure.main_annotation1_file,
                self.product_structure.schema_files["l1ab_main_ann_xsd"],
            ),
            (
                self.product_structure.lut_annotation2_file,
                None,
            ),
            (
                self.product_structure.main_annotation2_file,
                self.product_structure.schema_files["l1c_main_ann_xsd"],
            ),
            (
                self.product_structure.orbit1_file,
                self.product_structure.schema_files["aux_orb_xsd"],
            ),
            (
                self.product_structure.orbit2_file,
                self.product_structure.schema_files["aux_orb_xsd"],
            ),
            (
                self.product_structure.attitude1_file,
                self.product_structure.schema_files["aux_att_xsd"],
            ),
            (
                self.product_structure.attitude2_file,
                self.product_structure.schema_files["aux_att_xsd"],
            ),
            (
                self.product_structure.quicklook_file,
                None,
            ),
            (
                self.product_structure.overlay_file,
                self.product_structure.schema_files["l1_ovr_xsd"],
            ),
        ]
        for file_path, xsd_file in files:
            if file_path is not None and file_path.exists():
                _add_file_to_mph(xml3, file_path, product_path, xsd_file)

        xml2 = ET.SubElement(xml1, "{" + MPH_NAMESPACES["eop"] + "}metaDataProperty")
        xml3 = ET.SubElement(xml2, "{" + MPH_NAMESPACES["bio"] + "}EarthObservationMetaData")
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}identifier")
        xml4.text = self.product.name
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["eop"] + "}doi")
        xml4.text = self.product.product_doi
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
        xml6.text = "other: L1c"

        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
        xml6.text = self.file_name_aux_pps
        if self.file_name_fnf is not None:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["eop"] + "}auxiliaryDataSetFileName")
            xml6.text = self.file_name_fnf
        xml6 = ET.SubElement(
            xml5,
            "{" + MPH_NAMESPACES["eop"] + "}processingMode",
            attrib={"codeSpace": "urn:esa:eop:Biomass:P-SAR:processingMode"},
        )

        xml6.text = "OPERATIONAL"
        assert self.source_product_names is not None
        for source_product_name in self.source_product_names:
            xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}sourceProduct")
            xml6.text = source_product_name
        xml6 = ET.SubElement(xml5, "{" + MPH_NAMESPACES["bio"] + "}isCoregistrationPrimary")
        xml6.text = "true" if self.product.is_coreg_primary else "false"

        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}TAI-UTC")
        xml4.text = str(self.source_mph_tai)
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}isIncomplete")
        xml4.text = "true" if self.product.frame_status != "NOMINAL" else "false"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}isPartial")
        xml4.text = "true" if self.product.frame_status == "PARTIAL" else "false"
        xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}isMerged")
        xml4.text = "true" if self.product.frame_status == "MERGED" else "false"

        if self.source_mph_ref_docs is not None:
            for ref_doc in self.source_mph_ref_docs:
                xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
                xml4.text = ref_doc.text
        else:
            xml4 = ET.SubElement(xml3, "{" + MPH_NAMESPACES["bio"] + "}refDoc")
            xml4.text = MISSING_FROM_INPUT_PRODUCT

        # Write MPH file.
        self.product_structure.mph_file.write_text(
            minidom.parseString(ET.tostring(xml1)).toprettyxml(indent="   "),
            encoding="utf-8",
        )

    def _write_measurement_files(self):
        """Write the measurement files."""

        stack_nodata_mask = (
            np.isnan(self.stack_nodata_mask) if self.stack_nodata_mask is not None else np.array([], dtype=bool)
        )

        geotiff_conf_abs = GeotiffConf(
            compression_schema=self.product.product_compression_method_abs,
            max_z_error=self.product.product_max_z_error_abs,
            nodata_value=self.product.product_nodata_value,
            block_size=self.product.sar_image_parameters.block_size,
            overview_resampling="NEAREST",
            gdal_num_threads=self.gdal_num_threads,
        )

        geotiff_conf_phase = copy.copy(geotiff_conf_abs)
        geotiff_conf_phase.compression_schema = self.product.product_compression_method_phase
        geotiff_conf_phase.max_z_error = self.product.product_max_z_error_phase

        symmetrized_scattering = len(self.product.polarization_list) == 3
        geotiff_metadata = GeotiffMetadata(
            polarizations=[pol.replace("/", "").replace("XX", "HV") for pol in self.product.polarization_list],
            creation_date=no_zulu_isoformat(self.product.creation_date, timespec="microseconds"),
            swath=self.product.swath_list[0],
            software=f"BPS-{self.processor_name}-{self.processor_version}",
            description="BIOMASS STA",
            matrix_representation=("SYMMETRIZED_SCATTERING" if symmetrized_scattering else "SCATTERING"),
        )

        abs_measurement_file = self.product_structure.measurement_files.get("abs")
        phase_measurement_file = self.product_structure.measurement_files.get("phase")
        assert abs_measurement_file is not None
        assert phase_measurement_file is not None

        write_geotiff(
            abs_measurement_file,
            data_list=[np.abs(d) for d in self.product.data_list],
            nodata_mask=stack_nodata_mask,
            ecef_gcp_list=self.product.product_primary.gcp_list,
            geotiff_metadata=geotiff_metadata,
            geotiff_conf=geotiff_conf_abs,
        )

        write_geotiff(
            phase_measurement_file,
            data_list=[np.angle(d) for d in self.product.data_list],
            nodata_mask=stack_nodata_mask,
            ecef_gcp_list=self.product.product_primary.gcp_list,
            geotiff_metadata=geotiff_metadata,
            geotiff_conf=geotiff_conf_phase,
        )

        vrt_info = VRTInfo(
            raster_xsize=self.product.data_list[0].shape[1],
            raster_ysize=self.product.data_list[0].shape[0],
            ground_corner_points_ecef=self.product.product_primary.gcp_list,
            abs_measurement_file=abs_measurement_file,
            phase_measurement_file=phase_measurement_file,
            geotiff_metadata=geotiff_metadata,
            abs_geotiff_conf=geotiff_conf_abs,
            phase_geotiff_conf=geotiff_conf_phase,
        )
        write_vrt_file(self.product_structure.vrt_file, vrt_info)

    def __write_main_annotation1_file(self):
        """Write the annotation file for product 1."""
        # Copy from primary image.
        shutil.copy(
            self.source_product1_path.joinpath(self.source_product1_content.main_annotation),
            self.product_structure.main_annotation1_file,
        )

    def __write_main_annotation2_file(self):
        """Write the annotation file for product 2."""
        # Read main annotation file from secondary image.
        main_annotation1_path = Path(
            self.source_product1_path.joinpath(self.source_product1_content.main_annotation),
        )
        main_annotation1_model: main_annotation_models_l1c.MainAnnotation = parse(
            main_annotation1_path.read_text(encoding="utf-8"),
            main_annotation_models_l1c.MainAnnotation,
        )

        # Read main annotation file from secondary image.
        main_annotation2_path = Path(
            self.source_product2_path.joinpath(self.source_product2_content.main_annotation),
        )
        main_annotation2_model: main_annotation_models_l1c.MainAnnotation = parse(
            main_annotation2_path.read_text(encoding="utf-8"),
            main_annotation_models_l1c.MainAnnotation,
        )

        # Fill coregistered image main annotation model.
        # - acquisition_information.
        acquisition_information = main_annotation2_model.acquisition_information
        acquisition_information.product_type = self.product.type
        acquisition_information.polarisation_list = main_annotation_models.PolarisationListType(
            polarisation=[translate_polarization_tag(pol, poltype="bps") for pol in self.product.polarization_list],
            count=len(self.product.polarization_list),
        )
        acquisition_information.start_time = no_zulu_isoformat(
            self.product.start_time,
            timespec="microseconds",
        )
        acquisition_information.stop_time = no_zulu_isoformat(
            self.product.stop_time,
            timespec="microseconds",
        )

        sar_image = main_annotation1_model.sar_image

        # We need to update the size of the raster in azimuth direction, in
        # case the data was cropped to a TOI.
        sar_image.number_of_lines = self.product.number_of_lines

        sar_image.first_line_azimuth_time = no_zulu_isoformat(self.product.start_time, timespec="microseconds")
        sar_image.last_line_azimuth_time = no_zulu_isoformat(self.product.stop_time, timespec="microseconds")
        sar_image.no_data_value = self.product.product_nodata_value

        # As per PFD convention, it's not specified whether the footprint must
        # be closed or open. We leave it as we inherited from the primary L1a
        # product.
        sar_image.footprint = translate_float_array_with_units_to_model(
            np.reshape(self.product.stack_footprint, -1),
            units=UomType.DEG,
        )

        instrument_parameters = main_annotation2_model.instrument_parameters

        raw_data_analysis = main_annotation2_model.raw_data_analysis

        processing_parameters = main_annotation2_model.processing_parameters

        internal_calibration = main_annotation2_model.internal_calibration

        rfi_mitigation = main_annotation2_model.rfi_mitigation

        doppler_parameters = main_annotation2_model.doppler_parameters

        radiometric_calibration = main_annotation2_model.radiometric_calibration

        polarimetric_distortion = main_annotation2_model.polarimetric_distortion

        ionosphere_correction = main_annotation2_model.ionosphere_correction

        geometry = main_annotation2_model.geometry

        quality = main_annotation2_model.quality

        # Populate the LUT layers.
        layers = [
            annot_lut_layer
            for lut_layer, annot_lut_layer in STACK_LUT_LAYER_MAP.items()
            if lut_layer in self.product_lut
        ]
        annotation_lut = main_annotation_models.LayerListType(layers, count=len(layers))

        stack_main_annotation_model = main_annotation_models_l1c.MainAnnotation(
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
            ionosphere_correction=ionosphere_correction,
            geometry=geometry,
            quality=quality,
            annotation_lut=annotation_lut,
            sta_processing_parameters=self.product.stack_processing_parameters.to_l1c_annotation(),
            sta_coregistration_parameters=self.product.stack_coregistration_parameters.to_l1c_annotation(),
            sta_in_sarparameters=self.product.stack_in_sarparameters.to_l1c_annotation(),
            sta_quality=self.product.stack_quality.to_l1c_annotation(),
        )

        # Write coregistered image main annotation file.
        self.product_structure.main_annotation2_file.write_text(
            serialize(stack_main_annotation_model),
            encoding="utf-8",
        )

    def __write_lut_annotation2_file(self):
        """Write LUT annotation file for product 2."""
        # Write LUT annotation file: Initialize Dataset.
        ncfile = NetCDF4Dataset(
            str(self.product_structure.lut_annotation2_file),
            mode="w",
            format="NETCDF4",
            clobber=True,
        )

        # Add attributes (global, common for all the LUT elements).
        start_time_iso = no_zulu_isoformat(self.product.start_time, timespec="microseconds")
        stop_time_iso = no_zulu_isoformat(self.product.stop_time, timespec="microseconds")
        reference_time_iso = no_zulu_isoformat(self.product.start_time, timespec="microseconds")

        ncfile.mission = self.product.mission
        ncfile.swath = self.product.swath_list[0]
        ncfile.productType = self.product.type
        ncfile.polarisationList = self.product.polarization_list
        ncfile.startTime = start_time_iso
        ncfile.stopTime = stop_time_iso
        ncfile.referenceAzimuthTime = reference_time_iso
        ncfile.missionPhaseID = self.product.mission_phase_id[0:3]
        ncfile.driftPhaseFlag = "False"
        ncfile.sensorMode = "Measurement"
        ncfile.globalCoverageID = np.uint16(translate_global_coverage_id(self.product.global_coverage_id))
        ncfile.majorCycleID = np.uint16(translate_major_cycle_id(self.product.major_cycle_id))
        ncfile.repeatCycleID = np.uint16(translate_repeat_cycle_id(self.product.repeat_cycle_id))
        ncfile.absoluteOrbitNumber = np.uint16(self.product.orbit_number)
        ncfile.relativeOrbitNumber = np.uint16(self.product.track_number)
        ncfile.orbitPass = self.product.orbit_direction.title()
        ncfile.platformHeading = self.product.platform_heading
        ncfile.dataTakeID = np.uint32(self.product.datatake_id)
        ncfile.frame = np.uint16(self.product.frame_number)
        ncfile.productComposition = self.product.frame_status.title()
        ncfile.noDataValue = self.product.product_nodata_value

        if len(self.product_lut) == 0:
            # Write LUT annotation file: Finalize Dataset.
            bps_logger.warning("No LUT provided")
            ncfile.close()
            return

        # The time axes of the primary product LUTs.
        self.__populate_axes(
            root=ncfile,
            names=("relativeAzimuthTime", "slantRangeTime"),
            types=(np.float64, np.float64),
            units=("s", "s"),
            axes=(
                self.lut_axes_primary[0] - self.lut_axes_primary[0][0],
                self.lut_axes_primary[1],
            ),
        )

        # Add the LUTs.
        lut_radiometry_group = ncfile.createGroup("radiometry")
        for value in ("sigma", "gamma"):
            self.__populate_lut(
                root=lut_radiometry_group,
                lut_name=f"{value}Nought",
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description="{:s} Nought".format(value.capitalize()),
                lut_unit=None,
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=True,
                warning_msg="is missing from L1a product",
            )

        lut_denoising_group = ncfile.createGroup("denoising")
        for pol in self.product.polarization_list:
            self.__populate_lut(
                root=lut_denoising_group,
                lut_name="denoising{:s}".format(pol.replace("/", "")),
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description=f"Denoising {pol}",
                lut_unit=None,
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=False,
            )

        lut_geometry_group = ncfile.createGroup("geometry")
        for coordinate, unit in [
            ("latitude", "deg"),
            ("longitude", "deg"),
            ("height", "m"),
        ]:
            self.__populate_lut(
                root=lut_geometry_group,
                lut_name=coordinate,
                lut_type=np.float64,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description=coordinate.capitalize(),
                lut_unit=unit,
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=True,
                warning_msg="is missing from L1a product",
            )
        for angle_type in ("incidence", "elevation"):
            self.__populate_lut(
                root=lut_geometry_group,
                lut_name=f"{angle_type}Angle",
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description="{:s} Angle".format(angle_type.capitalize()),
                lut_unit="deg",
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=True,
                warning_msg="is missing from L1a product or due to coregistration failure",
            )
        self.__populate_lut(
            root=lut_geometry_group,
            lut_name="terrainSlope",
            lut_type=np.float32,
            lut_axes=("relativeAzimuthTime", "slantRangeTime"),
            lut_description="Terrain Slope",
            lut_unit=None,  # As per L1_PFD (L1a LUTs).
            expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
            warn_on_missing=True,
            warning_msg="is missing from L1a product",
        )

        lut_coreg_group = ncfile.createGroup("coregistration")
        for axis in ("azimuth", "range"):
            self.__populate_lut(
                root=lut_coreg_group,
                lut_name=f"{axis}CoregistrationShifts",
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description="{:s} Coregistration Shifts".format(axis.capitalize()),
                lut_unit="m",
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=True,
                warning_msg="data is missing due to coregistration failure",
            )
            self.__populate_lut(
                root=lut_coreg_group,
                lut_name=f"{axis}OrbitCoregistrationShifts",
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description="{:s} Coregistration Shifts from Orbit".format(axis.capitalize()),
                lut_unit="m",
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=True,
                warning_msg="data is missing due to coregistration failure",
            )
        self.__populate_lut(
            root=lut_coreg_group,
            lut_name="coregistrationShiftsQuality",
            lut_type=np.float32,
            lut_axes=("relativeAzimuthTime", "slantRangeTime"),
            lut_description="Coregistration Shifts Quality",
            lut_unit=None,
            expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
            warn_on_missing=True,
            warning_msg="data  is missing due to selected coregistration mode",
        )
        self.__populate_lut(
            root=lut_coreg_group,
            lut_name="waveNumbers",
            lut_type=np.float32,
            lut_axes=("relativeAzimuthTime", "slantRangeTime"),
            lut_description="Vertical Wave Numbers (Kz)",
            lut_unit="rad/m",
            expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
            warn_on_missing=True,
            warning_msg="data is missing due to coregistration failure",
        )
        self.__populate_lut(
            root=lut_coreg_group,
            lut_name="flatteningPhaseScreen",
            lut_type=np.float32,
            lut_axes=("relativeAzimuthTime", "slantRangeTime"),
            lut_description="Flattening Phases Screen (DSI)",
            lut_unit="rad",
            expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
            warn_on_missing=True,
            warning_msg="data is missing due to coregistration failure",
        )

        if "skpCalibrationPhaseScreen" in self.product_lut:
            lut_skp_group = ncfile.createGroup("skpPhaseCalibration")
            self.__populate_lut(
                root=lut_skp_group,
                lut_name="skpCalibrationPhaseScreen",
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description="SKP Calibration Phases",
                lut_unit="rad",
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=False,
            )
            self.__populate_lut(
                root=lut_skp_group,
                lut_name="skpCalibrationPhaseScreenQuality",
                lut_type=np.float32,
                lut_axes=("relativeAzimuthTime", "slantRangeTime"),
                lut_description="SKP Estimation Quality",
                lut_unit=None,
                expected_shape=tuple(ax.size for ax in self.lut_axes_primary),
                warn_on_missing=True,
                warning_msg="is missing but SKP LUTs are expected",
            )
        elif self.product.stack_processing_parameters.skp_phase_calibration_flag:
            bps_logger.warning("skpPhaseCalibration LUTs are missing")

        # Write LUT annotation file: Finalize Dataset.
        ncfile.close()

    def __write_orbit1_file(self):
        """Write orbit file for product 1."""
        # Copy from primary image.
        shutil.copy(
            self.source_product1_path.joinpath(self.source_product1_content.orbit),
            self.product_structure.orbit1_file,
        )

    def __write_attitude1_file(self):
        """Write attitude file for product 1."""
        # Copy from primary image.
        shutil.copy(
            self.source_product1_path.joinpath(self.source_product1_content.attitude),
            self.product_structure.attitude1_file,
        )

    def __write_orbit2_file(self):
        """Write orbit file for product 2."""
        # Copy from primary image.
        shutil.copy(
            self.source_product2_path.joinpath(self.source_product2_content.orbit),
            self.product_structure.orbit2_file,
        )

    def __write_attitude2_file(self):
        """Write attitude file for product 2."""
        # Copy from primary image.
        shutil.copy(
            self.source_product2_path.joinpath(self.source_product2_content.attitude),
            self.product_structure.attitude2_file,
        )

    def __write_quicklook_file(self):
        """Write the quick-looks."""
        # Possibly, apply the no-data mask of the stack.
        data = self.product.data_list
        if self.stack_nodata_mask is not None:
            stack_nodata_mask = np.isfinite(self.stack_nodata_mask)
            data = [d * stack_nodata_mask for d in data]

        write_quicklook_to_file(
            compute_quicklook_from_pol_data(
                dict(zip(self.product.polarization_list, data)),
                self.ql_conf,
            ),
            self.product_structure.quicklook_file,
        )

    def __write_schema_files(self):
        """Write the schemas (copy)."""
        copy_biomass_xsd_files(
            self.product_structure.xsd_schema_dir,
            [f.name for f in self.product_structure.schema_files.values()],
        )

    def __populate_axes(
        self,
        *,
        root: NetCDF4Dataset,
        names: tuple[str, str],
        types: Generic,
        units: tuple[str, str],
        axes: tuple[npt.NDArray[float], npt.NDArray[float]],
    ):
        """Populate the dimension."""
        for name, tp, unit, axis in zip(names, types, units, axes):
            root.createDimension(name, len(axis))
            var = root.createVariable(
                name,
                tp,
                (name,),
                compression="zlib",
                complevel=ZLIB_L1C_COMPLEVEL,
            )
            var.units = unit
            var[:] = axis

    def __populate_lut(
        self,
        *,
        root: NetCDF4Dataset | NetCDF4Group,
        lut_name: str,
        lut_type: Generic,
        lut_axes: tuple[str, str],
        lut_description: str,
        lut_unit: str | None,
        expected_shape: tuple[int, int],
        warn_on_missing: bool,
        warning_msg: str | None = None,
    ):
        """Utility to populate the LUT and, optionally, report a missing one."""
        # NOTE: We use this check to make sure that the user updates the exported.
        # LUTs from the reader.
        if lut_name not in LUT_LAYERS:
            raise InvalidBIOMASSStackProductError(f"'{lut_name}' must be added to the LUT_LAYERS variable")

        lut_var = root.createVariable(
            lut_name,
            lut_type,
            lut_axes,
            compression="zlib",
            complevel=ZLIB_L1C_COMPLEVEL,
            fill_value=self.product.product_nodata_value,
        )
        lut_var.description = lut_description
        if lut_unit is not None:
            lut_var.units = lut_unit

        data = np.full(expected_shape, np.nan)
        if lut_name in self.product_lut:
            data = self.product_lut[lut_name].copy()
        elif warn_on_missing:
            assert warning_msg is not None, "missing a valid warning message"
            bps_logger.warning("LUT %s %s", lut_name, warning_msg)

        if expected_shape != data.shape:
            raise InvalidBIOMASSStackProductError(
                f"{lut_name} has invalid shape (got={data.shape}, expected={expected_shape})"
            )

        data[np.isnan(data)] = self.product.product_nodata_value
        lut_var[:, :] = data

    def __raise_if_inconsistent_l1c_product(self):
        """Run a minimal validation on the L1c product."""
        # Check that the product has at least 1 channel.
        if self.product.channels <= 0:
            raise InvalidBIOMASSStackProductError("Product has no channels")

        # Check that number of channels is consistent with polarizations.
        if self.product.channels != len(self.product.polarization_list):
            raise InvalidBIOMASSStackProductError(
                "Channels and polarizations mismatch: num channels={}, num pols={}".format(
                    self.product.channels, len(self.product.polarization_list)
                )
            )

        # Check that number of channels is consistent with the number of rasters.
        if self.product.channels != len(self.product.raster_info_list):
            raise InvalidBIOMASSStackProductError(
                "Channels and raster list mismatch: num channels={}, num rasters={}".format(
                    self.product.channels, len(self.product.raster_info_list)
                )
            )

        # Check that number of channels is consistent with the data list.
        if self.product.channels != len(self.product.data_list) and not self.product.is_monitoring:
            raise InvalidBIOMASSStackProductError(
                "Channels and data list mismatch: num channels: {}, num images: {}".format(
                    self.product.channels, len(self.product.data_list)
                )
            )

        # Check that all data have same shape and are consistent with the rasters.
        data_shapes = [d.shape for d in self.product.data_list]
        if any(sh != self.product.data_list[0].shape for sh in data_shapes):
            raise InvalidBIOMASSStackProductError(
                "Not all frames have same shape: {}".format(dict(zip(self.product.polarization_list, data_shapes)))
            )

        raster_shapes = [(r.lines, r.samples) for r in self.product.raster_info_list]
        if any(dsh != rsh for dsh, rsh in zip(data_shapes, raster_shapes)):
            raise InvalidBIOMASSStackProductError(
                "Raster info and data have mismatching shapes: rasters={}, data={}".format(
                    dict(zip(self.product.polarization_list, raster_shapes)),
                    dict(zip(self.product.polarization_list, data_shapes)),
                )
            )

        # Check that the start/stop time makes sense.
        raster_start_times = [r.lines_start for r in self.product.raster_info_list]
        if any(rt != self.product.start_time for rt in raster_start_times):
            raise InvalidBIOMASSStackProductError(
                "Annotated start time and raster are inconsistent: start={}, rasters={}".format(
                    self.product.start_time, raster_start_times
                )
            )

        raster_stop_times = [_stop_time(r) for r in self.product.raster_info_list]
        if any(rt != self.product.stop_time for rt in raster_stop_times):
            raise InvalidBIOMASSStackProductError(
                "Annotated stop time and raster info are inconsistent: stop={}, rasters={}".format(
                    self.product.stop_time, raster_stop_times
                )
            )

        # Check the sampling steps.
        eps = np.power(10.0, -np.finfo(np.float64).precision)

        azm_sampling_steps = [r.lines_step for r in self.product.raster_info_list]
        if not all(np.isclose(s, self.product.az_time_interval, atol=eps) for s in azm_sampling_steps):
            raise InvalidBIOMASSStackProductError(
                "Annotated azimuth sampling step and rasters are inconsistent: rasters={}, annot={}".format(
                    azm_sampling_steps, self.product.az_time_interval
                )
            )

        rng_sampling_steps = [r.samples_step for r in self.product.raster_info_list]
        if not all(np.isclose(s, self.product.rg_time_interval, atol=eps) for s in rng_sampling_steps):
            raise InvalidBIOMASSStackProductError(
                "Annotated range sampling step and rasters are inconsistent: rasters={}, annot={}".format(
                    rng_sampling_steps, self.product.rg_time_interval
                )
            )

        # Check consistency of number of lines/samples and raster info.
        num_azimuths = [r.lines for r in self.product.raster_info_list]
        if any(naz != self.product.number_of_lines for naz in num_azimuths):
            raise InvalidBIOMASSStackProductError(
                "Annotated num of azimuths and rasters mismatch: rasters={}, annot={}".format(
                    num_azimuths, self.product.number_of_lines
                )
            )

        num_ranges = [r.samples for r in self.product.raster_info_list]
        if any(nrg != self.product.number_of_samples for nrg in num_ranges):
            raise InvalidBIOMASSStackProductError(
                "Annotated num of ranges and rasters mismatch: rasters={}, annot={}".format(
                    num_ranges, self.product.number_of_samples
                )
            )


def _add_file_to_mph(
    base_element: ET.Element,
    file_path: Path,
    product_path: Path,
    rds_file: Path | None,
):
    """Add product file to mph list"""
    file_size = file_path.stat().st_size

    product_elem = ET.SubElement(base_element, "{" + MPH_NAMESPACES["eop"] + "}product")
    product_information_elem = ET.SubElement(product_elem, "{" + MPH_NAMESPACES["bio"] + "}ProductInformation")
    file_name_elem = ET.SubElement(product_information_elem, "{" + MPH_NAMESPACES["eop"] + "}fileName")
    service_reference_elem = ET.SubElement(
        file_name_elem,
        "{" + MPH_NAMESPACES["ows"] + "}ServiceReference",
        attrib={"{" + MPH_NAMESPACES["xlink"] + "}href": _relative_href_path(file_path, root_dir=product_path)},
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
        rds_elem.text = _relative_href_path(rds_file, root_dir=product_path)


def _relative_href_path(path: Path, *, root_dir: Path) -> str:
    """Create a relative path ./relative/path."""
    return "./{}".format(path.relative_to(root_dir))


def _closed_footprint(footprint: list[list[float]]) -> str:
    """Serialize the footprint."""
    if len(footprint) == 4:
        return serialize_footprint([*footprint, footprint[0]])
    if len(footprint) == 5:
        return serialize_footprint(footprint)
    raise InvalidBIOMASSStackProductError("footprint is invalid ( points)")


def _stop_time(raster_info: RasterInfo) -> PreciseDateTime:
    """The stop time of a raster."""
    return round_precise_datetime(
        raster_info.lines_start + (raster_info.lines - 1) * raster_info.lines_step,
        timespec="microseconds",
    )
