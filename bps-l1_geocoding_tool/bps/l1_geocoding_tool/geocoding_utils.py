# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Geocoding utils"""

import xml.etree.ElementTree as ET
from pathlib import Path
from subprocess import CalledProcessError, Popen, run
from xml.dom import minidom

from arepyextras.copernicus_dem_extractor import extract_data_to_pf
from arepyextras.runner import Environment
from arepytools.io import (
    iter_channels,
    open_product_folder,
    read_raster_with_raster_info,
)
from bps.l1_geocoding_tool import DEM_EXTRACTOR_MARGIN, GEOCODER_EXE_NAME
from bps.transcoder.sarproduct.sarproduct import SARProduct


def run_dem_extractor(product_footprint: list[list[float]], dem_path: Path, extracted_dem_path: Path):
    """Run DEM Extractor"""
    dem_index_file_path = dem_path / "copernicus" / "COP-DEM_GLO-90-DGED-2021_1" / "demIndex.xml"

    lat = [f[0] for f in product_footprint]
    lon = [f[1] for f in product_footprint]
    roi = (
        min(lon) - DEM_EXTRACTOR_MARGIN,
        max(lon) + DEM_EXTRACTOR_MARGIN,
        min(lat) - DEM_EXTRACTOR_MARGIN,
        max(lat) + DEM_EXTRACTOR_MARGIN,
    )

    path_to_egm2008_tif = dem_path / "copernicus" / "COP-DEM_GLO-90-DGED-2021_1" / "egm2008-2.5.tif"

    extract_data_to_pf(
        dem_index_file_path=dem_index_file_path,
        roi=roi,
        output_path=extracted_dem_path,
        path_to_egm2008_tif=path_to_egm2008_tif,
    )


def create_input_file(
    geocoder_inputfile_path: Path,
    input_product_path: Path,
    dem_product_path: Path,
    geocoder_conffile_path: Path,
    output_product_path: Path,
):
    """Generate input file for Generic Geocoder"""
    xml1 = ET.Element("AresysXmlInput")

    xml2 = ET.SubElement(xml1, "Step", attrib={"Number": "1", "Total": "1"})

    xml3 = ET.SubElement(xml2, "Geocoder")

    xml4 = ET.SubElement(xml3, "InputProduct")
    xml4.text = str(input_product_path)
    xml4 = ET.SubElement(xml3, "DEMProduct")
    xml4.text = str(dem_product_path)
    xml4 = ET.SubElement(xml3, "ConfigurationFileName")
    xml4.text = str(geocoder_conffile_path)
    xml4 = ET.SubElement(xml3, "OutputGeocodedProduct")
    xml4.text = str(output_product_path)
    xml4 = ET.SubElement(xml3, "OutputProjection")

    xml5 = ET.SubElement(xml4, "ProjectionKind")
    xml5.text = "LATLON"

    xmlstr = minidom.parseString(ET.tostring(xml1)).toprettyxml(indent="   ")
    with open(file=geocoder_inputfile_path, mode="w", encoding="utf-8") as f:
        f.write(xmlstr)


def create_configuration_file(geocoder_conffile_path: Path, lat_lon_step: str | None):
    """Generate configuration file for Generic Geocoder"""
    xml1 = ET.Element("AresysXmlDoc")

    xml2 = ET.SubElement(xml1, "NumberOfChannels")
    xml2.text = "1"
    xml2 = ET.SubElement(xml1, "VersionNumber")
    xml2.text = "1.0"
    xml2 = ET.SubElement(xml1, "Description")
    xml2.text = "Configuration file"
    xml2 = ET.SubElement(xml1, "Channel", attrib={"Number": "1", "Total": "1"})

    xml3 = ET.SubElement(xml2, "GeocodingConf")

    xml4 = ET.SubElement(xml3, "Steps")

    xml5 = ET.SubElement(xml4, "Lat")
    xml5.text = lat_lon_step
    xml5 = ET.SubElement(xml4, "Long")
    xml5.text = lat_lon_step

    xmlstr = minidom.parseString(ET.tostring(xml1)).toprettyxml(indent="   ")
    with open(file=geocoder_conffile_path, mode="w", encoding="utf-8") as f:
        f.write(xmlstr)


def read_geocoded(product_path: Path) -> SARProduct:
    """Read geocoded product"""
    pf = open_product_folder(product_path)
    first_channel = pf.get_channels_list()[0]

    product = SARProduct()
    product.name = product_path.name
    product.channels = len(pf.get_channels_list())

    for channel, ch in iter_channels(pf):
        data_file = pf.get_channel_data(channel)
        data = read_raster_with_raster_info(data_file, ch.get_raster_info())
        product.data_list.append(data)

        product.raster_info_list.append(ch.get_raster_info())
        product.burst_info_list.append(ch.get_burst_info())
        if channel == first_channel:
            product.dataset_info.append(ch.get_dataset_info())
        product.swath_info_list.append(ch.get_swath_info())
        product.sampling_constants_list.append(ch.get_sampling_constants())
        product.acquisition_timeline_list.append(ch.get_acquisition_time_line())
        product.data_statistics_list.append(ch.get_data_statistics())
        if channel == first_channel:
            product.general_sar_orbit.append(ch.get_state_vectors())
        product.dc_vector_list.append(ch.get_doppler_centroid())
        product.dc_eff_vector_list.append(ch.get_doppler_centroid())
        product.dr_vector_list.append(ch.get_doppler_rate())
        product.slant_to_ground_list.append(ch.get_slant_to_ground())
        product.ground_to_slant_list.append(ch.get_ground_to_slant())
        if channel == first_channel:
            product.attitude_info.append(ch.get_attitude_info())
        product.pulse_list.append(ch.get_pulse())

    # Set remaining product attributes
    product.mission = product.dataset_info[0].sensor_name
    product.acquisition_mode = product.dataset_info[0].acquisition_mode
    product.type = "GEC"

    for si in product.swath_info_list:
        product.swath_list.append(si.swath)
        product.polarization_list.append(si.polarization.value)

    product.orbit_number = product.general_sar_orbit[0].orbit_number
    product.orbit_direction = product.general_sar_orbit[0].orbit_direction.value

    return product


def run_tool(geocoder_inputfile_path: Path):
    """Run Generic Geocoder executable"""
    try:
        _ = run(
            [
                GEOCODER_EXE_NAME,
                geocoder_inputfile_path,
                "1",
            ],
            capture_output=False,
            text=True,
            check=True,
        )
        return True
    except CalledProcessError:
        return False


def run_tool_with_env(
    geocoder_inputfile_path: Path,
    env: Environment,
):
    """Run Generic Geocoder executable within an environment"""
    try:
        popen_args, popen_kwargs = env.build_run_command_arguments(
            GEOCODER_EXE_NAME, *[str(geocoder_inputfile_path.absolute()), 1]
        )
    except ValueError as exc:
        raise RuntimeError(f"Command '{GEOCODER_EXE_NAME}' not found") from exc

    with Popen(popen_args, **popen_kwargs) as process:
        process.wait()
        returncode = process.returncode

    if returncode != 0:
        command_line = " ".join(popen_args)
        raise RuntimeError(f"Command failed with code {returncode}:\n" + f"'{command_line}'\n")


def geocode_product(
    product: SARProduct,
    dem_path: Path,
    lat_lon_step: str | None,
    output_path: Path,
    env: Environment,
):
    """Run Generic Geocoder"""
    # Set useful variables
    input_product_path = Path(output_path, "iL1")
    output_dem_product_path = Path(output_path, "iDEM")
    geocoder_inputfile_path = Path(output_path, "geocoder_input_file.xml")
    geocoder_conffile_path = Path(output_path, "geocoder_conf_file.xml")
    output_geocoded_product_path = Path(output_path, "iL1Geocoded")

    # Write product to disk
    product.write(input_product_path)

    # Step 1/2: DEM Extraction
    product_footprint = product.footprint
    if not output_dem_product_path.exists():
        run_dem_extractor(product_footprint, dem_path, output_dem_product_path)
    else:
        raise RuntimeError("DEM database not found")

    # Step 2/2: Data Geocoding
    # - Create input file
    create_input_file(
        geocoder_inputfile_path,
        input_product_path,
        output_dem_product_path,
        geocoder_conffile_path,
        output_geocoded_product_path,
    )

    # - Create configuration file
    create_configuration_file(geocoder_conffile_path, lat_lon_step)

    # - Run Generic Geocoder
    # run_tool(geocoder_inputfile_path)
    run_tool_with_env(geocoder_inputfile_path, env)

    # Read geocoded product from disk
    geocoded_product = read_geocoded(output_geocoded_product_path)
    geocoded_product.name = product.name

    # Delete input file, configuration files and Generic Geocoder outputs
    if input_product_path.exists():
        open_product_folder(input_product_path).delete()
    if output_dem_product_path.exists():
        for file in output_dem_product_path.glob("*cutovr*"):
            if file.is_file():
                file.unlink()
        open_product_folder(output_dem_product_path).delete()
    if geocoder_inputfile_path.exists():
        geocoder_inputfile_path.unlink(missing_ok=False)
    if geocoder_conffile_path.exists():
        geocoder_conffile_path.unlink(missing_ok=False)
    if output_geocoded_product_path.exists():
        open_product_folder(output_geocoded_product_path).delete()

    return geocoded_product
