# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.transcoder.sarproduct import mph
from bps.transcoder.sarproduct.biomass_l0product import BIOMASSL0Product
from bps.transcoder.sarproduct.l0.product_content import L0ProductContent


def _retrieve_xml_file(product_path: Path, content: L0ProductContent):
    mph_file = product_path.joinpath(content.mph_file)
    # Support to legacy datasets
    # where products name and internal file naming don't match
    if not mph_file.exists():
        xml_files = list(Path(product_path).glob("*.xml"))
        if not xml_files:
            raise RuntimeError(f"Cannot find mph file in {Path(product_path)}")
        mph_file = xml_files[0]

    return mph_file


@dataclass
class MPHL0:
    phenomenon_time: tuple[PreciseDateTime, PreciseDateTime] | None
    valid_time: tuple[PreciseDateTime, PreciseDateTime] | None
    acquisition: mph.MPHAcquisition
    product_information: mph.MPHProductInformation
    metadata: mph.MPHMetadata


def read_l0_mph(mph_file: Path) -> MPHL0:
    tree = ET.parse(mph_file)
    root = tree.getroot()

    return MPHL0(
        phenomenon_time=mph.get_phenomenon_time(root),
        valid_time=mph.get_valid_time(root),
        acquisition=mph.get_acquisition(root),
        product_information=mph.get_product_information(root),
        metadata=mph.get_metadata(root),
    )


def read_l0_product(product_path: Path | str) -> BIOMASSL0Product:
    product_path = Path(product_path)
    if not product_path.exists():
        raise RuntimeError(f"Cannot read product {product_path}: product does not exist")

    content = L0ProductContent.from_name(product_path.name)
    mph_file = _retrieve_xml_file(product_path, content)

    bps_logger.debug("Reading BIOMASS L0 product..")
    bps_logger.debug("..MPH file")
    mph_l0 = read_l0_mph(mph_file)

    product = BIOMASSL0Product()
    product.name = product_path.name

    phenomenon_time_length = None
    if mph_l0.phenomenon_time:
        product.sensing_start_time, product.sensing_stop_time = mph_l0.phenomenon_time
        phenomenon_time_length = product.sensing_stop_time - product.sensing_start_time

    valid_time_length = None
    if product.start_time is None or product.stop_time is None:
        if mph_l0.valid_time:
            product.start_time, product.stop_time = mph_l0.valid_time
            valid_time_length = product.stop_time - product.start_time

    product.orbit_number = mph_l0.acquisition.orbit_number
    product.orbit_direction = mph_l0.acquisition.orbit_direction
    product.track_number = mph_l0.acquisition.track_number
    product.slice_number = mph_l0.acquisition.slice_number
    product.anx_time = mph_l0.acquisition.anx_time
    product.mission_phase_id = mph_l0.acquisition.mission_phase_id
    product.instrument_configuration_id = mph_l0.acquisition.instrument_configuration_id
    product.datatake_id = mph_l0.acquisition.datatake_id
    product.orbit_drift_flag = mph_l0.acquisition.orbit_drift_flag
    product.global_coverage_id = mph_l0.acquisition.global_coverage_id
    product.major_cycle_id = mph_l0.acquisition.major_cycle_id
    product.repeat_cycle_id = mph_l0.acquisition.repeat_cycle_id

    product.baseline_id = mph_l0.product_information.baseline_id
    product.file_sizes = mph_l0.product_information.file_sizes

    if mph_l0.metadata.is_partial is not None and mph_l0.metadata.is_partial:
        product.slice_status = "PARTIAL"
    if mph_l0.metadata.is_merged is not None and mph_l0.metadata.is_merged:
        product.slice_status = "MERGED"

    default_length = 100.0
    length = phenomenon_time_length or valid_time_length or default_length
    product.bit_rate = max(product.file_sizes.values()) / length * 8 / 1e6  # Mbit/s

    bps_logger.debug("..done")

    return product
