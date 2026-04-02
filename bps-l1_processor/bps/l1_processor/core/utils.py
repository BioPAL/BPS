# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities functions
-------------------
"""

import re
import shutil
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

from arepytools.io import open_product_folder
from bps.common.io.parsing import ParsingError, parse, serialize
from bps.l1_processor.core.chirp_replica_utils import analyse_chirp_replica
from bps.l1_processor.core.iobpr_utils import analyse_raw_product
from bps.transcoder.io.biomass_l1_preproc_annotations import models
from bps.transcoder.io.preprocessor_report import InvalidL1PreProcAnnotations

_LEVEL0_RXH_DATA_REGEX = r"_raw__0s.*[0-9_]_rxh\.dat$"
_LEVEL0_RXV_DATA_REGEX = r"_raw__0s.*[0-9_]_rxv\.dat$"


def _retrieve_data_file(product: Path, pattern: str) -> Path | None:
    for file in product.iterdir():
        if re.search(pattern, file.name):
            return file
    return None


retrieve_rxh_data_file = partial(_retrieve_data_file, pattern=_LEVEL0_RXH_DATA_REGEX)
retrieve_rxv_data_file = partial(_retrieve_data_file, pattern=_LEVEL0_RXV_DATA_REGEX)


def update_bps_l1_core_processor_status_file(status_file: Path, curr_param_file: Path):
    """Update the status file with a new position for the processing parameter file"""
    assert status_file.exists()

    root = ET.parse(status_file).getroot()

    resource_node = root.find(".//Resource")
    if resource_node is None:
        return

    previous_param_file = resource_node.text
    if str(previous_param_file) == str(curr_param_file):
        return

    status_file.write_text(status_file.read_text().replace(str(previous_param_file), str(curr_param_file)))


def save_reference_extracted_raw_annotation(extracted_raw: Path, reference_annotation: Path):
    """Copy a reference metadata file from the extracted raw"""
    raw = open_product_folder(extracted_raw)
    reference_metadata = raw.get_channel_metadata(1)
    shutil.copy2(reference_metadata, reference_annotation)


def update_bps_l1_pre_processor_report_file_with_chirp_pars(report_file: Path, chirp_product: Path):
    """Update pre-processor report file with chirp replica parameters"""
    try:
        report_model: models.L1PreProcessorAnnotations = parse(
            report_file.read_text(), models.L1PreProcessorAnnotations
        )
    except ParsingError as exc:
        raise InvalidL1PreProcAnnotations from exc

    chirp_parameters = analyse_chirp_replica(chirp_product)

    report_model.chirp_replica_parameters = [
        models.L1PreProcessorAnnotations.ChirpReplicaParameters(
            bandwidth=float(pars.bandwidth),
            pslr=float(pars.pslr),
            islr=float(pars.islr),
            location_error=float(pars.location_error),
            validity_flag=pars.validity_flag,
            polarization=models.PolarizationType(pars.polarization),
        )
        for pars in chirp_parameters
    ]

    report_text = serialize(report_model)
    report_file.write_text(report_text, encoding="utf-8")


def update_bps_l1_pre_processor_report_file_with_iobpr(report_file: Path, raw_product: Path, obf_file: Path):
    """Update pre-processor report file with in/out-of-band power ratio measurements"""
    try:
        report_model: models.L1PreProcessorAnnotations = parse(
            report_file.read_text(), models.L1PreProcessorAnnotations
        )
    except ParsingError as exc:
        raise InvalidL1PreProcAnnotations from exc

    iobpr_measurements = analyse_raw_product(raw_product, obf_file)

    report_model.in_out_band_power_ratio_data = [
        models.L1PreProcessorAnnotations.InOutBandPowerRatioData(
            reference_time=key,
            power_ratios=models.L1PreProcessorAnnotations.InOutBandPowerRatioData.PowerRatios(
                power_ratio=[
                    models.PowerRatioType(value=float(val[1]), polarization=models.PolarizationType(val[0]))
                    for val in value
                ]
            ),
        )
        for key, value in iobpr_measurements.items()
    ]

    report_text = serialize(report_model)
    report_file.write_text(report_text, encoding="utf-8")
