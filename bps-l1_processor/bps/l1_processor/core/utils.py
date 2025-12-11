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


def update_bps_l1_pre_processor_report_file(report_file: Path, chirp_product: Path):
    """Update pre-processor report file with chirp replica parameters"""
    try:
        chirp_parameters_model: models.L1PreProcessorAnnotations = parse(
            report_file.read_text(), models.L1PreProcessorAnnotations
        )
    except ParsingError as exc:
        raise InvalidL1PreProcAnnotations from exc

    chirp_parameters = analyse_chirp_replica(chirp_product)

    chirp_parameters_model.chirp_replica_parameters = [
        models.L1PreProcessorAnnotations.ChirpReplicaParameters(
            float(pars.bandwidth),
            float(pars.pslr),
            float(pars.islr),
            float(pars.location_error),
            pars.validity_flag,
            models.PolarizationType(pars.polarization),
        )
        for pars in chirp_parameters
    ]

    chirp_parameters_text = serialize(chirp_parameters_model)
    report_file.write_text(chirp_parameters_text, encoding="utf-8")
