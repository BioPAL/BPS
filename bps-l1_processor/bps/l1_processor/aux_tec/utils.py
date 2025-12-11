# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to retrieve ionex file from AUX_TEC product(s)
--------------------------------------------------------
"""

import gzip
from pathlib import Path

import unlzw3
from bps.common import bps_logger
from bps.common.common import retrieve_aux_product_data_single_content
from bps.l1_processor.aux_tec.ionex_files_merger import merge_ionex_files
from bps.l1_processor.core.interface import IonexFiles


def decompress_ionex_if_needed(ionex_file: Path, decompressed_ionex_file: Path) -> Path:
    """Decompress ionex file to provided destination if needed, returns the decompressed path"""
    if ionex_file.suffix.lower() not in (".gz", ".z"):
        bps_logger.debug(f"Input ionex file: {ionex_file} is not compressed")
        return ionex_file

    if ionex_file.suffix.lower() == ".gz":
        bps_logger.debug(f"Input ionex file: {ionex_file} is gzip compressed")
        with gzip.open(ionex_file, "rb") as file:
            decompressed_ionex_file.write_bytes(file.read())
    else:
        assert ionex_file.suffix.lower() == ".z"
        bps_logger.debug(f"Input ionex file: {ionex_file} is z compressed")
        decompressed_ionex_file.write_bytes(unlzw3.unlzw(ionex_file.read_bytes()))

    bps_logger.debug(f"Compressed ionex file extracted to: {decompressed_ionex_file}")
    return decompressed_ionex_file


def retrieve_single_decompressed_ionex_file(aux_tec_products: list[Path], ionex_layout: IonexFiles) -> Path:
    """From one or two AUX_TEC product, retrieve a single ionex text file. Possibly decompressing."""
    if len(aux_tec_products) not in (1, 2):
        raise RuntimeError(f"Unexpected number of AUX TEC products provided: {len(aux_tec_products)} != 1 or 2")
    aux_tec_from_same_day = False
    if len(aux_tec_products) == 2:
        if aux_tec_products[0].name[15:46] == aux_tec_products[1].name[15:46]:
            bps_logger.warning(
                "Provided AUX TEC products have same validity. First one is used, second one is discarded"
            )
            aux_tec_from_same_day = True

    ionex_files = [retrieve_aux_product_data_single_content(product) for product in aux_tec_products]

    decompressed_ionex_files = [
        decompress_ionex_if_needed(file, decompressed_map)
        for file, decompressed_map in zip(
            ionex_files,
            (
                ionex_layout.decompressed_1,
                ionex_layout.decompressed_2,
            ),
        )
    ]

    if len(decompressed_ionex_files) == 2 and not aux_tec_from_same_day:
        ionex_files_pair = decompressed_ionex_files[0], decompressed_ionex_files[1]
        merge_ionex_files(ionex_files_pair, ionex_layout.ionex_file)
        return ionex_layout.ionex_file
    else:
        return decompressed_ionex_files[0]
