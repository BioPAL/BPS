# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Filter bank writing utils
-------------------------
"""

from pathlib import Path

import numpy as np
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    metadata,
    write_metadata,
    write_raster_with_raster_info,
)


def fill_filter_bank_metadata(
    lines: int, samples: int, raster_file_name: str, sampling_step: float
) -> metadata.MetaData:
    """Fill minimal metadata filter information"""
    channel_metadata = create_new_metadata()
    raster_info = metadata.RasterInfo(
        lines=lines,
        samples=samples,
        celltype=metadata.ECellType.float64,
        filename=raster_file_name,
    )
    raster_info.set_lines_axis(lines_start=0, lines_start_unit="", lines_step=1, lines_step_unit="")
    raster_info.set_samples_axis(
        samples_start=0,
        samples_start_unit="",
        samples_step=sampling_step,
        samples_step_unit="",
    )
    channel_metadata.insert_element(raster_info)
    return channel_metadata


def write_filter_bank_product(filter_bank: np.ndarray, product_path: Path, sampling_step: float):
    """Write filter bank product"""
    product = create_product_folder(product_path, overwrite_ok=True)

    channel_index = 1
    raster_file = product.get_channel_data(channel_index)
    metadata_file = product.get_channel_metadata(channel_index)

    lines, samples = filter_bank.shape

    meta = fill_filter_bank_metadata(lines, samples, raster_file.name, sampling_step)

    write_metadata(meta, metadata_file)

    write_raster_with_raster_info(
        raster_file=raster_file,
        data=filter_bank,
        raster_info=meta.get_raster_info(),
    )
