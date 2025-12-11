# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""A generic product"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arepytools.io import (
    iter_channels,
    open_product_folder,
    read_raster_with_raster_info,
)
from bps.common import bps_logger


@dataclass
class GenericProduct:
    """A generic product"""

    name: str
    channels: int
    data_list: list
    samples_axis_list: list
    lines_axis_list: list

    @classmethod
    def read_from_product_path(
        cls,
        product_path: Path,
        samples_dwns_factor: int = 1,
        lines_dwns_factor: int = 1,
    ) -> GenericProduct:
        """Read a generic product"""
        bps_logger.debug(f"Reading product: {product_path}...")

        pf = open_product_folder(product_path)

        data_list = []
        samples_axis_list = []
        lines_axis_list = []
        for channel_idx, metadata in iter_channels(pf):
            # Set product data
            raster_file = pf.get_channel_data(channel_idx)
            ri = metadata.get_raster_info()
            data = read_raster_with_raster_info(raster_file=raster_file, raster_info=ri)
            if samples_dwns_factor > 1 or lines_dwns_factor > 1:
                data = data[::lines_dwns_factor, ::samples_dwns_factor]
            data_list.append(data)

            # Set product metadata
            samples_axis = np.arange(ri.samples) * ri.samples_step + ri.samples_start
            lines_axis = np.arange(ri.lines) * ri.lines_step + ri.lines_start
            if samples_dwns_factor > 1 or lines_dwns_factor > 1:
                samples_axis = samples_axis[::samples_dwns_factor]
                lines_axis = lines_axis[::lines_dwns_factor]
            samples_axis_list.append(samples_axis)
            lines_axis_list.append(lines_axis)

        bps_logger.debug("..done")
        return GenericProduct(
            name=product_path.name,
            channels=len(pf.get_channels_list()),
            data_list=data_list,
            samples_axis_list=samples_axis_list,
            lines_axis_list=lines_axis_list,
        )
