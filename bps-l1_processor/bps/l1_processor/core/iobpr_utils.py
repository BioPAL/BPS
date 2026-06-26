# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
In/out-of-band power ratio utilities functions
----------------------------------------------
"""

from pathlib import Path

import numpy as np
from arepytools.io import open_product_folder, read_metadata, read_raster_with_raster_info
from netCDF4 import Dataset


def read_on_board_filters_file(obf_file_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Read on-board filters characterization file"""
    with Dataset(obf_file_path, "r") as root:
        filter_spectrum_axis = np.array(root.variables["rangeFreq"][:])
        on_board_filters_group = root.groups["onBoardFilters"]
        filter_spectrum_dict = {
            "H/H": np.array(on_board_filters_group.variables["onBoardFilterHH"][:]),
            "H/V": np.array(on_board_filters_group.variables["onBoardFilterHV"][:]),
            "V/H": np.array(on_board_filters_group.variables["onBoardFilterVH"][:]),
            "V/V": np.array(on_board_filters_group.variables["onBoardFilterVV"][:]),
        }
    return filter_spectrum_dict, filter_spectrum_axis


def analyse_raw_product(
    raw_product_path: Path,
    obf_file_path: Path,
    azimuth_block_len: int = 10000,
) -> dict[str, list[tuple[str, float]]]:
    """Performing in/out-of-band power ratio computations on input RAW data product.

    Parameters
    ----------
    raw_product_path : Path
        Path to RAW product
    obf_file_path : Path
        Path to on-board filters file
    azimuth_block_len : int, optional
        Azimuth lines per processing block, by default 10000
    """

    # Open input RAW data product
    pf = open_product_folder(raw_product_path)
    channels = pf.get_channels_list()
    metadata_ref = read_metadata(pf.get_channel_metadata(1))
    raster_info_ref = metadata_ref.get_raster_info()
    pulse_ref = metadata_ref.get_pulse()

    # Read input on-board filters file
    filter_spectrum_dict, filter_spectrum_axis = read_on_board_filters_file(obf_file_path)

    # Precompute frequency axis and binning
    data_spectrum_axis = np.fft.fftshift(np.fft.fftfreq(raster_info_ref.samples, d=raster_info_ref.samples_step))
    n_bins = filter_spectrum_axis.size
    inv_step = 1.0 / (filter_spectrum_axis[1] - filter_spectrum_axis[0])
    data_spectrum_axis_id = ((data_spectrum_axis - filter_spectrum_axis[0]) * inv_step + 0.5).astype(np.int64)
    data_spectrum_axis_id = np.clip(data_spectrum_axis_id, 0, n_bins - 1)
    bin_count = np.bincount(data_spectrum_axis_id, minlength=n_bins)

    # Precompute band masks
    in_band_mask = (filter_spectrum_axis > -pulse_ref.bandwidth / 2) & (filter_spectrum_axis < pulse_ref.bandwidth / 2)
    out_band_mask = ~in_band_mask

    # Loop over input product channels
    results = {}
    azimuth_times = []
    for channel_idx in channels:
        # Read useful metadata
        metadata = read_metadata(pf.get_channel_metadata(channel_idx))
        raster_info = metadata.get_raster_info()
        swath_info = metadata.get_swath_info()

        # Loop over azimuth data blocks
        n_blocks = int(np.ceil(raster_info.lines / azimuth_block_len))
        for block_id in range(n_blocks):
            start_line = block_id * azimuth_block_len
            block_len = min(raster_info.lines - start_line, azimuth_block_len)
            block = [start_line, 0, block_len, raster_info.samples]

            if channel_idx == channels[0]:
                # Use always the first channel to determine azimuth times
                azimuth_time = str(raster_info.lines_start + start_line * raster_info.lines_step)
                azimuth_times.append(azimuth_time)
                results[azimuth_time] = []

            # Read input product azimuth data block
            data = read_raster_with_raster_info(
                raster_file=pf.get_channel_data(channel_idx),
                raster_info=raster_info,
                block_to_read=block,
            )

            # Compute average range spectrum (normalized)
            abs_sq = np.abs(data) ** 2
            normalization_factor = abs_sq.mean()
            data = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)
            abs_sq_fft = np.abs(data) ** 2
            data_spectrum = abs_sq_fft.mean(axis=0) / normalization_factor

            # Resample range spectrum
            with np.errstate(divide="ignore", invalid="ignore"):
                data_spectrum = np.bincount(data_spectrum_axis_id, weights=data_spectrum, minlength=n_bins) / bin_count
                data_spectrum[~np.isfinite(data_spectrum)] = 0

            # Equalize range spectrum
            filter_spectrum = filter_spectrum_dict[swath_info.polarization.value]
            valid_mask = filter_spectrum != 0
            data_spectrum_eq = np.zeros_like(data_spectrum)
            data_spectrum_eq[valid_mask] = data_spectrum[valid_mask] / filter_spectrum[valid_mask]

            # Compute in / out band power
            in_band_power = data_spectrum_eq[in_band_mask & valid_mask].mean()
            out_band_power = data_spectrum_eq[out_band_mask & valid_mask].mean()
            inout_band_power_ratio = in_band_power / out_band_power

            # Store results in a dictionary with azimuth time as key and list of (polarization, power ratio) as value
            results[azimuth_times[block_id]].append((swath_info.polarization.value, inout_band_power_ratio))

    return results
