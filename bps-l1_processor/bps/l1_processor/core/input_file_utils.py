# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to fill the input file
--------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from arepytools import io
from arepytools.io import metadata
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_core_processor.input_file import (
    AntennaProducts,
    BPSL1CoreProcessorInputFile,
    CoreProcessorInputs,
)
from bps.l1_core_processor.pf_selector_input_file import (
    IndexInterval,
    PFSelectorAreaRasterCoordinates,
    PFSelectorAreaTimeCoordinates,
)
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder


def _get_metadata_sections(
    product: Path,
) -> tuple[metadata.RasterInfo, metadata.AcquisitionTimeLine, metadata.SwathInfo]:
    """Get metadata sections from product first channel"""
    raw_pf = io.open_product_folder(product)
    first_channel = raw_pf.get_channels_list()[0]
    raw_metadata = io.read_metadata(raw_pf.get_channel_metadata(first_channel))

    raster_info = raw_metadata.get_raster_info()
    timeline = raw_metadata.get_acquisition_time_line()
    swath_info = raw_metadata.get_swath_info()
    return raster_info, timeline, swath_info


def _clamp(value: int, v_min: int, v_max: int) -> int:
    """Clamp"""
    return min(max(value, v_min), v_max)


def convert_range_time_interval_to_samples_interval(
    time_interval: tuple[float, float], raster_info: metadata.RasterInfo, timeline: metadata.AcquisitionTimeLine
) -> tuple[int, int]:
    """Convert range time interval into samples interval"""
    assert isinstance(raster_info.samples_start, float)
    sample_interval = (
        math.floor((time_interval[0] - raster_info.samples_start) / raster_info.samples_step),
        math.ceil((time_interval[1] - raster_info.samples_start) / raster_info.samples_step),
    )
    num_changes, change_times, change_values = timeline.swst_changes
    if num_changes > 1:
        assert change_times is not None
        assert change_values is not None

        max_increment = np.max(change_values) - change_values[0]
        max_decrement = change_values[0] - np.min(change_values)
        max_increment = math.ceil(max_increment / raster_info.samples_step)
        max_decrement = math.ceil(max_decrement / raster_info.samples_step)
        sample_interval = (sample_interval[0] - max_increment, sample_interval[1] + max_decrement)

    return (
        _clamp(sample_interval[0], 0, raster_info.samples - 1),
        _clamp(sample_interval[1], 0, raster_info.samples - 1),
    )


def convert_azimuth_time_interval_to_lines_interval(
    time_interval: tuple[PreciseDateTime, PreciseDateTime], raster_info: metadata.RasterInfo
):
    """ "Convert azimuth time interval into lines interval"""
    time_interval_relative = (
        time_interval[0] - raster_info.lines_start_date,
        time_interval[1] - raster_info.lines_start_date,
    )
    lines_interval = (
        math.floor(time_interval_relative[0] / raster_info.lines_step),
        math.ceil(time_interval_relative[1] / raster_info.lines_step),
    )

    return (
        _clamp(lines_interval[0], 0, raster_info.lines - 1),
        _clamp(lines_interval[1], 0, raster_info.lines - 1),
    )


def fill_bps_l1_core_processor_input_file(
    *,
    job_order: L1JobOrder,
    input_raw_product: Path,
    processing_options: Path,
    processing_parameters: Path,
    bps_configuration_file: Path,
    bps_log_file: Path,
    output_dir: Path,
    input_chirp_replica_product: Path | None = None,
    input_per_line_correction_factors_product: Path | None = None,
    input_processing_dc_poly_file_name: Path | None = None,
    input_d1h_pattern_product: Path | None = None,
    input_d2h_pattern_product: Path | None = None,
    input_d1v_pattern_product: Path | None = None,
    input_d2v_pattern_product: Path | None = None,
    input_tx_power_tracking_product: Path | None = None,
    input_noise_product: Path | None = None,
    input_geomagnetic_field_product: Path | None = None,
    input_tec_map_product: Path | None = None,
    input_climatological_model_file: Path | None = None,
    input_faraday_rotation_product: Path | None = None,
    input_phase_screen_product: Path | None = None,
) -> BPSL1CoreProcessorInputFile:
    """Fill the input file of the BPS L1 Core processor

    Parameters
    ----------
    job_order : L1JobOrder
        job order object
    input_raw_product : Path
        input extracted raw product
    processing_options: Path
        BPS L1 core processor processing options file
    processing_parameters: Path
        BPS L1 core processor processing parameters file
    input_chirp_replica_product : Path
        input chirp replica product
    input_per_line_correction_factors_product : Path
        input per-line correction factors product
    input_processing_dc_poly_file_name : Path
        input doppler centroid polynomials file
    input_d1h_pattern_product: Path
        First doublet h pol pattern product
    input_d2h_pattern_product: Path
        Second doublet h pol pattern product
    input_d1v_pattern_product: Path
        First doublet v pol pattern product
    input_d2v_pattern_product: Path
        Second doublet v pol pattern product
    input_tx_power_tracking_product: Path
        TX power tracking product
    bps_configuration_file: Path
        BPS configuration file
    bps_log_file: Path
        BPS log file
    output_path: Path
        BPS L1 core processor output folder
    input_noise_product : Path
        input est noise product
    input_geomagnetic_field_product : Optional[Path]
        input geomagnetic field folder
    input_tec_map_product : Optional[Path]
        input tec map product
    input_climatological_model_file : Optional[Path]
        input climatological xml file with ionospheric height
    input_faraday_rotation_product : Optional[Path]
        input faraday rotation product
    input_phase_screen_product : Optional[Path]
        input phase screen product

    Returns
    -------
    BPSL1CoreProcessorInputFile
        BPS L1 core processor app input file object
    """
    area_to_process = None
    if (
        job_order.processor_configuration.azimuth_interval is not None
        and job_order.processing_parameters.range_interval is None
    ):
        start = job_order.processor_configuration.azimuth_interval[0]
        duration = job_order.processor_configuration.azimuth_interval[1] - start
        azimuth_time_interval = PFSelectorAreaTimeCoordinates.AzimuthTimeInterval(start_time=start, duration=duration)
        area_to_process = PFSelectorAreaTimeCoordinates(
            azimuth_time_interval=azimuth_time_interval, range_time_interval=None
        )
    elif job_order.processing_parameters.range_interval is not None:
        raster_info, timeline, swath_info = _get_metadata_sections(input_raw_product)
        swath = swath_info.swath
        assert swath is not None

        sample_start, sample_stop = convert_range_time_interval_to_samples_interval(
            job_order.processing_parameters.range_interval, raster_info=raster_info, timeline=timeline
        )
        length = sample_stop + 1 - sample_start
        samples_interval = IndexInterval(start_index=sample_start, length=length)

        azimuth_time_interval = None
        if job_order.processor_configuration.azimuth_interval is not None:
            line_start, line_stop = convert_azimuth_time_interval_to_lines_interval(
                job_order.processor_configuration.azimuth_interval, raster_info=raster_info
            )
            length = line_stop + 1 - line_start
            azimuth_time_interval = IndexInterval(start_index=line_start, length=length)

        area_to_process = PFSelectorAreaRasterCoordinates(
            swaths=[
                PFSelectorAreaRasterCoordinates.Swath(
                    name=swath, samples_interval=samples_interval, lines_interval=azimuth_time_interval
                ),
            ],
        )

    core_processor_inputs = CoreProcessorInputs(
        input_level0_product=input_raw_product,
        input_chirp_replica_product=input_chirp_replica_product,
        input_per_line_correction_factors_product=input_per_line_correction_factors_product,
        input_noise_product=input_noise_product,
        input_processing_dc_poly_file_name=input_processing_dc_poly_file_name,
        processing_options_file=processing_options,
        processing_parameters_file=processing_parameters,
        area_to_process=area_to_process,
        output_directory=output_dir,
    )

    antenna_pattern_inputs_availabilty = (
        input_d1h_pattern_product is not None,
        input_d2h_pattern_product is not None,
        input_d1v_pattern_product is not None,
        input_d2v_pattern_product is not None,
        input_tx_power_tracking_product is not None,
    )

    if all(antenna_pattern_inputs_availabilty):
        assert input_d1h_pattern_product is not None
        assert input_d2h_pattern_product is not None
        assert input_d1v_pattern_product is not None
        assert input_d2v_pattern_product is not None
        assert input_tx_power_tracking_product is not None
        antenna_pattern_products = AntennaProducts(
            d1h_pattern_product=input_d1h_pattern_product,
            d2h_pattern_product=input_d2h_pattern_product,
            d1v_pattern_product=input_d1v_pattern_product,
            d2v_pattern_product=input_d2v_pattern_product,
            tx_power_tracking_product=input_tx_power_tracking_product,
        )
    elif not any(antenna_pattern_inputs_availabilty):
        assert input_d1h_pattern_product is None
        assert input_d2h_pattern_product is None
        assert input_d1v_pattern_product is None
        assert input_d2v_pattern_product is None
        assert input_tx_power_tracking_product is None
        antenna_pattern_products = None
    else:
        raise RuntimeError("Not all antenna pattern input products are available")

    return BPSL1CoreProcessorInputFile(
        core_processor_input=core_processor_inputs,
        input_antenna_products=antenna_pattern_products,
        input_geomagnetic_field_model_product=input_geomagnetic_field_product,
        input_tec_map_product=input_tec_map_product,
        input_climatological_model_file=input_climatological_model_file,
        input_faraday_rotation_product=input_faraday_rotation_product,
        input_phase_screen_product=input_phase_screen_product,
        bps_configuration_file=bps_configuration_file,
        bps_log_file=bps_log_file,
    )
