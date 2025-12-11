# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
PARC processing utils
---------
"""

import shutil
from pathlib import Path

from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import create_product_folder, iter_channels, open_product_folder, read_metadata, write_metadata
from arepytools.io.metadata import Pulse
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.common import STRIPMAP_SWATHS
from bps.l1_processor.parc.parc_info import ParcInfoList
from bps.l1_processor.parc.parc_processing_info import (
    Delays,
    ParcProcessingInfo,
    ScatteringResponse,
    Window,
)
from bps.l1_processor.processor_interface.aux_pp1 import AuxProcessingParametersL1
from bps.l1_processor.processor_interface.joborder_l1 import (
    L1JobOrder,
    L1StripmapOutputProducts,
)


def retrieve_chirp_length(chirp_replica_product: Path | None, pulse: Pulse) -> float:
    """Returns the length in seconds of the chirp replica if specified,
    else the length of the nominal chirp
    """

    # Chirp size from chirp replica
    if chirp_replica_product is not None:
        prod = open_product_folder(chirp_replica_product)
        ri = read_metadata(prod.get_channel_metadata(prod.get_channels_list()[0])).get_raster_info()
        return ri.samples * ri.samples_step

    # Chirp size from nominal chirp
    if pulse.pulse_length is None:
        raise RuntimeError("Missing Pulse section in metadata")

    return pulse.pulse_length


def compute_processing_window(
    window: Window,
    az_interval: tuple[PreciseDateTime, PreciseDateTime] | None,
    rg_interval: tuple[float, float] | None,
) -> Window:
    """Returns the processed product window"""

    return Window(
        azimuth_start=window.azimuth_start if az_interval is None else az_interval[0],
        azimuth_stop=window.azimuth_stop if az_interval is None else az_interval[1],
        range_start=window.range_start if rg_interval is None else rg_interval[0],
        range_stop=window.range_stop if rg_interval is None else rg_interval[1],
    )


def compute_parc_processing_window(
    az_time: PreciseDateTime,
    rg_time: float,
    roi_az_length: float,
    roi_rg_length: float,
    overlap_az_length: float,
    overlap_rg_length: float,
    chirp_length: float,
) -> Window:
    """Given a PARC in time coordinates, constructs a window centered at it"""

    half_window_az_length = (roi_az_length + overlap_az_length) / 2
    half_window_rg_length = (roi_rg_length + overlap_rg_length) / 2
    return Window(
        azimuth_start=az_time - half_window_az_length,
        azimuth_stop=az_time + half_window_az_length,
        range_start=rg_time - half_window_rg_length,
        range_stop=rg_time + half_window_rg_length + chirp_length,
    )


def convert_scattering_response_delay_to_az_rg_delays(scattering_response_delay: float, pri: float) -> Delays:
    """Scattering response delay may be larger than PRI;
    if so, delay is split in azimuth and range delays
    """
    az_delay = scattering_response_delay / 2.0  # azimuth shift considering extra bistatic correction
    rg_delay = (
        scattering_response_delay - round(scattering_response_delay / pri) * pri
    )  # round is needed here to get range delay within echo window position
    return Delays(az_delay, rg_delay)


def is_point_contained(
    window: Window,
    az_time: PreciseDateTime,
    rg_time: float,
) -> bool:
    """Checks whether a time point is contained in a window"""
    return window.azimuth_start <= az_time <= window.azimuth_stop and window.range_start <= rg_time <= window.range_stop


def is_window_contained(container: Window, contained: Window) -> bool:
    """Checks whether a window is fully contained in a bigger window"""
    return is_point_contained(container, contained.azimuth_start, contained.range_start) and is_point_contained(
        container, contained.azimuth_stop, contained.range_stop
    )


def earth2sat(general_sar_orbit, position) -> tuple[PreciseDateTime, float]:
    """Just to hide some unpacking"""
    az_times, rg_times = general_sar_orbit.earth2sat(position)
    return az_times[0], rg_times[0]


def detect_parc_processing(
    parc_info_list: ParcInfoList,
    raw_product: Path,
    azimuth_processing_interval: tuple[PreciseDateTime, PreciseDateTime] | None,
    range_processing_interval: tuple[float, float] | None,
    chirp_replica_product: Path | None,
    block_overlap_lines: int,
    block_overlap_samples: int,
    parc_roi_lines: int,
    parc_roi_samples: int,
    azimuth_sampling_frequency: float | None,
) -> ParcProcessingInfo | None:
    """Determines if PARC processing is to be executed and in case
    returns the relevant information.

    Parameters
    ----------
    parc_info_list : ParcInfoList
        List of PARCs
    raw_product : Path
        path to the raw product
    azimuth_processing_interval : Optional[Tuple[PreciseDateTime, PreciseDateTime]],
        optional raw product processing sub-axis in azimuth direction
    range_processing_interval : Optional[Tuple[float, float]],
        optional raw product processing sub-axis in range direction
    chirp_replica_product : Union[Path, None]
        optional path to the chirp replica product for the chirp size;
        if unspecified, chirp size is taken from metadata of the raw product
    block_overlap_lines : int
        block overlap (transient) in azimuth direction expressed in number of lines
    block_overlap_samples : int
        block overlap (transient) in range direction expressed in number of samples
    parc_roi_lines : int
        number of lines around the PARC for the PARC processing
    parc_roi_samples : int
        number of samples around the PARC for the PARC processing
    azimuth_sampling_frequency : Optional[float]
        azimuth sampling frequency at azimuth compression level
    """

    # Extract information from raw product metadata
    prod = open_product_folder(raw_product)
    product_reference_metadata = read_metadata(prod.get_channel_metadata(prod.get_channels_list()[0]))
    raster_info = product_reference_metadata.get_raster_info()
    orbit = create_general_sar_orbit(product_reference_metadata.get_state_vectors())
    chirp_length = retrieve_chirp_length(chirp_replica_product, product_reference_metadata.get_pulse())
    bps_logger.debug("PARC detection: chirp length = %e s", chirp_length)

    # Compute overlap
    azimuth_sampling_frequency = (
        azimuth_sampling_frequency if azimuth_sampling_frequency is not None else 1.0 / raster_info.lines_step
    )
    overlap_az_length = block_overlap_lines / azimuth_sampling_frequency
    overlap_rg_length = block_overlap_samples * raster_info.samples_step

    # Raw product window
    product_window = Window(
        raster_info.lines_start,  # type: ignore
        raster_info.lines_start + raster_info.lines_step * raster_info.lines,  # type: ignore
        raster_info.samples_start,
        raster_info.samples_start + raster_info.samples_step * raster_info.samples,
    )

    # Raw product processing window
    processing_window = compute_processing_window(
        product_window, azimuth_processing_interval, range_processing_interval
    )

    # Fill list of PARCs for which PARC processing is feasible
    parc_processing_info_list = []
    for parc_info in parc_info_list:
        parc_az_time, parc_rg_time = earth2sat(orbit, parc_info.position)

        # Compute and check PARC processing window for each scattering response
        parc_processing_info = ParcProcessingInfo(parc_info.parc_id)
        valid_parc = True
        for scattering_response in ScatteringResponse:
            delays = convert_scattering_response_delay_to_az_rg_delays(
                scattering_response_delay=parc_info.delays[scattering_response],
                pri=raster_info.lines_step,
            )

            # PARC that is delayed according to scattering response must be
            # contained at least in the raw product processing window
            delayed_parc_az_time = parc_az_time + delays.azimuth_delay
            delayed_parc_rg_time = parc_rg_time + delays.range_delay
            if not is_point_contained(processing_window, delayed_parc_az_time, delayed_parc_rg_time):
                valid_parc = False
                break

            # PARC processing window must be fully contained in the
            # product window
            parc_processing_window = compute_parc_processing_window(
                az_time=delayed_parc_az_time,
                rg_time=delayed_parc_rg_time,
                roi_az_length=parc_roi_lines * raster_info.lines_step,
                roi_rg_length=parc_roi_samples * raster_info.samples_step,
                overlap_az_length=overlap_az_length,
                overlap_rg_length=overlap_rg_length,
                chirp_length=chirp_length,
            )
            if not is_window_contained(product_window, parc_processing_window):
                valid_parc = False
                break

            # PARC is valid: store processing data
            parc_processing_info.processing_data[scattering_response] = (
                delays,
                parc_processing_window,
            )

        if valid_parc:
            assert len(parc_processing_info.processing_data) == 4
            parc_processing_info_list.append(parc_processing_info)

    if len(parc_processing_info_list) > 1:
        raise RuntimeError(f"Detected {len(parc_processing_info_list)} PARCs in the scene: only one PARC is supported")

    if len(parc_processing_info_list) == 1:
        return parc_processing_info_list[0]

    return None


def update_job_order_for_parc_processing(job_order: L1JobOrder, parc_processing_roi: Window) -> L1JobOrder:
    """Creates a copy of the JobOrder with proper azimuth and
    range intervals for PARC processing.

    Parameters
    ----------
    job_order : L1JobOrder
        JobOrder object
    parc_processing_roi : Window
        Processing ROI for the PARC processing
    """

    job_order.processor_configuration.azimuth_interval = (
        parc_processing_roi.azimuth_start,
        parc_processing_roi.azimuth_stop,
    )
    job_order.processing_parameters.range_interval = (
        parc_processing_roi.range_start,
        parc_processing_roi.range_stop,
    )

    if not isinstance(job_order.io_products.output, L1StripmapOutputProducts):
        raise RuntimeError("Cannot perform PARC processing for non-stripmap acquisitions")
    job_order.io_products.output.dgm_standard_required = False

    return job_order


def update_aux_pp1_for_parc_processing(
    aux_pp1: AuxProcessingParametersL1, parc_delays: Delays
) -> AuxProcessingParametersL1:
    """Creates a copy of the AUX_PP1 with proper azimuth and
    range delays for PARC processing.

    Parameters
    ----------
    aux_pp1 : AuxProcessingParametersL1
        AUX_PP1 object
    parc_delays : Window
        Azimuth and range delays for the PARC processing
    """

    for swath in STRIPMAP_SWATHS:
        aux_pp1.azimuth_compression.parameters[swath].time_bias += parc_delays.azimuth_delay
        aux_pp1.range_compression.parameters[swath].time_bias += parc_delays.range_delay

    aux_pp1.azimuth_compression.block_lines = (
        aux_pp1.azimuth_compression.block_overlap_lines
    ) + aux_pp1.general.parc_roi_lines // 2

    return aux_pp1


def apply_delay_to_product(input_product: Path, parc_delays: Delays, delayed_product: Path):
    """Apply delay directly to raster info"""

    input_pf = open_product_folder(input_product)
    output_pf = create_product_folder(delayed_product, overwrite_ok=True)

    for channel_index, channel_metadata in iter_channels(input_pf):
        input_raster_file = input_pf.get_channel_data(channel_index)
        output_raster_file = output_pf.get_channel_data(channel_index)
        output_metadata_file = output_pf.get_channel_metadata(channel_index)

        shutil.copy2(input_raster_file, output_raster_file)

        raster_info = channel_metadata.get_raster_info()
        raster_info.file_name = output_raster_file.name
        raster_info.set_lines_axis(
            raster_info.lines_start - parc_delays.azimuth_delay,
            raster_info.lines_start_unit,
            raster_info.lines_step,
            raster_info.lines_step_unit,
        )
        raster_info.set_samples_axis(
            raster_info.samples_start - parc_delays.range_delay,
            raster_info.samples_start_unit,
            raster_info.samples_step,
            raster_info.samples_step_unit,
        )

        write_metadata(channel_metadata, output_metadata_file)
