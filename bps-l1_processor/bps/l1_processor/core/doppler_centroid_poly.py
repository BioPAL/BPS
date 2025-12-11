# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to write a doppler centroid polynomial file
-----------------------------------------------------
"""

from pathlib import Path

from arepytools.io import write_metadata
from arepytools.io.metadata import (
    DopplerCentroid,
    DopplerCentroidVector,
    EPolarization,
    MetaData,
    MetaDataChannel,
    SwathInfo,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_processor.io.l0_mph_utils import L0MainProductHeader
from bps.l1_processor.processor_interface.aux_pp1 import DopplerEstimationConf

DEFAULT_ACQUISITION_START_TIME = PreciseDateTime.from_numeric_datetime(2020)


def write_doppler_centroid_poly_file(
    mph: L0MainProductHeader,
    doppler_estimation_conf: DopplerEstimationConf,
    metadata_file_path: Path,
):
    """Write a doppler centroid polynomial file"""
    polarization_list = [
        EPolarization.hh,
        EPolarization.hv,
        EPolarization.vh,
        EPolarization.vv,
    ]

    swath = mph.swath

    offset_seconds = 20
    azimuth_start_time = mph.phenomenon_time_begin - offset_seconds

    metadata = MetaData()

    for polarization in polarization_list:
        swath_info = SwathInfo(swath.name, polarization_i=polarization)
        swath_info.acquisition_start_time = DEFAULT_ACQUISITION_START_TIME

        coefficient = doppler_estimation_conf.value

        doppler_centroid_list = [
            DopplerCentroid(
                i_ref_az=azimuth_start_time,
                i_ref_rg=0.0,
                i_coefficients=[coefficient],
            ),
        ]
        doppler_centroid_vector = DopplerCentroidVector(doppler_centroid_list)

        channel = MetaDataChannel()
        channel.insert_element(swath_info)
        channel.insert_element(doppler_centroid_vector)

        metadata.append_channel(channel=channel)

    write_metadata(metadata_obj=metadata, metadata_file=metadata_file_path)
