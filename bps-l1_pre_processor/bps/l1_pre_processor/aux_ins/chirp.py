# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX INS Chirp management
------------------------
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    metadata,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import AcquisitionMode, Polarization, bps_logger
from bps.l1_pre_processor.aux_ins import netcdf_utils
from numpy import typing as npt

DEFAULT_ACQUISITION_START_TIME = PreciseDateTime.from_numeric_datetime(2020)


@dataclass
class ChirpReplica:
    """Biomass chirp replica"""

    axis: npt.NDArray[np.float64]
    replica: npt.NDArray[np.complex128]


ChirpReplicas = dict[Polarization, ChirpReplica]
"""Biomass chirp replicas by polarization"""


class InvalidChirpReplicaNETCDFFile(RuntimeError):
    """Raised when a chirp replica netcdf file is invalid"""


NETCDF_CHIRP_REPLICA_GROUP = "chirpReplicas"
NETCDF_CHIRP_REPLICA_VARIABLES: dict[Polarization, str] = {
    Polarization.HH: "chirpReplicaVV",
    Polarization.HV: "chirpReplicaHV",
    Polarization.VH: "chirpReplicaVH",
    Polarization.VV: "chirpReplicaVV",
}
NETCDF_SLANTRANGETIME_NAME = "slantRangeTime"


def read_chirp_replicas(
    chirp_replica_netcdf_file: Path,
) -> ChirpReplicas:
    """Read chirp replicas from the input netcdf file

    Parameters
    ----------
    chirp_replica_netcdf_file : Path
        input netcdf file containing the chirp replica

    Returns
    -------
    ChirpReplicas
        chirp replicas

    Raises
    ------
    InvalidChirpReplicaNETCDFFile
        In case of expected information is not found in the netcdf file
    """

    dataset = netcdf_utils.get_dataset(chirp_replica_netcdf_file)

    slant_range_time = netcdf_utils.read_dimension(dataset, NETCDF_SLANTRANGETIME_NAME)
    if slant_range_time is None:
        raise InvalidChirpReplicaNETCDFFile(chirp_replica_netcdf_file)

    output_replicas: ChirpReplicas = {}
    for polarization_id, variable_name in NETCDF_CHIRP_REPLICA_VARIABLES.items():
        replica = netcdf_utils.read_group_variable(dataset, NETCDF_CHIRP_REPLICA_GROUP, variable_name)
        if replica is None:
            raise InvalidChirpReplicaNETCDFFile(chirp_replica_netcdf_file)

        output_replicas[polarization_id] = ChirpReplica(axis=slant_range_time, replica=replica)

    return output_replicas


def fill_swath_info(swath_name: str, polarization: Polarization) -> metadata.SwathInfo:
    """Fill basic swath info"""
    polarization_str = polarization.value[0] + "/" + polarization.value[1]
    swath_info = metadata.SwathInfo(swath_i=swath_name, polarization_i=polarization_str)
    swath_info.acquisition_start_time = DEFAULT_ACQUISITION_START_TIME
    return swath_info


def write_chirp_replicas_to_product_folder(
    chirp_replica_product: Path,
    chirp_replicas: ChirpReplicas,
    swath_name: str,
):
    """Write chirp replicas to product folder

    Parameters
    ----------
    chirp_replica_product : Path
        chirp replica product folder
    chirp_replicas : ChirpReplicas
        chirp replicas by polarization
    swath_name : str
        current swath
    """

    product = create_product_folder(chirp_replica_product, overwrite_ok=True)

    for channel_index, (polarization, replica) in enumerate(chirp_replicas.items()):
        ch_idx = channel_index + 1
        raster_name = product.get_channel_data(ch_idx)

        meta = create_new_metadata()
        raster_info = metadata.RasterInfo(
            lines=1,
            samples=replica.axis.size,
            celltype=metadata.ECellType.fcomplex,
            filename=raster_name.name,
        )
        raster_info.set_lines_axis(lines_start=0, lines_start_unit="s", lines_step=0, lines_step_unit="s")
        raster_info.set_samples_axis(
            samples_start=float(replica.axis[0]),
            samples_start_unit="s",
            samples_step=float((replica.axis[1] - replica.axis[0] if replica.axis.size > 1 else 0.0)),
            samples_step_unit="s",
        )
        meta.insert_element(raster_info)

        swath_info = fill_swath_info(swath_name, polarization)
        meta.insert_element(swath_info)

        write_metadata(meta, metadata_file=product.get_channel_metadata(ch_idx))
        write_raster_with_raster_info(
            raster_file=raster_name,
            data=np.reshape(replica.replica, (1, -1)),
            raster_info=raster_info,
        )


def transcode_input_chirp_replica_to_product_folder(
    chirp_files: dict[AcquisitionMode, Path],
    chirp_replica_product: Path,
    acquisition_mode: AcquisitionMode,
):
    """Translate chirp replica from netcdf file to product folders"""
    chirp_file = chirp_files[acquisition_mode]
    bps_logger.debug(f"Internal Chirp file: {chirp_file}")
    replicas = read_chirp_replicas(chirp_file)
    swath, _ = acquisition_mode
    write_chirp_replicas_to_product_folder(chirp_replica_product, replicas, swath.value)
