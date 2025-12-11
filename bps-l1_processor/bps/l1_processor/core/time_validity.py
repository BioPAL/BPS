# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Validity interval
-----------------
"""

from pathlib import Path

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.common import retrieve_aux_product_data_single_content
from bps.common.io.parsing import parse
from bps.l1_processor.core.time_interval import TimeInterval, are_overlapping, contains
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder
from bps.transcoder.io import aux_att_models, aux_orb_models


def _raise_on_inconsistent_timings(
    data_interval: TimeInterval,
    jo_interval: TimeInterval | None,
    orbit_interval: TimeInterval,
    attitude_interval: TimeInterval,
):
    """Check wether processing time in the job order, in the standard product
    and in the orbit and attitude files are consistent"""
    if jo_interval is not None:
        if not are_overlapping(jo_interval, data_interval):
            raise RuntimeError(
                f"Requested times [{jo_interval[0]}, {jo_interval[1]}] are incompatible"
                + f" with input phenomenon times [{data_interval[0]}, {data_interval[1]}]"
            )

    display_name = "requested times" if jo_interval else "phenomenon times"
    proc_interval = jo_interval if jo_interval else data_interval
    if not contains(orbit_interval, proc_interval):
        raise RuntimeError(
            f"Orbit validity times [{orbit_interval[0]}, {orbit_interval[1]}] are "
            + f"incompatible with input {display_name} [{proc_interval[0]}, {proc_interval[1]}]"
        )

    if not contains(attitude_interval, proc_interval):
        raise RuntimeError(
            f"Attitude validity times [{attitude_interval[0]}, {attitude_interval[1]}] are "
            + f"incompatible with input {display_name} [{proc_interval[0]}, {proc_interval[1]}]"
        )


def _utc_to_pdt(utc: str) -> PreciseDateTime:
    return PreciseDateTime.fromisoformat(utc.strip("UTC="))


def _retrieve_validity_interval_from_fixed_header(
    orbit_or_attitude: (aux_orb_models.EarthObservationFile | aux_att_models.EarthObservationFile),
) -> tuple[PreciseDateTime, PreciseDateTime]:
    """Retrieve validity interval from orbit or attitude"""

    assert orbit_or_attitude.earth_observation_header is not None
    assert orbit_or_attitude.earth_observation_header.fixed_header is not None
    validity_period = orbit_or_attitude.earth_observation_header.fixed_header.validity_period
    assert validity_period is not None
    assert validity_period.validity_start is not None
    assert validity_period.validity_stop is not None
    start = _utc_to_pdt(validity_period.validity_start)
    stop = _utc_to_pdt(validity_period.validity_stop)
    return start, stop


def _retrieve_validity_interval_orbit(
    orbit: aux_orb_models.EarthObservationFile,
) -> tuple[PreciseDateTime, PreciseDateTime]:
    """Retrieve validity interval from orbit or attitude"""
    assert orbit.data_block is not None
    assert orbit.data_block.list_of_osvs is not None
    start_utc = orbit.data_block.list_of_osvs.osv[0].utc
    stop_utc = orbit.data_block.list_of_osvs.osv[-1].utc
    assert start_utc is not None and stop_utc is not None
    return _utc_to_pdt(start_utc), _utc_to_pdt(stop_utc)


def _retrieve_validity_interval_attitude(
    attitude: aux_att_models.EarthObservationFile,
) -> tuple[PreciseDateTime, PreciseDateTime]:
    """Retrieve validity interval from attitude"""
    assert attitude.data_block is not None
    assert attitude.data_block.quaternion_data is not None
    assert attitude.data_block.quaternion_data.list_of_quaternions is not None
    quat = attitude.data_block.quaternion_data.list_of_quaternions.quaternions
    assert quat[0].time is not None and quat[-1].time is not None
    start_utc = quat[0].time.value
    stop_utc = quat[-1].time.value
    return _utc_to_pdt(start_utc), _utc_to_pdt(stop_utc)


def retrieve_orbit_validity_interval(orbit_product: Path):
    """Retrieve validity interval from orbit"""
    xml_file = retrieve_aux_product_data_single_content(orbit_product)
    model: aux_orb_models.EarthObservationFile = parse(xml_file.read_text(), aux_orb_models.EarthObservationFile)
    return _retrieve_validity_interval_orbit(model)


def retrieve_attitude_validity_interval(attitude_product: Path):
    """Retrieve validity interval from attitude"""
    xml_file = retrieve_aux_product_data_single_content(attitude_product)
    model: aux_att_models.EarthObservationFile = parse(xml_file.read_text(), aux_att_models.EarthObservationFile)
    return _retrieve_validity_interval_attitude(model)


def raise_on_inconsistent_timings(data_interval: TimeInterval, job_order: L1JobOrder):
    """Check wether processing time in the job order, in the standard product
    and in the orbit and attitude files are consistent"""
    _raise_on_inconsistent_timings(
        data_interval=data_interval,
        jo_interval=job_order.processor_configuration.azimuth_interval,
        orbit_interval=retrieve_orbit_validity_interval(job_order.auxiliary_files.orbit),
        attitude_interval=retrieve_attitude_validity_interval(job_order.auxiliary_files.attitude),
    )
