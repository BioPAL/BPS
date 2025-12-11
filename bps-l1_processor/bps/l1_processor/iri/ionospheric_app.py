# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
IRI app wrapper
---------------
"""

import os
from pathlib import Path
from typing import Any

from arepyextras.runner.environment import Environment
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.runner_helper import run_application_args
from bps.l1_processor.settings.l1_binaries import BPS_IRI_WRAPPER_EXE_NAME


def run_ionospheric_height_estimation(
    env: Environment,
    time: PreciseDateTime,
    point_llh: tuple[float, float, float],
    input_iri_data: Path,
    output_ionospheric_height: Path,
):
    """Wrapper to iri model app

    Parameters
    ----------
    time : PreciseDateTime
        input time
    point_llh : Tuple[float, float, float]
        input point lat [deg], lon [deg], height [m]
    input_iri_data : Path
        input folder containing the iri20 data
    output_ionospheric_height : Path
        output xml file containing the ionospheric height
    """

    lat, lon, h = point_llh

    year = time.year
    day_of_year = time.day_of_the_year
    utc_sec = time.hour_of_day * 3600.0 + time.minute_of_hour * 60.0
    decimal_hours = utc_sec / 3600.0 + 25.0

    cli_args: list[Any] = [
        str(input_iri_data) + os.sep,
        year,
        -day_of_year,
        decimal_hours,
        lat,
        lon,
        h,
        output_ionospheric_height,
    ]
    run_application_args(env, BPS_IRI_WRAPPER_EXE_NAME, cli_args)


def run_iri_wrapper(
    env: Environment,
    time: PreciseDateTime,
    point: tuple[float, float, float],
    input_iri_data: Path,
    ionospheric_height_model_file: Path,
):
    """Execute iri wrapper, write ionospheric height model file"""
    run_ionospheric_height_estimation(
        env,
        time,
        point,
        input_iri_data,
        ionospheric_height_model_file,
    )
