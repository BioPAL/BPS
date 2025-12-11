# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""L1 LUT reader"""

from pathlib import Path
from typing import Any

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from netCDF4 import Dataset

VALID_LUT_LAYERS = [
    "rfiMaskHH",  # for backward compatibility
    "rfiMaskHV",  # for backward compatibility
    "rfiMaskVH",  # for backward compatibility
    "rfiMaskVV",  # for backward compatibility
    "rfiTimeMaskHH",
    "rfiTimeMaskHV",
    "rfiTimeMaskVH",
    "rfiTimeMaskVV",
    "rfiFreqMaskHH",
    "rfiFreqMaskHV",
    "rfiFreqMaskVH",
    "rfiFreqMaskVV",
    "sigmaNought",
    "gammaNought",
    "faradayRotation",
    "phaseScreen",
    "tec",
    "rangeShifts",
    "azimuthShifts",
    "autofocusPhaseScreen",
    "denoisingHH",
    "denoisingHV",
    "denoisingVH",
    "denoisingVV",
    "latitude",
    "longitude",
    "height",
    "incidenceAngle",
    "elevationAngle",
    "terrainSlope",
    "faradayRotationPlane",
    "faradayRotationStd",
]


def read_lut_file(
    lut_file: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read lut annotation"""

    lut_dataset = Dataset(lut_file, mode="r")

    groups: dict[str, Any] = lut_dataset.groups

    variables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group_name, group in groups.items():
        for layer_name in group.variables:
            try:
                variables[layer_name] = read_lut_variable(
                    lut_dataset,
                    group_name,
                    layer_name,
                )
            except Exception:
                pass

    # Check unexpected layers
    unexpected_layers = set(filter(lambda layer: layer not in VALID_LUT_LAYERS, variables))
    if unexpected_layers:
        raise RuntimeError(f"{lut_file:s} contains unexpected LUTs ({unexpected_layers})")

    lut_start_time = PreciseDateTime.from_utc_string(read_lut_attribute(lut_dataset, "startTime"))

    # Backward compatibility
    old_tag_translation_map = {
        "rfiMaskHH": "rfiTimeMaskHH",
        "rfiMaskHV": "rfiTimeMaskHV",
        "rfiMaskVH": "rfiTimeMaskVH",
        "rfiMaskVV": "rfiTimeMaskVV",
    }
    for old_tag, new_tag in old_tag_translation_map.items():
        if old_tag in variables:
            variables[new_tag] = variables.pop(old_tag)

    return (
        {layer: var[2] for layer, var in variables.items()},
        {layer: var[0] for layer, var in variables.items()},
        {layer: var[1] + lut_start_time for layer, var in variables.items()},
    )


def read_lut_attribute(dataset, attribute):
    return dataset.__dict__[attribute]


def _read_lut_axis(dataset, group, variable):
    dim_azm, dim_rng = dataset.groups[group][variable].dimensions
    return (
        np.array(dataset.variables[dim_rng][:]),
        np.array(dataset.variables[dim_azm][:]),
    )


def _read_lut_array(dataset, group, variable, dtype):
    return np.asarray(dataset.groups[group][variable][:], dtype=dtype)


def read_lut_variable(dataset, group, variable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read variable with axis"""

    dtype = np.float64
    if dataset.groups[group][variable][:].dtype == np.uint8:
        dtype = np.uint8
    array = _read_lut_array(dataset, group, variable, dtype)

    rg_axis, az_axis = _read_lut_axis(dataset, group, variable)
    return rg_axis, az_axis, array
