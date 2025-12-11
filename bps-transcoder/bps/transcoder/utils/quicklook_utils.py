# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Quicklook utilities"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import numpy.typing as npt
from bps.common import bps_logger


@dataclass
class QuickLookConf:
    """Quicklook configuration"""

    range_decimation_factor: int = 2
    range_averaging_factor: int = 2
    azimuth_decimation_factor: int = 12
    azimuth_averaging_factor: int = 12
    absolute_scaling_factor: float | dict = 1.0
    max_percentile: float = 99

    @property
    def apply_decimation(self) -> bool:
        """Wether decimation is required based on configuration"""
        return self.range_averaging_factor > 1 or self.azimuth_averaging_factor > 1


def warn_on_invalid_values(data: npt.NDArray[np.complex64] | npt.NDArray[np.float32], data_name: str) -> bool:
    """Warn if data contains NaN and Inf values."""
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    if nan_count == 0 and inf_count == 0:
        return False

    total_pixels = data.size
    nan_percentage = (nan_count / total_pixels) * 100.0
    inf_percentage = (inf_count / total_pixels) * 100.0

    bps_logger.warning(
        f"{data_name} contains NaN - percentage: {nan_percentage:.2f}% "
        f"(count: {nan_count}) and Inf - percentage: {inf_percentage:.2f}% "
        f"(count: {inf_count})."
    )
    return True


@dataclass
class RGBConverter:
    hh: npt.NDArray[np.complex64] | npt.NDArray[np.float32]
    hv: npt.NDArray[np.complex64] | npt.NDArray[np.float32]
    vh: npt.NDArray[np.complex64] | npt.NDArray[np.float32]
    vv: npt.NDArray[np.complex64] | npt.NDArray[np.float32]
    xx: npt.NDArray[np.complex64] | npt.NDArray[np.float32] | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, npt.NDArray[np.complex64] | npt.NDArray[np.float32]],
        dtype=np.complex64,
    ) -> Self:
        """Create QuadpolData from a dictionary of arrays"""
        assert dtype in (np.complex64, np.float32)
        return cls(
            hh=data.get("H/H", np.zeros(shape=(1, 1), dtype=dtype)),
            hv=data.get("H/V", np.zeros(shape=(1, 1), dtype=dtype)),
            vh=data.get("V/H", np.zeros(shape=(1, 1), dtype=dtype)),
            vv=data.get("V/V", np.zeros(shape=(1, 1), dtype=dtype)),
            xx=data.get("X/X", None),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the image"""
        return max(self.hh.shape, self.hv.shape, self.vh.shape, self.vv.shape, key=lambda x: x[0] * x[1])

    @property
    def is_complex(self) -> bool:
        """Check if the data is complex"""
        return (
            np.iscomplexobj(self.hh)
            or np.iscomplexobj(self.hv)
            or np.iscomplexobj(self.vh)
            or np.iscomplexobj(self.vv)
            or (self.xx is not None and np.iscomplexobj(self.xx))
        )

    def pauli_decomposition(self) -> npt.NDArray[np.float32]:
        """Compute RGB via Pauli decomposition"""
        rgb = np.zeros(self.shape + (3,), np.float32)

        rgb[:, :, 0] = np.abs(self.hh - self.vv)
        rgb[:, :, 1] = np.abs(self.hv + self.vh)
        if self.xx is not None:
            rgb[:, :, 1] += 2 * np.abs(self.xx)
        rgb[:, :, 2] = np.abs(self.hh + self.vv)

        return rgb

    def lexicographc_representation(self) -> npt.NDArray[np.float32]:
        """Compute RGB via lexicographic representation"""
        rgb = np.zeros(self.shape + (3,), np.float32)

        rgb[:, :, 0] = np.abs(self.hh)
        rgb[:, :, 1] = np.sqrt(self.hv**2 + self.vh**2) * 0.5
        if self.xx is not None:
            rgb[:, :, 1] += np.abs(self.xx)
        rgb[:, :, 2] = np.abs(self.vv)

        return rgb

    def to_rgb(self) -> npt.NDArray[np.float32]:
        """Convert quad pol complex data to RGB channels"""
        if self.is_complex:
            bps_logger.debug("Converting complex data to RGB using Pauli decomposition.")
            return self.pauli_decomposition()

        bps_logger.debug("Converting real data to RGB using lexicographic representation.")
        return self.lexicographc_representation()


def decimation(rgb: np.ndarray, conf: QuickLookConf):
    """Low pass filtering and decimation"""
    boxcar = np.ones((conf.azimuth_averaging_factor, conf.range_averaging_factor)) / (
        conf.azimuth_averaging_factor * conf.range_averaging_factor
    )
    rgb = rgb**2
    cv2.filter2D(
        src=rgb,
        ddepth=-1,
        dst=rgb,
        kernel=boxcar,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.sqrt(rgb[:: conf.azimuth_decimation_factor, :: conf.range_decimation_factor, :])


def scale_to_256(
    rgb: np.ndarray, absolute_scaling_factor: float | dict, max_percentile: float
) -> npt.NDArray[np.uint8]:
    """Rescale image to 0-255 uint8"""
    if isinstance(absolute_scaling_factor, dict):
        factors = [
            absolute_scaling_factor["RED"],
            absolute_scaling_factor["GREEN"],
            absolute_scaling_factor["BLUE"],
        ]
    else:
        factors = [absolute_scaling_factor] * 3
    for channel in range(3):
        rgb_channel = rgb[:, :, channel]
        max_value = np.percentile(rgb_channel, max_percentile)
        if max_value == 0:
            max_value = 1.0
        rgb_channel = rgb_channel * factors[channel]
        rgb_channel[rgb_channel > max_value] = max_value
        rgb_channel = rgb_channel / max_value * 255
        rgb[:, :, channel] = rgb_channel
    return rgb.astype("uint8")


def compute_quicklook_from_pol_data(
    data: dict[str, npt.NDArray[np.complex64] | npt.NDArray[np.float32]], conf: QuickLookConf | None = None
):
    """
    Compute quick-look from an arbitrary list of polarized data.

    Parameters
    ----------
    data : Dict[str, npt.NDArray[np.complex64] | npt.NDArray[np.float32]]
        Dictionary contained the polarization/image pairs. All images must
        have same shape. Empty data is not allowed.

    conf : QuickLookConf
        The quick-look's exporting options.

    Return
    ------
    rgb: np.NDArray[float]
        The [Naz x Nrg x 3] rgb image, possibly filtered and downsampled.
    """
    conf = conf if conf is not None else QuickLookConf()

    for key, value in data.items():
        warn_on_invalid_values(value, data_name=f"Channel {key}")

    with np.errstate(over="ignore", invalid="ignore"):
        rgb = RGBConverter.from_dict(data).to_rgb()

        if conf.apply_decimation:
            rgb = decimation(rgb, conf)

    contains_invalid_values = warn_on_invalid_values(rgb, data_name="Quicklook")
    if contains_invalid_values:
        replace_value = 0.0
        np.nan_to_num(rgb, copy=False, nan=replace_value, posinf=replace_value, neginf=replace_value)

    return scale_to_256(rgb, conf.absolute_scaling_factor, max_percentile=conf.max_percentile)


def write_quicklook_to_file(rgb: np.ndarray, file: Path):
    """RGB to BGR and write to disk"""
    bps_logger.debug(f"Writing quicklook to {file}")
    file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
