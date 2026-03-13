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
from bps.common.io.common import ColourCodingMethod


@dataclass
class QuickLookConf:
    """Quicklook configuration"""

    l1a_colour_coding_method: ColourCodingMethod | None = ColourCodingMethod.PAULI
    l1b_colour_coding_method: ColourCodingMethod | None = ColourCodingMethod.LEXICOGRAPHIC
    l1c_colour_coding_method: ColourCodingMethod | None = ColourCodingMethod.PAULI
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

        for key, value in data.items():
            warn_on_invalid_values(value, data_name=f"Channel {key}")

        data_shape = data.get("H/H").shape if "H/H" in data else data.get("V/V").shape
        return cls(
            hh=data.get("H/H", np.zeros(shape=data_shape, dtype=dtype)),
            hv=data.get("H/V", np.zeros(shape=data_shape, dtype=dtype)),
            vh=data.get("V/H", np.zeros(shape=data_shape, dtype=dtype)),
            vv=data.get("V/V", np.zeros(shape=data_shape, dtype=dtype)),
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

    def pauli_decomposition(self, conf: QuickLookConf) -> npt.NDArray[np.uint8]:
        """Compute RGB via Pauli decomposition"""
        rgb = np.zeros(self.shape + (3,), np.float32)

        rgb[:, :, 0] = np.abs(self.hh - self.vv)
        rgb[:, :, 1] = np.abs(self.hv + self.vh)
        if self.xx is not None:
            rgb[:, :, 1] += 2 * np.abs(self.xx)
        rgb[:, :, 2] = np.abs(self.hh + self.vv)

        if conf.apply_decimation:
            rgb = decimation(rgb, conf)

        contains_invalid_values = warn_on_invalid_values(rgb, data_name="Quicklook")
        if contains_invalid_values:
            replace_value = 0.0
            np.nan_to_num(rgb, copy=False, nan=replace_value, posinf=replace_value, neginf=replace_value)

        return scale_to_256(rgb, conf.absolute_scaling_factor, max_percentile=conf.max_percentile)

    def lexicographic_representation(self, conf: QuickLookConf) -> npt.NDArray[np.uint8]:
        """Compute RGB via lexicographic representation"""
        rgb = np.zeros(self.shape + (3,), np.float32)

        rgb[:, :, 0] = np.abs(self.hh)
        rgb[:, :, 1] = np.sqrt(self.hv**2 + self.vh**2) * 0.5
        if self.xx is not None:
            rgb[:, :, 1] += np.abs(self.xx)
        rgb[:, :, 2] = np.abs(self.vv)

        if conf.apply_decimation:
            rgb = decimation(rgb, conf)

        contains_invalid_values = warn_on_invalid_values(rgb, data_name="Quicklook")
        if contains_invalid_values:
            replace_value = 0.0
            np.nan_to_num(rgb, copy=False, nan=replace_value, posinf=replace_value, neginf=replace_value)

        return scale_to_256(rgb, conf.absolute_scaling_factor, max_percentile=conf.max_percentile)

    def hsv_representation(self, conf: QuickLookConf) -> npt.NDArray[np.uint8]:
        """Compute RGB via HSV representation"""

        def multilook(carr: np.ndarray, ws: tuple) -> np.ndarray:
            """
            Returns multilooked values of given array (block average).

            Args:
                carr : complex np.ndarray, shape (ny, nx)
                ws   : tuple (wy, wx) multilook window
            """
            ny, nx = carr.shape
            wy, wx = ws
            my, mx = ny // wy, nx // wx

            return carr[: my * wy, : mx * wx].reshape(my, wy, mx, wx).mean(axis=(1, 3))

        def get_covariance_matrix(
            HH: np.ndarray, HV: np.ndarray, VH: np.ndarray, VV: np.ndarray, ws: tuple
        ) -> np.ndarray:
            """
            Calculates multilooked covariance matrix from SLCs.

            C_ij = < k_i * conj(k_j) >
            """
            ny, nx = HH.shape
            wy, wx = ws
            my, mx = ny // wy, nx // wx

            k = np.stack((HH, VH, HV, VV))  # shape (4, ny, nx)
            k_conj = np.conj(k)

            cvx = np.empty((my, mx, 4, 4), dtype=np.complex64)
            for i in range(4):
                for j in range(4):
                    cvx[:, :, i, j] = multilook(k[i] * k_conj[j], ws)

            return cvx

        def colormap(x: np.ndarray) -> np.ndarray:
            """
            Maps normalized values [0,1] to Turbo RGB colors.

            Args:
                x : ndarray of floats in [0,1], any shape

            Returns:
                ndarray of shape (*x.shape, 3) with RGB floats in [0,1]
            """
            x = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)

            col_bgr = cv2.applyColorMap(x, cv2.COLORMAP_TURBO)
            col_rgb = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB)

            return np.squeeze(col_rgb) / 255.0

        def create_hsv_colormap(h, s, v):
            """
            Makes an HSV-like RGB representation based on the given colormap instead
            of 'hsv' colormap.

            See https://en.wikipedia.org/wiki/HSL_and_HSV

            Parameters
            ----------
            h : ndarray
                Hue values. Usually between 0 and 1.0.
            s : ndarray
                Saturation values. Between 0 and 1.0.
            v : ndarray
                Value values. Between 0 and 1.0.

            Returns
            -------
            rgb: ndarray
                An array with the same shape as input + (3,) representing the RGB.
            """
            # Generate color between given colormap (colormap(h)) and white (ones)
            # according to the given saturation
            return v[..., np.newaxis] * ((1 - s)[..., np.newaxis] * np.ones(3) + s[..., np.newaxis] * colormap(h))  # type: ignore

        # generate covariance matrix.
        if self.xx is not None:
            self.hv = self.xx
            self.vh = np.zeros_like(self.vh)
        cvx = get_covariance_matrix(
            self.hh, self.hv, self.vh, self.vv, (conf.azimuth_averaging_factor, conf.range_averaging_factor)
        )

        # convert to Pauli basis
        U = np.matrix([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -1, 1, 0]]) / np.sqrt(2)  # to Pauli bais
        tvx = np.einsum("ij,...jk,kl", U, cvx, U.H)  # coherency matrix

        # HSV
        TP = np.sqrt(np.einsum("...ii", tvx).real)  # amplitude from total power

        # hue of image
        alpha = np.arccos(np.sqrt(tvx[..., 0, 0].real) / (TP + 1e-8))  # alpha1
        # NOTE: instead of mean alpha, the alpha angle from the first column is used here.
        alpha /= 85 * np.pi / 180  # normalise to 85 deg
        h = (1 + np.sin(np.pi * (alpha - 1 / 2))) / 2  # sinusoidal hue adjustment

        # brightness of image (amplitude)
        scale_factor = 2 * np.sqrt(0.5)
        amp = np.clip(TP, 0, scale_factor * np.mean(TP))
        v = amp / np.max(amp)  # normalise

        # saturation of image
        s = np.ones_like(alpha)  # saturation is set to constant one

        # hsv to rgb
        rgb = create_hsv_colormap(h, s, v)

        # rescale to 0-255 uint8
        rgb = np.uint8(rgb * 255)

        return rgb

    def to_rgb_l1ab(self, conf: QuickLookConf) -> dict[ColourCodingMethod, npt.NDArray[np.uint8]]:
        """Convert quad pol complex data to RGB channels"""
        if self.is_complex:
            if conf.l1a_colour_coding_method == ColourCodingMethod.PAULI:
                bps_logger.debug("Converting complex data to RGB using Pauli decomposition.")
                return {ColourCodingMethod.PAULI: self.pauli_decomposition(conf)}

            if conf.l1a_colour_coding_method == ColourCodingMethod.HSV:
                bps_logger.debug("Converting complex data to RGB using HSV representation.")
                return {ColourCodingMethod.HSV: self.hsv_representation(conf)}
            if conf.l1a_colour_coding_method == ColourCodingMethod.PAULI_AND_HSV:
                bps_logger.debug(
                    "Converting complex data to RGB using both Pauli decomposition and HSV representation."
                )
                return {
                    ColourCodingMethod.PAULI: self.pauli_decomposition(conf),
                    ColourCodingMethod.HSV: self.hsv_representation(conf),
                }

            raise RuntimeError("Colour coding method not allowed for L1a product quicklook.")

        if conf.l1b_colour_coding_method == ColourCodingMethod.LEXICOGRAPHIC:
            bps_logger.debug("Converting real data to RGB using lexicographic representation.")
            return {ColourCodingMethod.LEXICOGRAPHIC: self.lexicographic_representation(conf)}

        raise RuntimeError("Colour coding method not allowed for L1b product quicklook.")

    def to_rgb_l1c(self, conf: QuickLookConf) -> dict[ColourCodingMethod, npt.NDArray[np.uint8]]:
        """Convert quad pol complex data to RGB channels"""
        if conf.l1c_colour_coding_method == ColourCodingMethod.PAULI:
            bps_logger.debug("Converting complex data to RGB using Pauli decomposition.")
            return {ColourCodingMethod.PAULI: self.pauli_decomposition(conf)}

        if conf.l1c_colour_coding_method == ColourCodingMethod.HSV:
            bps_logger.debug("Converting complex data to RGB using HSV representation.")
            return {ColourCodingMethod.HSV: self.hsv_representation(conf)}

        if conf.l1c_colour_coding_method == ColourCodingMethod.PAULI_AND_HSV:
            bps_logger.debug("Converting complex data to RGB using both Pauli decomposition and HSV representation.")
            return {
                ColourCodingMethod.PAULI: self.pauli_decomposition(conf),
                ColourCodingMethod.HSV: self.hsv_representation(conf),
            }

        raise RuntimeError("Colour coding method not allowed for L1c product quicklook.")


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
    data: dict[str, npt.NDArray[np.complex64] | npt.NDArray[np.float32]],
    conf: QuickLookConf | None = None,
    l1_product_type: str = "l1ab",
) -> dict[ColourCodingMethod, npt.NDArray[np.uint8]]:
    """
    Compute quick-look from an arbitrary list of polarized data.

    Parameters
    ----------
    data : Dict[str, npt.NDArray[np.complex64] | npt.NDArray[np.float32]]
        Dictionary contained the polarization/image pairs. All images must
        have same shape. Empty data is not allowed.

    conf : QuickLookConf
        The quick-look's exporting options.

    l1_product_type : {"l1ab","l1c"}, default="l1ab"
        The L1a product type.

    Return
    ------
    dict[ColourCodingMethod, npt.NDArray[np.uint8]]
        Dictionary with [Naz x Nrg x 3] rgb images, possibly filtered and downsampled,
        one for each colour coding method selected.
    """
    conf = conf if conf is not None else QuickLookConf()

    with np.errstate(over="ignore", invalid="ignore"):
        if l1_product_type == "l1ab":
            return RGBConverter.from_dict(data).to_rgb_l1ab(conf)
        elif l1_product_type == "l1c":
            return RGBConverter.from_dict(data).to_rgb_l1c(conf)

    raise ValueError(f"Unsupported {l1_product_type=}")


def compute_coherence_quicklook(
    coherence: npt.NDArray[float], data_mask: npt.NDArray[bool], conf: QuickLookConf | None = None
) -> npt.NDArray[np.uint8]:
    """
    Convert a coherence into a grayscale image.

    Parameters
    ----------
    coherence: npt.NDArray[float]
        The [Naz x Nrg] coherence image (in 0 - 1).

    nodata_mask: npt.NDArray[bool]
        The data mask.

    conf: QuickLookConf | None = None
        The quick-look's exporting options.

    Return
    ------
    npt.NDArray[np.uint8]
        The [Naz x Nrg] grayscale image, possibly downsampled.

    """
    conf = conf if conf is not None else QuickLookConf()
    coherence = (
        coherence[:: conf.azimuth_decimation_factor, :: conf.range_decimation_factor]
        * data_mask[:: conf.azimuth_decimation_factor, :: conf.range_decimation_factor]
    )
    return np.round(np.clip(coherence, 0, 1) * 255).astype(np.uint8)


def compute_interferogram_quicklook(
    interferogram: npt.NDArray[float],
    nodata_mask: npt.NDArray[bool],
    conf: QuickLookConf | None = None,
) -> npt.NDArray[float]:
    """
    Convert an interferogram into a JET image.

    Parameters
    ----------
    coherence: npt.NDArray[float]  [rad]
        The [Naz x Nrg] inteferogram normalized between -pi / +pi.

    nodata_mask: npt.NDArray[bool]
        The data mask.

    conf: QuickLookConf | None = None
        The quick-look's exporting options.

    Return
    ------
    npt.NDArray[np.uint8]
        The [Naz x Nrg] grayscale image, possibly downsampled.

    """
    conf = conf if conf is not None else QuickLookConf()
    interferogram = interferogram[:: conf.azimuth_decimation_factor, :: conf.range_decimation_factor]
    return (
        cv2.applyColorMap(
            np.round((interferogram - np.pi) / (2 * np.pi) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        * nodata_mask[:: conf.azimuth_decimation_factor, :: conf.range_decimation_factor, np.newaxis]
    )


def write_quicklook_to_file(rgb: np.ndarray, file: Path):
    """RGB to BGR and write to disk the l1a/b/c quick-look image"""
    bps_logger.debug(f"Writing quicklook to {file}")
    file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_coherence_quicklook_to_file(gray: np.ndarray, file: Path):
    """Write the coherence quick-look to disk as a gray-scale image"""
    bps_logger.debug(f"Writing coherence quicklook to {file}")
    file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file), gray)


def write_interferogram_quicklook_to_file(rgb: np.ndarray, file: Path):
    """Write the interferogram quick-look to disk using JET color scheme"""
    bps_logger.debug(f"Writing interferogram quicklook to {file}")
    file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file), rgb)
