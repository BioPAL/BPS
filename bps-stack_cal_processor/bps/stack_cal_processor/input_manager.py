# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Calibration Input Manager
-------------------------------
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from arepytools.io import open_product_folder, read_metadata
from arepytools.io.metadata import EPolarization
from arepytools.io.productfolder2 import ProductFolder2, is_product_folder
from arepytools.math.genericpoly import GenericPoly
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.io.common_types.models import CoregistrationMethodType
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from bps.stack_cal_processor.core.utils import (
    read_productfolder_data,
    read_productfolder_data_by_polarization,
    read_raster_info,
)

# As per BPSStackProcessor convention, in every coregistered product
# the metadata channel 0 is that of the coregistered product (that is,
# copy of the coregistration primary), while channel 1 is that of the
# original product before coregistration.
PRIMARY_METADATA_CHANNEL = 0
SECONDARY_METADATA_CHANNEL = 1


@dataclass
class StackCalProcessorInputProducts:
    """Input file of the Stack Calibration Processor apps"""

    coreg_product: Path | None = None
    """Coregistered product."""

    synth_geometry_product: Path | None = None
    """Synthetic phase from geometry (DSI)."""

    l1_iono_phase_screen_product: Path | None = None
    """The ionosphere phase screen estimated by L1."""

    l1_iono_range_shifts_product: Path | None = None
    """The range shifts due to the L1 ionosphere."""

    vertical_wavenumber_product: Path | None = None
    """Vertical wavenumbers."""

    azimuth_shifts_product: Path | None = None
    """Coregistration azimuth shifts."""

    azimuth_geo_shifts_product: Path | None = None
    """Coregistration azimuth shifts from orbit."""

    range_shifts_product: Path | None = None
    """Coregistration range shifts."""

    dist_product: Path | None = None
    """"LOS distances (0-doppler) from the coreg primary image."""

    l1a_product_name: str | None = None
    """Optionally, the name of the L1a source product."""


class InvalidStackCalInputError(RuntimeError):
    """Handle and invalid input for the stack-cal processor."""


class StackCalProcessorInputManager:
    """This class manages the input data of the calibration step."""

    def __init__(
        self,
        stack_input_products: tuple[StackCalProcessorInputProducts, ...],
        coreg_primary_image_index: int,
        coreg_actualized_parameters: tuple[dict, ...],
        polarizations: tuple[EPolarization, ...] = (
            EPolarization.hh,
            EPolarization.xx,
            EPolarization.vv,
        ),
        roi: RegionOfInterest | None = None,
    ):
        """
        Initialize a StackCalProcessorInputManager object.

        Parameters
        ----------
        stack_input_products: tuple[StackCalProcessorInputFile.CoregGeoProduct]
            The input products (Coreg images, synthetic phases etc.).

        coreg_primary_image_index: int
            The index of the coregistration primary image.

        coreg_actualized_parameters: tuple[dict, ...]
            Dictionaries (1 per frame) that contain the coregistration parameters
            and values of fitting metrics obtained during image coregistration.

        polarizations: tuple[EPolarization, ...]
            The polarization map. The images returned by the input manager class
            are sorted as this polarization tuple (i.e. image[...][i] corresponds
            to polarizations[i] for all i in 0,...,len(polarizations)-1).

        roi: RegionOfInterest | None = None
            Optionally, a ROI to restrict the calibration to a specific
            region of interest. See bps/stack_cal_processor/core/roi_utils.py
            for details on how to encode a ROI.

        Raises
        ------
        InvalidStackCalInputError

        """
        # We do not accept void data.
        if not 0 <= coreg_primary_image_index < len(stack_input_products):
            raise InvalidStackCalInputError(f"Invalid coreg primary index={coreg_primary_image_index}")
        if len(stack_input_products) < 2:
            raise InvalidStackCalInputError(
                "Need at least 2 image for calibrating the stack",
            )
        if len(polarizations) == 0:
            raise InvalidStackCalInputError(
                "Need at least 1 polarization for calibrating the stack",
            )

        # Store the coregistation primary image index.
        self._coreg_primary_image_index = coreg_primary_image_index

        # Store the coregistration methods.
        self._coreg_methods = tuple(params["coregistration_method"] for params in coreg_actualized_parameters)

        # Store the polarization map.
        self._polarizations = polarizations

        # Store the ROI.
        self.roi = roi

        # Initialize the Coreg product folders.
        self._coreg_products = tuple(
            _open_product_folder(stack_product.coreg_product) for stack_product in stack_input_products
        )

        # Possibly initialize the DSI product folders.
        self._synth_geometry_products = tuple(
            _open_product_folder(stack_product.synth_geometry_product) for stack_product in stack_input_products
        )

        # Initialize the L1 phase-screen (LUT) product folders.
        self._l1_iono_phase_screen_products = tuple(
            _open_product_folder(stack_product.l1_iono_phase_screen_product) for stack_product in stack_input_products
        )

        # Initialize the L1 range-shifts (LUT) product folders.
        self._l1_iono_range_shifts_products = tuple(
            _open_product_folder(stack_product.l1_iono_range_shifts_product) for stack_product in stack_input_products
        )

        # Initialize the vertical wavenumbers.
        self._vertical_wavenumbers_products = tuple(
            _open_product_folder(stack_product.vertical_wavenumber_product) for stack_product in stack_input_products
        )

        # Initialize the azimuth coregistration shifts.
        self._azimuth_coreg_shifts_products = tuple(
            _open_product_folder(stack_product.azimuth_shifts_product) for stack_product in stack_input_products
        )

        # Initialize the azimuth coregistration shifts from orbit (geometry only).
        self._azimuth_coreg_shifts_geo_products = tuple(
            _open_product_folder(stack_product.azimuth_geo_shifts_product) for stack_product in stack_input_products
        )

        # Initialize the azimuth coregistration shifts.
        self._range_coreg_shifts_products = tuple(
            _open_product_folder(stack_product.range_shifts_product) for stack_product in stack_input_products
        )

        # Initialize the distances.
        self._dist_products = tuple(
            _open_product_folder(stack_product.dist_product) for stack_product in stack_input_products
        )

    def compute_nodata_mask(self) -> npt.NDArray[float]:
        """
        A mask that flags where the data is valid.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        npt.NDArray[float]
            The mask (1.0: valid, np.nan: no data).

        """
        return _compute_nodata_mask(
            tuple(np.abs(read_productfolder_data(pf, roi=self.roi)) > 0.0 for pf in self._coreg_products)
        )

    def read_coreg_images(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
    ) -> tuple[tuple[npt.NDArray[complex], ...], ...]:
        """
        Read the coregistered stack images.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[tuple[npt.NDArray[complex], ...], ...]
            Coregistered stack images, packed as [Nimg x Npol] images, each of
            shape [Nazm x Nrng] and aligned on the coregistration primary grid.

        """
        return tuple(
            tuple(
                read_productfolder_data_by_polarization(pf, polarization=pol, roi=self.roi)
                for pol in self._polarizations
            )
            for pf in _select_by_indices(self._coreg_products, image_indices)
        )

    def read_synth_geometry_images(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
        bias_compensation: bool = True,
    ) -> tuple[npt.NDArray[float], ...]:
        """
        The synthetic phases from DEM (a.k.a. DSI).

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        bias_compensation: bool
            Possibly, compensate biases due to the residuals resulting from
            upsampling the DEM.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float], ...] [rad]
            The synthetic phases from DEM, each of shape [Nazm x Nrng]
            and aligned to the coregistration primary grid. [rad]

        """
        if image_indices is None:
            image_indices = list(range(len(self._synth_geometry_products)))

        synth_phases = tuple(
            read_productfolder_data(synth_pf, roi=self.roi)
            for synth_pf in _select_by_indices(self._synth_geometry_products, image_indices)
        )

        # Whether the coregistration primary was already loaded.
        has_coreg_primary = self._coreg_primary_image_index in image_indices

        # If requested, remove the biases due to the interpolation residuals of
        # related to the DEM upsampling.
        phase_bias = 0
        if bias_compensation:
            phase_bias = (
                read_productfolder_data(
                    self._synth_geometry_products[self._coreg_primary_image_index],
                    roi=self.roi,
                )
                if not has_coreg_primary
                else synth_phases[self._coreg_primary_image_index]
            )

        synth_phases = list(phi - phase_bias for phi in synth_phases)

        # DSI's for the coregistration primary are always 0.
        if has_coreg_primary:
            synth_phases[self._coreg_primary_image_index] = np.zeros_like(synth_phases[self._coreg_primary_image_index])

        return tuple(synth_phases)

    def read_vertical_wavenumber_images(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
    ) -> tuple[npt.NDArray[float], ...]:
        """
        Read the vertical wavenumbers.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float], ...] [rad/m]
            The vertical wavenumbers, each of shape [Nazm x Nrng] and aligned
            to the coregistration primary grid.

        """
        # NOTE: Vertical wavenumbers are independent from polarization, so we
        # need to 1 per image.

        # Vertical wavenumbers are constant 0's for the primary image, so it may
        # not be output at all by the preprocessor. We set the correct policy in
        # case that happens.
        return tuple(
            read_productfolder_data(wvn_pf, roi=self.roi)
            for wvn_pf in _select_by_indices(self._vertical_wavenumbers_products, image_indices)
        )

    def read_range_coreg_shifts(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
        bias_compensation: bool = True,
    ) -> tuple[npt.NDArray[float], ...]:
        """
        Read the range coregistration shifts.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        bias_compensation: bool
            Possibly, compensate biases due to the residuals resulting from
            upsampling the DEM.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float], ...]
            The coregisteration range shifts, each of shape [Nazm x Nrng] and
            aligned to the coreigstration primary grid.

        """
        # Check if we need to read the coreg primary or not. So we avoid read
        # it twice
        if image_indices is None:
            image_indices = list(range(len(self._range_coreg_shifts_products)))

        # Whether the selected indices contain the coreg primary or not.
        has_coreg_primary = self._coreg_primary_image_index in image_indices

        # Read the shifts.
        coreg_rng_shifts = tuple(
            read_productfolder_data(rng_shifts_pf, roi=self.roi)
            for rng_shifts_pf in _select_by_indices(self._range_coreg_shifts_products, image_indices)
        )

        # The range shifts bias. 0 by default, unless we require bias
        # compensation. Only under geometry-based coregistration makes sense to
        # remove biases.
        rng_shifts_bias = defaultdict(float)
        if bias_compensation:
            rng_shifts_bias[CoregistrationMethodType.GEOMETRY] = (
                read_productfolder_data(
                    self._range_coreg_shifts_products[self._coreg_primary_image_index],
                    roi=self.roi,
                )
                if not has_coreg_primary
                else coreg_rng_shifts[self._coreg_primary_image_index]
            )
            rng_shifts_bias[CoregistrationMethodType.GEOMETRY] -= _null_range_shifts(
                rng_shifts_bias[CoregistrationMethodType.GEOMETRY].shape
            )

        # Possibly, apply the bias.
        cor_rng_shifts = [
            cor_rg - rng_shifts_bias[coreg_method]
            for cor_rg, coreg_method in zip(
                coreg_rng_shifts,
                _select_by_indices(self._coreg_methods, image_indices),
            )
        ]

        # Coregistration shifts for the primary are always null.
        if has_coreg_primary:
            cor_rng_shifts[self._coreg_primary_image_index] = _null_range_shifts(
                cor_rng_shifts[self._coreg_primary_image_index].shape
            )

        return tuple(cor_rng_shifts)

    def read_azimuth_coreg_shifts(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
        bias_compensation: bool = True,
    ) -> tuple[npt.NDArray[float], ...]:
        """
        Read the azimuth coregistration shifts.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        bias_compensation: bool
            Possibly, compensate biases due to the residuals resulting from
            upsampling the DEM.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float], ...]
            The coregisteration azimuth shifts, each of shape [Nazm x Nrng] and
            aligned to the coreigstration primary grid.

        """
        # Check if we need to read the coreg primary or not. So we avoid read
        # it twice
        if image_indices is None:
            image_indices = list(range(len(self._azimuth_coreg_shifts_products)))

        # Whether the selected indices contain the coreg primary or not.
        has_coreg_primary = self._coreg_primary_image_index in image_indices

        # Read the shifts.
        coreg_azm_shifts = tuple(
            read_productfolder_data(azm_shifts_pf, roi=self.roi)
            for azm_shifts_pf in _select_by_indices(self._azimuth_coreg_shifts_products, image_indices)
        )

        # The range shifts bias. 0 by default, unless we require bias
        # compensation. Only under geometry-based coregistration makes sense to
        # remove biases.
        azm_shifts_bias = defaultdict(float)
        if bias_compensation:
            azm_shifts_bias[CoregistrationMethodType.GEOMETRY] = (
                read_productfolder_data(
                    self._range_coreg_shifts_products[self._coreg_primary_image_index],
                    roi=self.roi,
                )
                if not has_coreg_primary
                else coreg_azm_shifts[self._coreg_primary_image_index]
            )
            azm_shifts_bias[CoregistrationMethodType.GEOMETRY] -= _null_range_shifts(
                azm_shifts_bias[CoregistrationMethodType.GEOMETRY].shape
            )

        # Possibly, apply the bias.
        cor_azm_shifts = [
            cor_rg - azm_shifts_bias[coreg_method]
            for cor_rg, coreg_method in zip(
                coreg_azm_shifts,
                _select_by_indices(self._coreg_methods, image_indices),
            )
        ]

        # Coregistration shifts for the primary are always null.
        if has_coreg_primary:
            cor_azm_shifts[self._coreg_primary_image_index] = _null_range_shifts(
                cor_azm_shifts[self._coreg_primary_image_index].shape
            )

        return tuple(cor_azm_shifts)

    def read_azimuth_residual_shifts(
        self, *, image_indices: tuple[int, ...] | None = None
    ) -> tuple[npt.NDArray[float], ...]:
        """
        Read the coregistration azimuth residual shifts (in pixels).

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float], ...] [px]
            The residual shifts from data.

        """
        azimuth_residual_shifts = []
        for coreg_method, cor_az_pf, cor_geo_az_pf in zip(
            _select_by_indices(self._coreg_methods, image_indices),
            _select_by_indices(self._azimuth_coreg_shifts_products, image_indices),
            _select_by_indices(self._azimuth_coreg_shifts_geo_products, image_indices),
        ):
            num_lines, num_samples = _get_raster_shape(cor_az_pf, roi=self.roi)
            if coreg_method is CoregistrationMethodType.GEOMETRY:
                azimuth_residual_shifts.append(np.zeros((num_lines, num_samples)))
            elif coreg_method is CoregistrationMethodType.GEOMETRY_AND_DATA:
                azimuth_residual_shifts.append(
                    read_productfolder_data(cor_az_pf, roi=self.roi)
                    - read_productfolder_data(cor_geo_az_pf, roi=self.roi)
                )
        return tuple(azimuth_residual_shifts)

    def read_distances(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
    ) -> tuple[npt.NDArray[float], ...]:
        """
        Read the LOS absolute 0-doppler distances from the primary trajectory
        to the primary ECEF grid.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float], ...] [m]
            The distance products.

        """
        return tuple(
            read_productfolder_data(dist_pf, roi=self.roi)
            for dist_pf in _select_by_indices(self._dist_products, image_indices)
        )

    def read_l1_iono_phase_screens_luts(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
    ) -> tuple[npt.NDArray[float] | None, ...]:
        """
        Read the ionospheric phase screen (from L1). Phase screens
        are cached from LUTs.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float] | None, ...] [rad]
            The phase-screens from L1, each of shape [Nazm x Nrng] and aligned
            to the coregistration primary grid.

        """
        return tuple(
            read_productfolder_data(iono_pf, roi=self.roi) if iono_pf is not None else None
            for iono_pf in _select_by_indices(self._l1_iono_phase_screen_products, image_indices)
        )

    def read_l1_iono_range_shifts_luts(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
    ) -> tuple[npt.NDArray[float] | None, ...]:
        """
        Read the ionospheric range shifts (from L1). Range shifts
        are cached from LUTs.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        tuple[npt.NDArray[float] | None, ...] [m]
            The iono range shifts from L1, each of shape [Nazm x Nrng] and
            aligned to the coregistration primary grid.

        """
        return tuple(
            read_productfolder_data(iono_pf, roi=self.roi) if iono_pf is not None else None
            for iono_pf in _select_by_indices(self._l1_iono_range_shifts_products, image_indices)
        )

    def doppler_centroids(
        self,
        *,
        image_indices: tuple[int, ...] | None = None,
    ) -> npt.NDArray[float]:
        """
        Compute the Doppler centroids.

        Parameters
        ----------
        image_indices: tuple[int, ...] | None = None
            Optionally, load a subset of images. It defaults to all.

        Raises
        ------
        InvalidStackCalInputError

        Return
        ------
        npt.NDArray[float] [Hz]
             The frequencies of the doppler centroids, packed as a single
             [Nimg x Nrng] array.
        """
        if not (
            len(self._coreg_products)
            == len(self._azimuth_coreg_shifts_products)
            == len(self._range_coreg_shifts_products)
        ):
            raise InvalidStackCalInputError("CSC and Coreg Shift products have mismatching dimensions")

        doppler_centroids = []
        for coreg_product, azm_shifts_product, rng_shifts_product in zip(
            _select_by_indices(self._coreg_products, image_indices),
            _select_by_indices(self._azimuth_coreg_shifts_products, image_indices),
            _select_by_indices(self._range_coreg_shifts_products, image_indices),
        ):
            # The raster info of the coregistered data.
            metadata = read_metadata(coreg_product.get_channel_metadata(1))

            # If we are working on the primary, there's only 1 channel.
            primary_metadata_channel = PRIMARY_METADATA_CHANNEL
            secondary_metadata_channel = SECONDARY_METADATA_CHANNEL
            if metadata.get_number_of_channels() < 2:
                secondary_metadata_channel = primary_metadata_channel

            primary_raster_info = metadata.get_raster_info(primary_metadata_channel)
            secondary_raster_info = metadata.get_raster_info(secondary_metadata_channel)

            # The central azimuth row wrt the selected ROI.
            central_azimuth_line = int(primary_raster_info.lines / 2)
            if self.roi is not None:
                central_azimuth_line = self.roi[0] + round(self.roi[2] / 2)

            # Express the central azimuth row as a 1-row ROI.
            central_azimuth_row = [
                central_azimuth_line,
                0,
                1,  # take only 1 line.
                primary_raster_info.samples,
            ]
            raise_if_roi_is_invalid(primary_raster_info, central_azimuth_row)

            # Read the coregistration shifts.
            azimuth_coreg_shifts = read_productfolder_data(azm_shifts_product, roi=central_azimuth_row)
            range_coreg_shifts = read_productfolder_data(rng_shifts_product, roi=central_azimuth_row)

            # fmt: off
            assert (
                azimuth_coreg_shifts.shape == (1, primary_raster_info.samples)
            ), f"{azimuth_coreg_shifts.shape=} is not (1, {primary_raster_info.samples})"
            assert (
                range_coreg_shifts.shape == (1, primary_raster_info.samples)
            ), f"{range_coreg_shifts.shape=} is not (1, {primary_raster_info.samples})"
            # fmt: on

            # The azimuth and range times for the secondary product, associated
            # to the central azimuth row of the primary product.
            azimuth_times = azimuth_coreg_shifts * secondary_raster_info.lines_step + secondary_raster_info.lines_start
            range_times = range_coreg_shifts * secondary_raster_info.samples_step + secondary_raster_info.samples_start

            # The channel associated to the secondary image before coregistration.
            doppler_centroids_product = None
            try:
                doppler_centroids_poly_vector = metadata.get_doppler_centroid(secondary_metadata_channel)
                doppler_centroids_poly = doppler_centroids_poly_vector.get_poly(0)

                doppler_centroids_poly_gen = GenericPoly(
                    (
                        PreciseDateTime().set_from_utc_string(str(doppler_centroids_poly.t_ref_az)),
                        doppler_centroids_poly.t_ref_rg,
                    ),
                    doppler_centroids_poly.coefficients,
                    list(
                        zip(
                            doppler_centroids_poly.get_powers_x(),
                            doppler_centroids_poly.get_powers_y(),
                        )
                    ),
                )

                doppler_centroids_product = np.array(
                    doppler_centroids_poly_gen.evaluate((azimuth_times, range_times))
                ).reshape(-1)
                # pylint: disable-next=broad-exception-caught
            except Exception:
                bps_logger.warning(
                    "Could not initialize the doppler centroids for image '%s'. Setting them to 0's",
                    coreg_product.pf_dir_path,
                )
                doppler_centroids_product = np.zeros(primary_raster_info.samples, dtype=np.float32)

            doppler_centroids.append(doppler_centroids_product)

        return np.array(doppler_centroids, dtype=np.float32)

    def get_coreg_products(self) -> tuple[ProductFolder2, ...]:
        """Get the coregistration product folder."""
        return self._coreg_products


def select_calibration_reference_image(
    *,
    coreg_primary_image_index: int,
    polarization: EPolarization | None = None,
    reference: int | None = None,
    rfi_indices: tuple[dict[EPolarization, float], ...] | None = None,
    faraday_decorrelation_indices: tuple[float, ...] | None = None,
    input_stack_paths: list[Path],
) -> int:
    """
    Select the calibration reference image.

    Parameters
    ----------
    polarization: EPolarization | None
        Optionally, a polarization selected by the user. Only needed when
        `rfi_indices` is not None.

    reference: int | None
        Optionally, the reference image index selected by the user.

    rfi_indices: tuple[dict[EPolarization, float], ...] | None
        Optionally, the RFI indices from the preprocessor.

    faraday_decorrelation_indices: tuple[float, ...] | None
        Optionally, the Faraday decorrelations from the preprocessor.

    coreg_primary_image_index: int
        The coregistration primary image index. This is used as a fallback
        in case all other selection methods fail.

    input_stack_paths: list[Path]
        The paths to the input stack.

    Raises
    ------
    InvalidStackCalInputError

    Return
    ------
    int
        The index of the selected reference image.

    """
    # Minimal checks on inputs.
    num_images = len(input_stack_paths)
    if rfi_indices is not None and len(rfi_indices) != num_images:
        raise InvalidStackCalInputError("RFIs and stack images have mismatching dimensions")
    if rfi_indices is not None and polarization is None:
        raise InvalidStackCalInputError("a selected polarization is required when RFIs are provided")
    if faraday_decorrelation_indices is not None and len(faraday_decorrelation_indices) != num_images:
        raise InvalidStackCalInputError("Faraday Rotation indices and stack images have mismatching dimensions")

    # If the user has provided a preference, the user rules.
    if reference is not None:
        bps_logger.info(
            "Using provided calibration reference image %s (index=%d)",
            input_stack_paths[reference].name,
            reference,
        )
        return reference

    # The default value.
    calib_reference_image_index = coreg_primary_image_index
    if rfi_indices is None and faraday_decorrelation_indices is None:
        bps_logger.info(
            "Using coreg primary as calibration reference image %s (index=%d)",
            input_stack_paths[calib_reference_image_index].name,
            calib_reference_image_index,
        )
        return calib_reference_image_index

    # Select depending on the RFI and Faraday score.
    score = np.ones(num_images)
    if rfi_indices is not None:
        score *= np.array([rfi[polarization] for rfi in rfi_indices])
    if faraday_decorrelation_indices is not None:
        score *= np.array(faraday_decorrelation_indices)

    # Singular case, all scores are invalid.
    if np.all(np.isnan(score)):
        bps_logger.warning(
            "RFIs and Faraday rotations result in all NaNs, falling back to using coreg primary %s (index=%d)",
            input_stack_paths[calib_reference_image_index].name,
            calib_reference_image_index,
        )
        return calib_reference_image_index

    # Take the image with best score.
    calib_reference_image_index = np.nanargmax(score)
    bps_logger.info(
        "Using calibration reference image %s (index=%d) from RFI and/or Faraday rotations",
        input_stack_paths[calib_reference_image_index].name,
        calib_reference_image_index,
    )
    return calib_reference_image_index


def _compute_nodata_mask(stack_masks: tuple[npt.NDArray, ...]) -> npt.NDArray[float]:
    """
    Compute the no-data mask.

    Parameters
    ----------
    stack_masks: tuple[npt.NDArra], ...]
        The stack images non-zero boolean masks.

    Raises
    ------
    AssertionError

    Return
    ------
    npt.NDArray[float]
        Return the mask (1.0: data, np.nan: no-data).

    """
    assert len(stack_masks) > 0, "stack's masks are empty"
    mask = np.full(stack_masks[0].shape, np.nan)
    mask[reduce(lambda agg, m: agg & m, stack_masks)] = 1.0
    return mask


def _open_product_folder(pf_path: Path | None) -> ProductFolder2 | None:
    """Optionally open a product folder."""
    if pf_path is None or not is_product_folder(pf_path):
        return None
    return open_product_folder(pf_path)


def _select_by_indices(
    collection: Iterable[Any],
    indices: Iterable[int] | None,
) -> Iterable[Any]:
    """Filter the input collection using the indices."""
    if indices is None:
        return collection
    return (collection[i] for i in indices)


def _get_raster_shape(pf: ProductFolder2, roi: RegionOfInterest | None) -> tuple[int, int]:
    """Get the shape of the raster, with optionally a ROI."""
    if roi is None:
        raster_info = read_raster_info(pf)
        return (raster_info.lines, raster_info.samples)
    return roi[2:]


def _null_range_shifts(shape: tuple[int, int]) -> npt.NDArray[float]:
    """The null range shifts."""
    return np.ones((shape[0], 1)) * np.arange(shape[1])
