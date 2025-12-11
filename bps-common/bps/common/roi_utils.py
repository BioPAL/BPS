# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to manipulate Regions of Interest (ROI)
-------------------------------------------------

A ROI is defined as a tuple/list and it is encoded as follows:

  roi[0] = first azimuth index
  roi[1] = first range index
  roi[2] = azimuth index span
  roi[3] = range index span

Example
-------

  if  roi = [0, 0, 20, 20], then

    first azimuth index is 0
    last azimuth index is 19 (i.e. range(0, 20))

  Same for the range component.

"""

import numpy.typing as npt
from arepytools.io.metadata import RasterInfo

# Just a shortcut.
RegionOfInterest = tuple[int, int, int, int]


class InvalidRegionOfInterestError(ValueError):
    """Handle invalid ROIs in the stack."""


def raise_if_roi_is_invalid(raster_info: RasterInfo, roi: RegionOfInterest):
    """
    Check as to whether a ROI is a valid one with respect
    to a raster image.

    Parameters
    ----------
    raster_info: RasterInfo
        Raster metadata information.

    roi: RegionOfInterest
        The ROI that needs to be checked.

    Raises
    ------
    InvalidRegionOfInterestError

    """
    if (
        (len(roi) != 4)
        or (roi[0] < 0)
        or (roi[1] < 0)
        or (roi[2] <= 0)
        or (roi[3] <= 0)
        or (roi[0] + roi[2] > raster_info.lines)
        or (roi[1] + roi[3] > raster_info.samples)
    ):
        raise InvalidRegionOfInterestError(f"{roi} is not a valid ROI")


def crop_to_roi(
    image: npt.NDArray | None,
    roi: RegionOfInterest | None,
    *,
    subsampling_steps: tuple[int, int] = (1, 1),
) -> npt.NDArray | None:
    """
    Crop an image to a region of interest (ROI). This assumes that
    the provided ROI is a valid one.

    Parameters
    ----------
    image: Optional[npt.NDArray]
        Optionally, an [N x M] input data.

    roi: tuple[int, ...]
        The target ROI. No cropping is performed if None.

    subsampling_steps: tuple[int, int] = (1, 1)
        The subsampling steps. Defaults to no subsampling.

    Return
    ------
    Optional[npt.NDArray]
        The cropped image, if provided. None otherwise.

    """
    if image is None:
        return None

    if roi is None:
        roi = (0, 0, *image.shape)

    return image[
        roi[0] : roi[0] + roi[2] : subsampling_steps[0],
        roi[1] : roi[1] + roi[3] : subsampling_steps[1],
    ]
