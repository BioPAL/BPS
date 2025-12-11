# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS common
----------
"""

from enum import Enum
from pathlib import Path


class Swath(Enum):
    """BPS swaths"""

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    RO = "RO"


class Polarization(Enum):
    """BPS polarizations"""

    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"


class MissionPhaseID(Enum):
    """ID of the Biomass Mission Phase"""

    INTERFEROMETRIC = "int"
    TOMOGRAPHIC = "tom"
    COMMISSIONING = "com"


STRIPMAP_SWATHS = [Swath.S1, Swath.S2, Swath.S3]

AcquisitionMode = tuple[Swath, MissionPhaseID | None]
ACQUISITION_MODES: tuple[AcquisitionMode, ...] = (
    (Swath.S1, MissionPhaseID.INTERFEROMETRIC),
    (Swath.S2, MissionPhaseID.INTERFEROMETRIC),
    (Swath.S3, MissionPhaseID.INTERFEROMETRIC),
    (Swath.S1, MissionPhaseID.TOMOGRAPHIC),
    (Swath.S2, MissionPhaseID.TOMOGRAPHIC),
    (Swath.S3, MissionPhaseID.TOMOGRAPHIC),
    (Swath.S1, MissionPhaseID.COMMISSIONING),
    (Swath.S2, MissionPhaseID.COMMISSIONING),
    (Swath.S3, MissionPhaseID.COMMISSIONING),
    (Swath.RO, None),
)


class InvalidAuxProduct(RuntimeError):
    """Raised when the input aux product is invalid"""


def retrieve_aux_product_data_dir(product: Path) -> Path:
    """Retrieve the path to the 'product/data' directory

    Parameters
    ----------
    product : Path
        path to the product

    Returns
    -------
    Path
        path to the data dir
    """
    return product.joinpath("data")


def retrieve_aux_product_data_content(product: Path) -> set[Path]:
    """Retrieve the path to the files inside 'product/data' directory

    Parameters
    ----------
    product : Path
        path to the product

    Returns
    -------
    Set[Path]
        path to the content file

    Raises
    ------
    InvalidAuxProduct
        in case multiple files are found in the 'data/' folder
    """
    return set(retrieve_aux_product_data_dir(product).iterdir())


def retrieve_aux_product_data_single_content(product: Path) -> Path:
    """Retrieve the path to the file inside 'product/data' directory

    A single file is expected to be present in the 'product/data' directory

    Parameters
    ----------
    product : Path
        path to the product

    Returns
    -------
    Path
        path to the content file

    Raises
    ------
    InvalidAuxProduct
        in case multiple files are found in the 'data/' folder
    """
    data_folder = retrieve_aux_product_data_dir(product)
    content = list(data_folder.iterdir())
    if len(content) != 1:
        raise InvalidAuxProduct(f"Invalid {product} product: multiple files found in '{data_folder}' folder")
    return content[0]


__all__ = [
    "Swath",
    "Polarization",
    "MissionPhaseID",
    "STRIPMAP_SWATHS",
    "ACQUISITION_MODES",
    "AcquisitionMode",
    "InvalidAuxProduct",
    "retrieve_aux_product_data_content",
    "retrieve_aux_product_data_single_content",
]
