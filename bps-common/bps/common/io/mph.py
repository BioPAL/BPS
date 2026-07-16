# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
MPH (Main Product Header) utilities
------------------------------------
"""

from pathlib import Path


def get_mph_path(product_dir: Path) -> Path:
    """Return the path of the MPH file inside a BIOMASS product directory.

    The MPH file follows the convention ``<product_name_lowercase>.xml``
    and resides directly inside the product directory.

    Parameters
    ----------
    product_dir : Path
        Path to the product directory.

    Returns
    -------
    Path
        Full path to the MPH file.
    """
    return product_dir / (product_dir.name.lower() + ".xml")
