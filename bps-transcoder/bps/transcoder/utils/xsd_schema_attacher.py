# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to copy XSD inside products
-------------------------------------
"""

import importlib.resources
import shutil
from pathlib import Path

import bps.transcoder


def copy_biomass_xsd_files(destination_dir: Path, xsd_names: list[str]):
    """Copy all xsd files to a folder"""
    main_folder = importlib.resources.files(bps.transcoder)
    xsd_folder = Path(main_folder).joinpath("xsd", "biomass-xsd")

    for source, destination in [(xsd_folder.joinpath(name), destination_dir.joinpath(name)) for name in xsd_names]:
        shutil.copyfile(source, destination)
