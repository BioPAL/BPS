# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import shutil
import warnings
from pathlib import Path

from setuptools import setup


def xsd_files_copy():
    """Copy XSD files for AUX-PPS validations."""
    required_stack_xsd = [
        "bio-aux-pps.xsd",
        "bio-common-types.xsd",
    ]
    package_root = Path(__file__).resolve().parent.parent
    if not package_root.is_dir():
        raise NotADirectoryError(f"package structure is broken, expected package root at {package_root}")

    internal_biomass_xsd_dir = Path("bps", "stack_processor", "xsd", "biomass-xsd")
    internal_biomass_xsd_dir.mkdir(exist_ok=True, parents=True)

    for xsd in required_stack_xsd:
        source = package_root / "xsd" / "biomass-xsd" / xsd
        destination = internal_biomass_xsd_dir / xsd

        if not source.exists():
            if destination.exists():
                warnings.warn(f"input: {source} was not found. Using {destination} instead")
                continue
            raise RuntimeError(f"Cannot find {source}")

        try:
            shutil.copyfile(source, destination)
            # pylint: disable-next=broad-exception-caught
        except Exception as exc:
            raise RuntimeError(f"{internal_biomass_xsd_dir} is not writable") from exc


xsd_files_copy()


setup()
