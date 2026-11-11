# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS nox automation utilities file
---------------------------------
"""

import shutil
from pathlib import Path

import nox

RUFF_VERSION = "0.17.4"


def run_format_fix(session: nox.Session):
    """Fix files: format and sort import"""
    session.install(f"ruff=={RUFF_VERSION}")
    session.run("python", "-m", "ruff", "format", ".")
    session.run("python", "-m", "ruff", "check", ".", "--select", "I", "--fix")


def run_check_xsd(session: nox.Session, package: str, file_list: list[str]):
    """Check that XSD inside package are aligned with those main folder"""

    package_root = Path(__file__).resolve().parent.parent

    package_xsd_dir = Path("bps", package, "xsd", "biomass-xsd")
    package_xsd_dir.mkdir(exist_ok=True, parents=True)
    reference_xsd_dir = package_root / "xsd" / "biomass-xsd"

    fail = False
    for xsd in file_list:
        reference_xsd = reference_xsd_dir / xsd
        package_xsd = package_xsd_dir / reference_xsd.name
        if not package_xsd.exists():
            session.warn(f"xsd: {xsd} is not in the package folder")
            fail = True
        elif reference_xsd.read_bytes() != package_xsd.read_bytes():
            session.warn(f"xsd: {xsd} is not aligned")
            fail = True

    if fail:
        session.error(f"xsd inside package are not aligned: run 'cd {package}; nox -s align_xsd")


def run_align_xsd(session: nox.Session, package: str, file_list: list[str]):
    """Align XSD inside package with those main folder"""

    package_root = Path(__file__).resolve().parent.parent

    package_xsd_dir = Path("bps", package, "xsd", "biomass-xsd")
    package_xsd_dir.mkdir(exist_ok=True, parents=True)
    reference_xsd_dir = package_root / "xsd" / "biomass-xsd"

    for xsd in file_list:
        reference_xsd = reference_xsd_dir / xsd
        package_xsd = package_xsd_dir / reference_xsd.name

        session.log(f"Copy {reference_xsd} onto {package_xsd}")
        shutil.copyfile(reference_xsd, package_xsd)


# Default sessions
@nox.session()
def format_fix(session: nox.Session):
    """Fix files: format and sort import"""
    run_format_fix(session)
