# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS nox automation utilities file
---------------------------------
"""

import shutil
import sys
from pathlib import Path

import nox

UNITTEST_PYVERSIONS = ["3.12"]
CONDABUILD_BACKEND = "mamba"
WIN32 = sys.platform == "win32"
PLATFORM = "win" if WIN32 else "linux"
GDAL_VERSION = "3.10.2"
RUFF_VERSION = "0.15.7"

_LICENSE_HEADER_ARESYS = """# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""

_GPP_FILES = ["main_autofocus_bps.py", "af_lib.py", "signalproclib.py", "save_aresys.py"]

_LICENSE_HEADER_GPP = """# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l., DLR, Deimos Space
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""


def run_format_check(session: nox.Session):
    """Check files are: formatted, with license header and import sorted"""
    session.install(f"ruff=={RUFF_VERSION}")
    session.run("python", "-m", "ruff", "format", ".", "--check")
    session.run("python", "-m", "ruff", "check", ".", "--select", "I")

    def is_gpp_file(file: Path) -> bool:
        return file.name in _GPP_FILES

    def wrong_license_header(file: Path, license_header: str) -> bool:
        num_header_lines = len(license_header.splitlines())
        with file.open("r", encoding="utf-8") as f:
            header = "".join(next(f) for _ in range(num_header_lines))
        return header != license_header

    source_files = list(Path("bps").rglob("*.py"))
    no_licensed_files = list(
        filter(
            lambda x: wrong_license_header(x, _LICENSE_HEADER_ARESYS),
            filter(lambda x: not is_gpp_file(x), source_files),
        )
    ) + list(
        filter(
            lambda x: wrong_license_header(x, _LICENSE_HEADER_GPP),
            filter(lambda x: is_gpp_file(x), source_files),
        )
    )

    if len(no_licensed_files) > 0:
        for file in no_licensed_files:
            session.warn(f"{file} has no license header")
        session.error()
    else:
        session.log("All files have proper license header")


def run_format_fix(session: nox.Session):
    """Fix files: format and sort import"""
    session.install(f"ruff=={RUFF_VERSION}")
    session.run("python", "-m", "ruff", "format", ".")
    session.run("python", "-m", "ruff", "check", ".", "--select", "I", "--fix")


def run_ruff(session: nox.Session):
    """Assess code quality with ruff"""
    session.install(f"ruff=={RUFF_VERSION}")
    session.run("python", "-m", "ruff", "check", ".")


def run_fawlty_deps(session: nox.Session):
    """Check dependencies with fawlty deps"""
    if not Path("bps").exists():
        session.error("run fawlty_deps session inside package folder: e.g. 'cd bps-common; nox -s fawlty_deps'")

    session.install("FawltyDeps")
    session.run(
        "python",
        "-m",
        "fawltydeps",
        "--code",
        "bps",
        "--detailed",
        "--config-file",
        "../scripts/fawltydeps.toml",
    )


def run_unittest(session: nox.Session):
    """Run package unittest"""

    package_name = Path.cwd().name
    package_module = package_name.replace("-", ".")

    Path("_build").mkdir(exist_ok=True)
    session.install("-e", ".")

    session.install("unittest-xml-reporting", "coverage")
    session.run(
        "python",
        "-m",
        "coverage",
        "run",
        f"--source={package_module}",
        "-m",
        "xmlrunner",
        "--output-file",
        f"_build/unittest-report-{PLATFORM}-py{session.python}.xml",
        "discover",
    )
    session.run("python", "-m", "coverage", "report", "-m")
    session.run(
        "python",
        "-m",
        "coverage",
        "xml",
        "-o",
        f"_build/unittest-coverage-{PLATFORM}-py{session.python}.xml",
    )


def run_build_sdist(session: nox.Session):
    """Build source distribution"""
    if not Path("bps").exists():
        session.error("run build_sdist session inside package folder: e.g. 'cd bps-common; nox -s build_sdist'")

    session.install("build")
    session.run("python", "-m", "build", "--sdist", silent=True)


def run_build_wheel(session: nox.Session):
    """Build wheel distribution"""
    if not Path("bps").exists():
        session.error("run build_wheel session inside package folder: e.g. 'cd bps-common; nox -s build_wheel'")

    session.install("build")
    session.run("python", "-m", "build", "--wheel", silent=True)


def run_check_xsd(session: nox.Session, package: str, file_list: list[str]):
    """Check that XSD inside package are aligned with those main folder"""

    package_root = Path(__file__).resolve().parent.parent

    package_xsd_dir = Path("bps", package, "xsd", "biomass-xsd")
    package_xsd_dir.mkdir(exist_ok=True, parents=True)
    reference_xsd_dir = package_root / "xsd" / "biomass-xsd"

    fail = False
    for xsd in file_list:
        reference_xsd = reference_xsd_dir / xsd
        package_xsd = package_xsd_dir / xsd
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
        package_xsd = package_xsd_dir / xsd

        session.log(f"Copy {reference_xsd} onto {package_xsd}")
        shutil.copyfile(reference_xsd, package_xsd)


# Default sessions
@nox.session()
def format_check(session: nox.Session):
    """Check files are: formatted, with license header and import sorted"""
    run_format_check(session)


@nox.session()
def format_fix(session: nox.Session):
    """Fix files: format and sort import"""
    run_format_fix(session)


@nox.session()
def ruff(session: nox.Session):
    """Assess code quality with ruff"""
    run_ruff(session)


@nox.session()
def fawlty_deps(session: nox.Session):
    """Check dependencies with fawlty deps"""
    run_fawlty_deps(session)


@nox.session()
def build_sdist(session: nox.Session):
    """Build source distribution"""
    run_build_sdist(session)


@nox.session()
def build_wheel(session: nox.Session):
    """Build wheel distribution"""
    run_build_wheel(session)
