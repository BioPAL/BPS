# Copyright (C) 2025 ARESYS S.r.l.
# SPDX-FileCopyrightText: 2026 Aresys S.r.l.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""
BPS nox automation file
-----------------------
"""

import sys
from collections.abc import Callable
from pathlib import Path

import nox

# Import xsd generation helper module from scripts folder
sys.path.append("scripts")
import noxfile_common
import xsd_generation_helper

RUFF_VERSION = noxfile_common.RUFF_VERSION


@nox.session
def align_xsd(session: nox.Session):
    """Align xsd in the "subpackages"""
    session.install("nox")
    for package in [
        "bps-transcoder",
        "bps-l1_processor",
        "bps-stack_processor",
        "bps-l2a_processor",
        "bps-l2b_fh_processor",
        "bps-l2b_fd_processor",
        "bps-l2b_agb_processor",
    ]:
        session.chdir(package)
        session.run("python", "-m", "nox", "-s", "align_xsd")
        session.chdir("..")


@nox.session
def generate_xsd_models(session: nox.Session):
    """Generate XSD models with xsdata

    Usage:
        python -m nox -r -s generate_xsd_models

    Before xsd models generation, it is necessary to provide the
    xsd files describing Aresys BrickSAR configurations and interfaces.
    They have to be placed in the xsd/bricksar-xsd directory.
    Use scripts\restrict_xsd_to_bps.py to restrict the xsd files to
    the relevant BPS content.
    """
    session.install("xsdata[cli]", f"ruff=={RUFF_VERSION}")
    xsd_generation_helper.generate_xsd_models(session)


def walk_with_filter(
    path: Path,
    *,
    relevant_file: Callable[[Path], bool] | None = None,
    relevant_dir: Callable[[Path], bool] | None = None,
):
    """Generator that yields all the relevant files inside a given directory"""
    if path.is_file():
        if relevant_file and not relevant_file(path):
            return
        yield path
    else:
        if relevant_dir and not relevant_dir(path):
            return
        for item in path.iterdir():
            yield from walk_with_filter(
                item, relevant_file=relevant_file, relevant_dir=relevant_dir
            )


def get_bps_common_version_no_install() -> str:
    """Get bps-common version by parsing of the init"""

    content = Path("bps-common", "bps", "common", "__init__.py").read_text(
        encoding="utf-8"
    )

    current_version: str | None = None
    for line in content.splitlines():
        result = line.split("__version__ = ")
        if len(result) == 2:
            current_version = result[1].replace('"', "")
            break

    assert current_version
    return current_version


def get_version_in_bps_format(version: str) -> str:
    """Convert a version to the BPS format"""
    major, minor, patch = (int(x) for x in version.split(".")[0:3])
    if minor > 9 or patch > 9:
        raise RuntimeError(f"Minor and patch version cannot exceed '9': {version}")

    minor_bps = minor * 10 + patch
    assert minor_bps < 99
    return f"{major:02d}.{minor_bps:02d}"


@nox.session()
def version_update(session: nox.Session) -> None:
    """Update BPS version"""
    if len(session.posargs) != 1:
        raise RuntimeError(
            "Unexpected number of input arguments, usage: 'nox -s version_update -- 1.0.0'"
        )

    new_version = session.posargs[0]
    current_version = get_bps_common_version_no_install()
    session.log(f"Bumping version from {current_version} to {new_version}")

    def relevant_file(path: Path) -> bool:
        return path.suffix in [".py", ".toml", ".yaml"]

    def relevant_dir(path: Path) -> bool:
        name = path.resolve().name
        return name != "build" and not name.startswith(".")

    updated_files = 0
    for file in walk_with_filter(
        Path.cwd(), relevant_file=relevant_file, relevant_dir=relevant_dir
    ):
        text = file.read_text(encoding="utf-8")
        if current_version in text:
            updated_files += 1
            session.log(f"Updating version in {file}")
            file.write_text(
                text.replace(current_version, new_version), encoding="utf-8"
            )
    session.log(f"{updated_files} files updated")

    new_version_bps = get_version_in_bps_format(new_version)
    current_version_bps = get_version_in_bps_format(current_version)
    if new_version_bps == current_version_bps:
        session.log(f"BPS version {current_version_bps} requires no update")
        return

    session.log(f"Bumping BPS version from {current_version_bps} to {new_version_bps}")
    updated_files = 0
    for file in walk_with_filter(
        Path.cwd(), relevant_file=relevant_file, relevant_dir=relevant_dir
    ):
        text = file.read_text(encoding="utf-8")
        if current_version_bps in text:
            updated_files += 1
            session.log(f"Updating BPS version in {file}")
            file.write_text(
                text.replace(current_version_bps, new_version_bps), encoding="utf-8"
            )
    session.log(f"{updated_files} files updated")

    with session.chdir("bps-task-tables"):
        session.install("nox")
        session.run(
            *f"python -m nox -r -s version_update -- {current_version_bps} {new_version_bps}".split()
        )
