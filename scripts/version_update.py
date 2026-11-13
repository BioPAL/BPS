# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Update BPS package versions and task-table versions.

Run with:
    python scripts/version_update.py 1.2.3
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def walk_with_filter(
    path: Path,
    *,
    relevant_file: Callable[[Path], bool] | None = None,
    relevant_dir: Callable[[Path], bool] | None = None,
) -> Iterator[Path]:
    """Yield the relevant files below a directory."""
    if path.is_file():
        if relevant_file is None or relevant_file(path):
            yield path
        return

    if relevant_dir is not None and not relevant_dir(path):
        return
    for item in path.iterdir():
        yield from walk_with_filter(item, relevant_file=relevant_file, relevant_dir=relevant_dir)


def get_bps_common_version() -> str:
    """Read the current version from bps-common."""
    init_file = ROOT / "bps-common" / "bps" / "common" / "__init__.py"
    content = init_file.read_text(encoding="utf-8")
    match = re.search(r"^\s*__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not find __version__ in {init_file}")
    return match.group(1)


def get_version_in_bps_format(version: str) -> str:
    """Convert a version's numeric components to the BPS two-component format."""
    components = version.split(".")[:3]
    if len(components) != 3:
        raise ValueError(f"Version must start with major.minor.patch: {version}")
    try:
        major, minor, patch = (int(component) for component in components)
    except ValueError as error:
        raise ValueError(f"Version must start with major.minor.patch: {version}") from error

    if minor > 9 or patch > 9:
        raise ValueError(f"Minor and patch version cannot exceed '9': {version}")

    minor_bps = minor * 10 + patch
    if minor_bps >= 99:
        raise ValueError(f"BPS minor version must be less than '99': {version}")
    return f"{major:02d}.{minor_bps:02d}"


def relevant_file(path: Path) -> bool:
    """Select package version metadata, excluding transcoder I/O modules."""
    if any(path.parts[index : index + 2] == ("transcoder", "io") for index in range(len(path.parts) - 1)):
        return False
    return path.name in {"__init__.py", "pyproject.toml", "meta.yaml"}


def relevant_dir(path: Path) -> bool:
    """Skip generated build trees and hidden directories."""
    name = path.name
    return name != "build" and not name.startswith(".")


def replace_version(old_version: str, new_version: str, *, log_prefix: str = "") -> int:
    """Replace a version in relevant files and return the number changed."""
    updated_files = 0
    for file in walk_with_filter(ROOT, relevant_file=relevant_file, relevant_dir=relevant_dir):
        text = file.read_text(encoding="utf-8")
        new_text = text.replace(old_version, new_version)
        if new_text == text:
            continue
        print(f"Updating {log_prefix}version in {file.relative_to(ROOT)}")
        file.write_text(new_text, encoding="utf-8")
        updated_files += 1
    return updated_files


def update_task_table_files(task_tables_dir: Path, current_version: str, new_version: str) -> None:
    """Update and rename task tables and product metadata mapping files."""

    def run_git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=task_tables_dir, check=True)

    print(f"Bumping task-table version from {current_version} to {new_version}")
    proc_tag_old = f"<Processor_Version>{current_version}</Processor_Version>"
    proc_tag_new = f"<Processor_Version>{new_version}</Processor_Version>"
    sw_tag_old = f"<SW_Version_Tag>{current_version}</SW_Version_Tag>"
    sw_tag_new = f"<SW_Version_Tag>{new_version}</SW_Version_Tag>"
    task_tag_old = f"<Task_Version>{current_version}</Task_Version>"
    task_tag_new = f"<Task_Version>{new_version}</Task_Version>"

    task_tables = list(task_tables_dir.rglob("*TaskTable*.xml"))
    task_tables_renamed = 0
    task_tables_updated = 0
    for file in task_tables:
        text = file.read_text(encoding="utf-8")
        updated_text = (
            text.replace(proc_tag_old, proc_tag_new).replace(sw_tag_old, sw_tag_new).replace(task_tag_old, task_tag_new)
        )
        relative_file = file.relative_to(task_tables_dir)

        if updated_text != text:
            file.write_text(updated_text, encoding="utf-8")
            run_git("add", "--", str(relative_file))
            task_tables_updated += 1

        new_file = file.with_name(file.name.replace(current_version, new_version))
        if new_file != file:
            run_git("mv", "-f", "--", str(relative_file), str(new_file.relative_to(task_tables_dir)))
            task_tables_renamed += 1

    print(f"{task_tables_updated} task tables updated")
    print(f"{task_tables_renamed} task tables renamed")

    proc_tag_old = f"<ProcessorVersion>{current_version}</ProcessorVersion>"
    proc_tag_new = f"<ProcessorVersion>{new_version}</ProcessorVersion>"
    pmm_files = list(task_tables_dir.rglob("*ProductMetadataMapping*.xml"))
    pmm_renamed = 0
    pmm_updated = 0
    for file in pmm_files:
        text = file.read_text(encoding="utf-8")
        updated_text = text.replace(proc_tag_old, proc_tag_new)
        relative_file = file.relative_to(task_tables_dir)

        if updated_text != text:
            file.write_text(updated_text, encoding="utf-8")
            run_git("add", "--", str(relative_file))
            pmm_updated += 1

        new_file = file.with_name(file.name.replace(current_version, new_version))
        if new_file != file:
            run_git("mv", "-f", "--", str(relative_file), str(new_file.relative_to(task_tables_dir)))
            pmm_renamed += 1

    print(f"{pmm_updated} product metadata mappings updated")
    print(f"{pmm_renamed} product metadata mappings renamed")


def update(new_version: str) -> None:
    """Update package metadata, task tables, and product metadata mappings."""
    current_version = get_bps_common_version()
    current_version_bps = get_version_in_bps_format(current_version)
    new_version_bps = get_version_in_bps_format(new_version)

    print(f"Bumping version from {current_version} to {new_version}")
    updated = replace_version(current_version, new_version)
    print(f"{updated} files updated")

    if new_version_bps == current_version_bps:
        print(f"BPS version {current_version_bps} requires no update")
        return

    print(f"Bumping BPS version from {current_version_bps} to {new_version_bps}")
    updated = replace_version(current_version_bps, new_version_bps, log_prefix="BPS ")
    print(f"{updated} files updated")

    task_tables_dir = ROOT / "bps-task-tables"
    update_task_table_files(task_tables_dir, current_version_bps, new_version_bps)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/version_update.py major.minor.patch")
    update(sys.argv[1])


if __name__ == "__main__":
    main()
