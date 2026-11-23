# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Check or align package XSD files with the repository's reference XSD files."""

import argparse
import shutil
import sys
from pathlib import Path

PACKAGE_XSD_FILES = {
    "l1_processor": [
        "bio-aux-pp1.xsd",
        "bio-common-types.xsd",
    ],
    "l2a_processor": [
        "bio-aux-pp2_2a.xsd",
        "bio-common-types.xsd",
    ],
    "l2b_agb_processor": [
        "bio-aux-pp2_ab.xsd",
        "bio-common-types.xsd",
    ],
    "l2b_fd_processor": [
        "bio-aux-pp2_fd.xsd",
        "bio-common-types.xsd",
    ],
    "l2b_fh_processor": [
        "bio-aux-pp2_fh.xsd",
        "bio-common-types.xsd",
    ],
    "stack_processor": [
        "bio-aux-pps.xsd",
        "bio-common-types.xsd",
    ],
    "transcoder": [
        "annotation/v4.5.0/bio-l1ab-main-annotation.xsd",
        "annotation/v4.5.0/bio-l1c-main-annotation.xsd",
        "annotation/v4.5.0/bio-l1-annotations.xsd",
        "annotation/v4.5.0/bio-l2a-fd-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2a-fh-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2a-tfh-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2a-gn-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2b-agb-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2b-fd-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2b-fh-main-annotation.xsd",
        "annotation/v4.5.0/bio-l2l3-fd-proc-annotations.xsd",
        "annotation/v4.5.0/bio-l2l3-fh-proc-annotations.xsd",
        "annotation/v4.5.0/bio-l2l3-tfh-proc-annotations.xsd",
        "annotation/v4.5.0/bio-l2l3-agb-proc-annotations.xsd",
        "annotation/v4.5.0/bio-l2l3-common-annotations.xsd",
        "annotation/v4.5.0/bio-common-types.xsd",
        "bio-l1-overlay-support.xsd",
        "bio-l1-overlay.xsd",
        "bio-l1-vrt.xsd",
        "bio-aux-orbit.xsd",
        "bio-aux-attitude.xsd",
    ],
}
SUBPROJECT_TO_PACKAGE = {f"bps-{package}": package for package in PACKAGE_XSD_FILES}


def _get_xsd_directories(package: str) -> tuple[Path, Path]:
    repository_root = Path(__file__).resolve().parent.parent
    package_xsd_dir = repository_root / f"bps-{package}" / "bps" / package / "xsd" / "biomass-xsd"
    reference_xsd_dir = repository_root / "xsd" / "biomass-xsd"
    return package_xsd_dir, reference_xsd_dir


def check_xsd_alignment(package: str) -> bool:
    """Return whether the selected package XSD files match their references."""

    package_xsd_dir, reference_xsd_dir = _get_xsd_directories(package)
    aligned = True

    for xsd in PACKAGE_XSD_FILES[package]:
        reference_xsd = reference_xsd_dir / xsd
        package_xsd = package_xsd_dir / reference_xsd.name

        if not reference_xsd.is_file():
            print(f"Reference XSD does not exist: {reference_xsd}", file=sys.stderr)
            aligned = False
        elif not package_xsd.is_file():
            print(f"XSD is not in the package folder: {package_xsd}", file=sys.stderr)
            aligned = False
        elif reference_xsd.read_bytes() != package_xsd.read_bytes():
            print(f"XSD is not aligned: {xsd}", file=sys.stderr)
            aligned = False

    return aligned


def align_xsd(package: str) -> bool:
    """Copy the selected package's reference XSD files into the package."""

    package_xsd_dir, reference_xsd_dir = _get_xsd_directories(package)
    package_xsd_dir.mkdir(exist_ok=True, parents=True)
    aligned = True

    for xsd in PACKAGE_XSD_FILES[package]:
        reference_xsd = reference_xsd_dir / xsd
        package_xsd = package_xsd_dir / reference_xsd.name

        if not reference_xsd.is_file():
            print(f"Reference XSD does not exist: {reference_xsd}", file=sys.stderr)
            aligned = False
            continue

        print(f"Copy {reference_xsd} onto {package_xsd}")
        shutil.copyfile(reference_xsd, package_xsd)

    return aligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("check", "align"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("subproject", choices=SUBPROJECT_TO_PACKAGE, help="subproject to process")

    args = parser.parse_args()
    package = SUBPROJECT_TO_PACKAGE[args.subproject]

    if args.command == "align":
        if not align_xsd(package):
            return 1
        print(f"XSD files inside subproject {args.subproject!r} aligned successfully.")
        return 0

    if not check_xsd_alignment(package):
        print(f"XSD files inside subproject {args.subproject!r} are not aligned.", file=sys.stderr)
        return 1

    print(f"XSD files inside subproject {args.subproject!r} are aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
