import shutil
import warnings
from pathlib import Path

from setuptools import setup


def xsd_files_copy():
    required_biomass_xsd = {
        "bio-l1ab-main-annotation.xsd",
        "bio-l1c-main-annotation.xsd",
        "bio-l1-annotations.xsd",
        "bio-l1-overlay.xsd",
        "bio-l1-overlay-support.xsd",
        "bio-l1-vrt.xsd",
        "bio-l2a-fd-main-annotation.xsd",
        "bio-l2a-fh-main-annotation.xsd",
        "bio-l2a-tfh-main-annotation.xsd",
        "bio-l2a-gn-main-annotation.xsd",
        "bio-l2b-agb-main-annotation.xsd",
        "bio-l2b-fd-main-annotation.xsd",
        "bio-l2b-fh-main-annotation.xsd",
        "bio-l2l3-fd-proc-annotations.xsd",
        "bio-l2l3-fh-proc-annotations.xsd",
        "bio-l2l3-tfh-proc-annotations.xsd",
        "bio-l2l3-agb-proc-annotations.xsd",
        "bio-l2l3-common-annotations.xsd",
        "bio-common-types.xsd",
        "bio-aux-orbit.xsd",
        "bio-aux-attitude.xsd",
    }
    package_root = Path(__file__).resolve().parent.parent

    internal_biomass_xsd_dir = Path("bps", "transcoder", "xsd", "biomass-xsd")
    internal_biomass_xsd_dir.mkdir(exist_ok=True, parents=True)

    for xsd in required_biomass_xsd:
        source = package_root.joinpath("xsd", "biomass-xsd", xsd)
        destination = internal_biomass_xsd_dir.joinpath(xsd)

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
