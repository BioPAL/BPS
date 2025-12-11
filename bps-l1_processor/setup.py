import shutil
import warnings
from pathlib import Path

from setuptools import setup


def xsd_files_copy():
    required_biomass_xsd = {
        "bio-aux-pp1.xsd",
        "bio-common-types.xsd",
    }
    package_root = Path(__file__).resolve().parent.parent

    internal_biomass_xsd_dir = Path("bps", "l1_processor", "xsd", "biomass-xsd")
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
