"""Automation file"""

import sys

import nox

# Import common utilities from scripts folder
sys.path.append("../scripts")
import noxfile_common


def install_bps_dependencies(session: nox.Session):
    session.conda_install(f"gdal={noxfile_common.GDAL_VERSION}", channel="conda-forge")
    session.install(
        "-e",
        "../bps-common",
        "../bps-transcoder",
    )


@nox.session(python=noxfile_common.UNITTEST_PYVERSIONS, venv_backend=noxfile_common.CONDABUILD_BACKEND)
def unittest(session: nox.Session):
    """Run unittest for current package"""
    install_bps_dependencies(session)
    noxfile_common.run_unittest(session)


XSD_FILES = [
    "bio-aux-pp2_fh.xsd",
    "bio-common-types.xsd",
]


@nox.session
def check_xsd(session: nox.Session):
    """Check that XSD inside package are aligned with those main folder"""

    noxfile_common.run_check_xsd(session, "l2b_fh_processor", XSD_FILES)


@nox.session
def align_xsd(session: nox.Session):
    """Align XSD inside package with those main folder"""

    noxfile_common.run_align_xsd(session, "l2b_fh_processor", XSD_FILES)
