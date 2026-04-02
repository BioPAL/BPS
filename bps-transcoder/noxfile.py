"""Automation file"""

import sys

import nox

# Import common utilities from scripts folder
sys.path.append("../scripts")
import noxfile_common


@nox.session(python=noxfile_common.UNITTEST_PYVERSIONS, venv_backend=noxfile_common.CONDABUILD_BACKEND)
def unittest(session: nox.Session):
    """Run unittest for current package"""
    session.conda_install(f"gdal={noxfile_common.GDAL_VERSION}", channel="conda-forge")
    session.install("-e", "../bps-common")

    noxfile_common.run_unittest(session)


XSD_FILES = [
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
]


@nox.session
def check_xsd(session: nox.Session):
    """Check that XSD inside package are aligned with those main folder"""

    noxfile_common.run_check_xsd(session, "transcoder", XSD_FILES)


@nox.session
def align_xsd(session: nox.Session):
    """Align XSD inside package with those main folder"""

    noxfile_common.run_align_xsd(session, "transcoder", XSD_FILES)
