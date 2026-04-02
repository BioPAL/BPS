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
    session.install(
        "-e",
        "../bps-common",
        "../bps-stack_pre_processor",
        "../bps-stack_coreg_processor",
        "../bps-stack_cal_processor",
        "../bps-transcoder",
    )

    noxfile_common.run_unittest(session)


XSD_FILES = [
    "bio-aux-pps.xsd",
    "bio-common-types.xsd",
]


@nox.session
def check_xsd(session: nox.Session):
    """Check that XSD inside package are aligned with those main folder"""

    noxfile_common.run_check_xsd(session, "stack_processor", XSD_FILES)


@nox.session
def align_xsd(session: nox.Session):
    """Align XSD inside package with those main folder"""

    noxfile_common.run_align_xsd(session, "stack_processor", XSD_FILES)
