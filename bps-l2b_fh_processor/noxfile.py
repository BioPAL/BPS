"""Automation file"""

import sys

import nox

# Import common utilities from scripts folder
sys.path.append("../scripts")
import noxfile_common

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
