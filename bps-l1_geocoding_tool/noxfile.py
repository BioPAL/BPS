"""Automation file"""

import sys

import nox

# Import common utilities from scripts folder
sys.path.append("../scripts")
import noxfile_common  # pylint: disable=wrong-import-position


@nox.session(python=noxfile_common.UNITTEST_PYVERSIONS, venv_backend="mamba")
def unittest(session: nox.Session):
    """Run unittest for current package"""
    session.conda_install(f"gdal={noxfile_common.GDAL_VERSION}", channel="conda-forge")
    session.install("-e", "../bps-common")
    session.install("-e", "../bps-transcoder")

    noxfile_common.run_unittest(session)
