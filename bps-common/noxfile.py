# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

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
    noxfile_common.run_unittest(session)
