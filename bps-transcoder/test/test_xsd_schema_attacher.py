# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bps.transcoder.utils.xsd_schema_attacher import copy_biomass_xsd_files


class XsdCopyTest(unittest.TestCase):
    def test_copy_biomass_xsd_files(self):
        files = ["bio-common-types.xsd", "bio-l2a-fd-main-annotation.xsd"]
        with TemporaryDirectory() as output_dir:
            copy_biomass_xsd_files(Path(output_dir), files)
            self.assertTrue(Path(output_dir).joinpath("bio-common-types.xsd").exists())
            self.assertTrue(Path(output_dir).joinpath("bio-l2a-fd-main-annotation.xsd").exists())


if __name__ == "__main__":
    unittest.main()
