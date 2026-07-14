# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

from bps.l1_geocoding_tool import __version__ as VERSION
from bps.l1_geocoding_tool.cli import run_cli
from click.testing import CliRunner


class TestCLI(unittest.TestCase):
    """Test CLI"""

    def test_help(self):
        """Test help"""

        runner = CliRunner()
        result = runner.invoke(run_cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("BIOMASS L1 Geocoding Tool", result.output)

    def test_version(self):
        """Test version"""

        runner = CliRunner()
        result = runner.invoke(run_cli, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(f"bps_l1_geocoding_tool, version {VERSION}", result.output)


if __name__ == "__main__":
    unittest.main()
