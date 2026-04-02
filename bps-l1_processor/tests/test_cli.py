import unittest

import bps.l1_processor
from bps.l1_processor.cli import run
from click.testing import CliRunner


class CLITestCase(unittest.TestCase):
    """Test command line interface"""

    def setUp(self) -> None:
        self.cli_runner = CliRunner()

    def test_error_no_args(self):
        """Error when no JobOrder is provided"""
        result = self.cli_runner.invoke(run)
        self.assertEqual(result.exit_code, 2)

    def test_display_help(self):
        """Display help"""
        result = self.cli_runner.invoke(run, ["--help"])
        self.assertEqual(result.exit_code, 0)

    def test_display_version(self):
        """Display version"""
        result = self.cli_runner.invoke(run, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(bps.l1_processor.__version__ in result.output)


if __name__ == "__main__":
    unittest.main()
