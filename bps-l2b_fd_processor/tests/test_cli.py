import unittest

import bps.l2b_fd_processor
from bps.l2b_fd_processor.cli import run
from click.testing import CliRunner


class CLIRunTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cli_runner = CliRunner()

    def test_run_no_args(self):
        result = self.cli_runner.invoke(run.cli)
        self.assertEqual(result.exit_code, 2)

    def test_run_help(self):
        result = self.cli_runner.invoke(run.cli, ["--help"])
        self.assertEqual(result.exit_code, 0)

    def test_version(self):
        result = self.cli_runner.invoke(run.cli, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(bps.l2b_fd_processor.__version__ in result.output)


if __name__ == "__main__":
    unittest.main()
