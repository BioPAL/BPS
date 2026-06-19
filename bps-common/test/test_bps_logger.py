# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import io
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bps.common import bps_logger


class BPSLoggerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(bps_logger)
        importlib.reload(logging)

        bps_logger.init_logger("PROC", "TASK", "3.9.5.dev0", loglevel=logging.DEBUG, node="NODENAME")

    def test_info_log(self):
        stdout_captured = io.StringIO()
        stderr_captured = io.StringIO()
        bps_logger.enable_console_logging(stream_stdout=stdout_captured, stream_stderr=stderr_captured)

        bps_logger.info("INFO_MESSAGE")

        self.assertEqual(stderr_captured.getvalue(), "")
        (
            date,
            node,
            proc,
            ver,
            task,
            pid,
            msg_type,
            message,
        ) = stdout_captured.getvalue().split()

        self.assertEqual(len(date), 26)
        self.assertEqual(node, "NODENAME")
        self.assertEqual(proc, "PROC")
        self.assertEqual(task, "TASK")
        self.assertEqual(ver, "03.95")
        self.assertEqual(pid[0], "[")
        self.assertEqual(pid[11:13], "]:")
        self.assertEqual(msg_type, "[I]")
        self.assertEqual(message, "INFO_MESSAGE")

    def test_debug_log(self):
        stdout_captured = io.StringIO()
        stderr_captured = io.StringIO()
        bps_logger.enable_console_logging(stream_stdout=stdout_captured, stream_stderr=stderr_captured)

        bps_logger.debug("DEBUG_MESSAGE")

        self.assertEqual(stderr_captured.getvalue(), "")
        (
            date,
            node,
            proc,
            ver,
            task,
            pid,
            msg_type,
            message,
        ) = stdout_captured.getvalue().split()

        self.assertEqual(len(date), 26)
        self.assertEqual(node, "NODENAME")
        self.assertEqual(proc, "PROC")
        self.assertEqual(task, "TASK")
        self.assertEqual(ver, "03.95")
        self.assertEqual(pid[0], "[")
        self.assertEqual(pid[11:13], "]:")
        self.assertEqual(msg_type, "[D]")
        self.assertEqual(message, "DEBUG_MESSAGE")

    def test_error_log(self):
        stdout_captured = io.StringIO()
        stderr_captured = io.StringIO()
        bps_logger.enable_console_logging(stream_stdout=stdout_captured, stream_stderr=stderr_captured)

        bps_logger.error("ERROR_MESSAGE")

        self.assertEqual(stdout_captured.getvalue(), "")
        (
            date,
            node,
            proc,
            ver,
            task,
            pid,
            msg_type,
            message,
        ) = stderr_captured.getvalue().split()

        self.assertEqual(len(date), 26)
        self.assertEqual(node, "NODENAME")
        self.assertEqual(proc, "PROC")
        self.assertEqual(task, "TASK")
        self.assertEqual(ver, "03.95")
        self.assertEqual(pid[0], "[")
        self.assertEqual(pid[11:13], "]:")
        self.assertEqual(msg_type, "[E]")
        self.assertEqual(message, "ERROR_MESSAGE")

    def test_update_logger(self):
        bps_logger.update_logger(
            loglevel=logging.INFO,
            task="UPDATED_TASK",
            processor="UPDATED_PROCESSOR",
            node="UPDATED_NODE",
            version="1.0.0",
        )
        stdout_captured = io.StringIO()
        stderr_captured = io.StringIO()
        bps_logger.enable_console_logging(stream_stdout=stdout_captured, stream_stderr=stderr_captured)

        bps_logger.debug("DEBUG_MESSAGE_IGNORED")

        self.assertEqual(stdout_captured.getvalue(), "")
        bps_logger.info("INFO_MESSAGE")
        (
            date,
            node,
            proc,
            ver,
            task,
            pid,
            msg_type,
            message,
        ) = stdout_captured.getvalue().split()

        self.assertEqual(len(date), 26)
        self.assertEqual(node, "UPDATED_NODE")
        self.assertEqual(proc, "UPDATED_PROCESSOR")
        self.assertEqual(task, "UPDATED_TASK")
        self.assertEqual(ver, "01.00")
        self.assertEqual(pid[0], "[")
        self.assertEqual(pid[11:13], "]:")
        self.assertEqual(msg_type, "[I]")
        self.assertEqual(message, "INFO_MESSAGE")

    def test_file_logging(self):
        with TemporaryDirectory() as tmpdir:
            try:
                bps_logger.enable_file_logging(Path(tmpdir))

                file_list = list(Path(tmpdir).iterdir())
                self.assertEqual(len(file_list), 1)
                logfile = file_list[0]

                assert logfile.name.startswith("BIO_PROC_03.95_")
                assert logfile.name.endswith(".log")

                bps_logger.info("INFO_MESSAGE")
                bps_logger.error("ERROR_MESSAGE")
                info_line, error_line = logfile.read_text().splitlines()
                (
                    date,
                    node,
                    proc,
                    ver,
                    task,
                    pid,
                    msg_type,
                    message,
                ) = info_line.split()

                self.assertEqual(len(date), 26)
                self.assertEqual(node, "NODENAME")
                self.assertEqual(proc, "PROC")
                self.assertEqual(task, "TASK")
                self.assertEqual(ver, "03.95")
                self.assertEqual(pid[0], "[")
                self.assertEqual(pid[11:13], "]:")
                self.assertEqual(msg_type, "[I]")
                self.assertEqual(message, "INFO_MESSAGE")

                (
                    date,
                    node,
                    proc,
                    ver,
                    task,
                    pid,
                    msg_type,
                    message,
                ) = error_line.split()

                self.assertEqual(len(date), 26)
                self.assertEqual(node, "NODENAME")
                self.assertEqual(proc, "PROC")
                self.assertEqual(task, "TASK")
                self.assertEqual(ver, "03.95")
                self.assertEqual(pid[0], "[")
                self.assertEqual(pid[11:13], "]:")
                self.assertEqual(msg_type, "[E]")
                self.assertEqual(message, "ERROR_MESSAGE")
            finally:
                logging.shutdown()


class Invalid(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(bps_logger)
        importlib.reload(logging)

    def test_double_initialization(self):
        bps_logger.init_logger("PROC", "TASK", "3.9.5.dev0", loglevel=logging.DEBUG, node="NODENAME")

        with self.assertRaises(RuntimeError):
            bps_logger.init_logger("PROC", "TASK", "3.9.5.dev0", loglevel=logging.DEBUG, node="NODENAME")

    def test_not_initialized(self):
        with self.assertRaises(RuntimeError):
            bps_logger.enable_console_logging()

        with self.assertRaises(RuntimeError):
            bps_logger.enable_file_logging(Path("non_existing_dir"))

        with self.assertRaises(RuntimeError):
            bps_logger.update_logger(loglevel=logging.INFO)

    def test_invalid_facility_info_fields_on_init(self):
        with self.assertRaises(RuntimeError):
            bps_logger.init_logger(processor="proc", task="NEW TASK", version="1.0.0", node="node")

        with self.assertRaises(RuntimeError):
            bps_logger.init_logger(processor="proc", task="task", version="1.0.0", node="NEW NODE")

        with self.assertRaises(RuntimeError):
            bps_logger.init_logger(processor="NEW PROCESSOR", task="task", version="1.0.0", node="node")

        with self.assertRaises(ValueError):
            bps_logger.init_logger(processor="proc", task="task", version="v1.0.0", node="node")

        with self.assertRaises(ValueError):
            bps_logger.init_logger(processor="proc", task="task", version="100", node="node")

    def test_invalid_facility_info_fields_on_update(self):
        bps_logger.init_logger("PROC", "TASK", "3.9.5.dev0", loglevel=logging.DEBUG, node="NODENAME")

        with self.assertRaises(RuntimeError):
            bps_logger.update_logger(task="UPDATED TASK")

        with self.assertRaises(RuntimeError):
            bps_logger.update_logger(node="UPDATED NODE")

        with self.assertRaises(RuntimeError):
            bps_logger.update_logger(processor="UPDATED PROCESSOR")

        with self.assertRaises(ValueError):
            bps_logger.update_logger(version="v1.0.0")

        with self.assertRaises(ValueError):
            bps_logger.update_logger(version="100")


if __name__ == "__main__":
    unittest.main()
