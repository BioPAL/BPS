# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS logging
-----------

Examples
--------

initialize the logger, by default log level is set to `INFO` and the node name is set automatically
(see :meth:`init_logger` to change the defaults)

>>> from bps.common import bps_logger
>>> from bps.l1_processor import __version__ as VERSION
>>> bps_logger.init_logger(processor="L1_P", task="L1_P", version=VERSION)

enable console logging

>>> bps_logger.enable_console_logging()

enable file logging

>>> from pathlib import Path
>>> dir = Path("logging_example_dir")
>>> dir.mkdir(exist_ok=True)
>>> bps_logger.enable_file_logging(dir)

change level of logging, or task, or both

>>> import logging
>>> bps_logger.update_logger(loglevel=logging.DEBUG)
>>> bps_logger.update_logger(task="example_task")
>>> bps_logger.update_logger(loglevel=logging.DEBUG, task="example_task")

various logging messages

>>> bps_logger.info("info message")
>>> bps_logger.debug("debug message")
>>> bps_logger.warning("warning message")
>>> bps_logger.critical("critical message")
>>> bps_logger.error("error message")
>>> try:
>>>     assert False
>>> except AssertionError:
>>>     bps_logger.exception("exception message")


Warning
-------

The name of log file depends on the 'processor name' and its 'version'.
Those information can be changed during processing

>>> bps_logger.update_logger(processor="custom_proc_name")

and altering all the subsequent log records, but the log file name
will not change if :meth:`update_logger` was called after :meth:`enable_file_logging`.
"""

import datetime
import logging
import platform
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

_BPS_LOGGER = logging.getLogger("bps")

_LEVEL_MAPPING = {
    "CRITICAL": "E",
    "ERROR": "E",
    "WARNING": "W",
    "INFO": "I",
    "DEBUG": "D",
    "NOTSET": "D",
}

logging.raiseExceptions = False


def _stdout_level(record: logging.LogRecord) -> bool:
    return record.levelno in (logging.DEBUG, logging.INFO, logging.NOTSET)


def _stderr_level(record: logging.LogRecord) -> bool:
    return record.levelno in (logging.CRITICAL, logging.ERROR, logging.WARNING)


def _add_msg_type(record: logging.LogRecord) -> bool:
    """Filter to map logging levels to BPS message types in log record"""
    setattr(record, "msg_type", _LEVEL_MAPPING.get(record.levelname, "E"))
    return True


def _add_musecs(record: logging.LogRecord) -> bool:
    """Filter to add micro seconds to the log record"""
    setattr(record, "musecs", record.msecs * 1e3)
    return True


@dataclass
class _FacilityInfo(logging.Filter):
    """Facility info required to format a log record"""

    init: bool = False
    node: str = ""
    processor: str = ""
    task: str = ""
    version: str = ""

    def __setattr__(self, name, value):
        if isinstance(value, str) and " " in value:
            raise RuntimeError(f"Invalid value for {name}: '{value}' cannot contain spaces")
        self.__dict__[name] = value


@dataclass
class _FacilityInfoFilter(logging.Filter):
    """Filter to add facility info to log record"""

    facility_info: _FacilityInfo

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, "node", self.facility_info.node)
        setattr(record, "processor", self.facility_info.processor)
        setattr(record, "task", self.facility_info.task)
        setattr(record, "version", self.facility_info.version)
        return True


_FACILITY_INFO = _FacilityInfo()

_BPS_FORMATTER = logging.Formatter(
    "%(asctime)s.%(musecs)06d %(node)s %(processor)s %(version)s %(task)s [%(process)010d]: [%(msg_type)s] %(message)s",
    "%Y-%m-%dT%H:%M:%S",
)


def get_version_in_logger_format(version: str) -> str:
    """Convert a version to the format required by the BPS logger.

    Parameters
    ----------
    version : str
        version in standard format (e.g. 2.3.5)

    Returns
    -------
    str
        version in BPS format (e.g. 02.35)
    """
    major, minor, patch = (int(x) for x in version.split(".")[0:3])
    if minor > 9 or patch > 9:
        raise RuntimeError(f"Minor and patch version cannot exceed '9': {version}")

    minor_bps = minor * 10 + patch
    assert minor_bps < 99
    return f"{major:02d}.{minor_bps:02d}"


def get_default_logger_node() -> str:
    """Get a default name of the processing node.

    The default name is name of the current platform node.

    Returns
    -------
    str
        default node name
    """
    return platform.node()


def init_logger(
    processor: str,
    task: str,
    version: str,
    *,
    loglevel: int = logging.INFO,
    node: str | None = None,
):
    """Initialize the BPS logging

    It is necessary to enable console and/or file logging after initialization

    Parameters
    ----------
    processor : str
        processor name
    task : str
        task name
    version : str
        processor version
    loglevel : int, optional
        the log level, any of logging.{`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`}, by default logging.INFO
    node : Optional[str], optional
        platform node name, by default is set to the hostname

    See also
    --------
    enable_console_logging
    enable_file_logging
    update_logger
    """
    if _FACILITY_INFO.init:
        raise RuntimeError("BPS Logger already initialized: call 'update_logger'.")

    loglevel = logging.INFO if loglevel is None else loglevel
    node = get_default_logger_node() if node is None else node

    _BPS_LOGGER.setLevel(loglevel)

    version = get_version_in_logger_format(version)

    _FACILITY_INFO.node = node
    _FACILITY_INFO.processor = processor
    _FACILITY_INFO.task = task
    _FACILITY_INFO.version = version
    _FACILITY_INFO.init = True

    for bps_filter in [_add_msg_type, _add_musecs, _FacilityInfoFilter(_FACILITY_INFO)]:
        _BPS_LOGGER.addFilter(bps_filter)


def _check_logger_init():
    if not _FACILITY_INFO.init:
        raise RuntimeError("BPS Logger not initialized: call 'init_logger' first.")


def enable_console_logging(*, stream_stdout=None, stream_stderr=None):
    """Enable console logging"""
    stream_stdout = stream_stdout if stream_stdout is not None else sys.stdout
    stream_stderr = stream_stderr if stream_stderr is not None else sys.stderr

    _check_logger_init()

    stdout_handler = logging.StreamHandler(stream=stream_stdout)
    stdout_handler.setFormatter(_BPS_FORMATTER)
    stdout_handler.addFilter(_stdout_level)
    _BPS_LOGGER.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(stream=stream_stderr)
    stderr_handler.setFormatter(_BPS_FORMATTER)
    stderr_handler.addFilter(_stderr_level)
    _BPS_LOGGER.addHandler(stderr_handler)


def add_file_handler(log_file: Path):
    """Add file handler

    Parameters
    ----------
    log_file : Path
        log file
    """
    _check_logger_init()

    handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8", delay=False, errors=None)
    handler.setFormatter(_BPS_FORMATTER)
    _BPS_LOGGER.addHandler(handler)


def enable_file_logging(working_dir: Path):
    """Enable file logging

    Parameters
    ----------
    working_dir : Path
        logging directory
    """

    _check_logger_init()

    creation_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_file = working_dir.joinpath(f"BIO_{_FACILITY_INFO.processor}_{_FACILITY_INFO.version}_{creation_time}.log")
    add_file_handler(log_file)


def get_log_file() -> Path | None:
    """Get bps logger file, if set.

    Returns
    -------
    Optional[Path]
        Path to the log file if set, otherwise None
    """
    for handler in _BPS_LOGGER.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def update_logger(
    *,
    loglevel: int | None = None,
    node: str | None = None,
    processor: str | None = None,
    task: str | None = None,
    version: str | None = None,
):
    """Update BPS logging

    It allows to change one or more property of the bps logger.
    If the property is not specified, it is not modified.

    Warning
    -------
    The logger keeps logging to the same file, regardless of the update

    Parameters
    ----------
    loglevel : Optional[int], optional
        log level, by default None
    node : Optional[str], optional
        node name, by default None
    processor : Optional[str], optional
        processor name, by default None
    task : Optional[str], optional
        task name, by default None
    version : Optional[str], optional
        processor version, by default None
    """
    _check_logger_init()

    if loglevel is not None:
        _BPS_LOGGER.setLevel(loglevel)

    if node is not None:
        _FACILITY_INFO.node = node
    if processor is not None:
        _FACILITY_INFO.processor = processor
    if task is not None:
        _FACILITY_INFO.task = task
    if version is not None:
        _FACILITY_INFO.version = get_version_in_logger_format(version)


def stack_trace(
    *,
    loglevel: int = logging.DEBUG,
    limit: int | None = None,
):
    """
    Print the stack trace as log records.

    Parameters
    ----------
    loglevel : int = logging.DEBUG
        The logging severity level, any of
        logging.{`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`}.
        Defaulted to logging.DEBUG.
    limit : Optional[int] = None
        If provided, only this many frames are printed. Otherwise,
        all frames are printed.

    Raises
    ------
    ValueError : If provided loglevel is invalid.

    """
    log_fn = None
    if loglevel is logging.DEBUG:
        log_fn = _BPS_LOGGER.debug
    elif loglevel is logging.INFO:
        log_fn = _BPS_LOGGER.info
    elif loglevel is logging.WARNING:
        log_fn = _BPS_LOGGER.warning
    elif loglevel is logging.ERROR:
        log_fn = _BPS_LOGGER.error
    elif loglevel is logging.CRITICAL:
        log_fn = _BPS_LOGGER.critical
    else:
        raise ValueError(f"Unsupported logger severity {loglevel}")

    for line in traceback.format_exc(limit).splitlines():
        log_fn(line)


info = _BPS_LOGGER.info
"""See :meth:`logging.Logger.info`"""
debug = _BPS_LOGGER.debug
"""See :meth:`logging.Logger.debug`"""
warning = _BPS_LOGGER.warning
"""See :meth:`logging.Logger.warning`"""
error = _BPS_LOGGER.error
"""See :meth:`logging.Logger.error`"""
critical = _BPS_LOGGER.critical
"""See :meth:`logging.Logger.critical`"""
exception = _BPS_LOGGER.exception
"""See :meth:`logging.Logger.exception`"""
log = _BPS_LOGGER.log
"""See :meth:`logging.Logger.log`"""
