# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
BPS utils
----------
"""

import logging
from enum import Enum

from bps.common import bps_logger


class EarthModel(Enum):
    """Earth models"""

    WGS84 = "WGS84"
    GETASSE = "GETASSE"
    SRTM = "SRTM"
    COPERNICUS = "COPERNICUS"


class ProductFormat(Enum):
    """Aresys products format"""

    BIN = "BIN+XML"
    TIFF = "TIFF+XML"


class LogLevel(Enum):
    """Logging levels"""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    DEBUG = "DEBUG"


def floating_point_error_handler(
    loglevel: int,
    fn_name: str,
    error_type: str,
    *args,
):
    """
    Floating point warning callback function. This is used by
    numpy to report floating point errors such as floating point
    under- or overflows.

    Example
    -------
    Possible usage within the stack (see for instance
    https://numpy.org/devdocs/reference/generated/numpy.seterr.html)

        np.seterr(all="call")
        np.seterrcall(
            functools.partial(floating_point_error_handler, loglevel, fn_name)
        )

    or

        np.seterr(all="call")
        np.seterrcall(
            lambda t, _: floating_point_error_handler(loglevel, fn_name, t, _)
        )

    Parameters
    ----------
    loglevel: int
        The logger level (i.e. DEBUG, INFO, etc.)

    fn_name: str
        The function throwing the warning.

    error_type: str
        The error type (e.g. divide by zero).

    Raises
    ------
    ValueError

    """
    if loglevel == logging.DEBUG:
        logger_fn = bps_logger.debug
    elif loglevel == logging.INFO:
        logger_fn = bps_logger.info
    elif loglevel == logging.WARNING:
        logger_fn = bps_logger.warning
    elif loglevel == logging.ERROR:
        logger_fn = bps_logger.error
    elif loglevel == logging.CRITICAL:
        logger_fn = bps_logger.critical
    else:
        raise ValueError(f"Unsupported logging level '{loglevel}'")

    logger_fn(
        "Floating point error %s ecountered in %s",
        error_type,
        fn_name,
    )
