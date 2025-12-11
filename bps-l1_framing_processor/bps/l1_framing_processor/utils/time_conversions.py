# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Time conversions module
"""

from datetime import datetime

import numpy as np
from bps.l1_framing_processor.utils.constants import REFERENCE_EPOCH


def to_mjd2000(timestamp, ref=np.datetime64("2000-01-01T00:00:00")):
    """Time datetime64-to-MJD2000 conversion

    Parameters
    ----------
    timestamp : np.datetime64
        Time (datetime64)
    ref : np.datetime64, optional
        Reference time, by default np.datetime64("2000-01-01T00:00:00Z")

    Returns
    -------
    double
        Time (MJD2000)
    """
    if hasattr(timestamp, "__iter__"):
        timestamp = np.asarray(timestamp)
        if isinstance(timestamp[0], datetime):
            ref = datetime(2000, 1, 1)
            time = [(t_ - ref).total_seconds() for t_ in timestamp]
            time = np.asarray(time)
        elif isinstance(timestamp[0], np.datetime64):
            time = (timestamp - ref).astype("timedelta64[ns]")
            time = (np.double(time) * 1e-9).reshape(timestamp.shape)

    else:
        if isinstance(timestamp, datetime):
            ref = datetime(2000, 1, 1)
            time = (timestamp - ref).total_seconds()
        elif isinstance(timestamp, np.datetime64):
            time = (timestamp - ref).astype("timedelta64[ns]")
            time = np.double(time) * 1e-9

    return time


def to_datetime64(mjd2000, ref=np.datetime64("2000-01-01T00:00:00")):
    """Time MJD2000-to-datetime64 conversion

    Parameters
    ----------
    mjd2000 : double
        Time (MJD2000)
    ref : np.datetime64, optional
        Reference time, by default np.datetime64("2000-01-01T00:00:00Z")

    Returns
    -------
    np.datetime64
        Time (datetime64)
    """
    mjd2000 = np.atleast_1d(mjd2000)
    date = (mjd2000 * 1e9).astype(int)
    date = date.astype("timedelta64[ns]")
    date = date.squeeze()

    return ref + date


def datetime64_to_string(date, unit="s"):
    """Time datetime64-to-string conversion

    Parameters
    ----------
    date : np.datetime64
        Time (datetime64)
    unit : str, optional
        Output string precision, by default "s"

    Returns
    -------
    str
        Time (str)
    """
    return np.datetime_as_string(date, unit=unit).item()


def datetime64_to_compact_string(date):
    """Time datetime64-to-string (compact) conversion

    Parameters
    ----------
    date : np.datetime64
        Time (datetime64)

    Returns
    -------
    str
        Time (str, compact)
    """
    return np.datetime_as_string(date, unit="s").item().replace("-", "").replace(":", "")


def base36encode(number, alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Base-36 encoding

    Parameters
    ----------
    number : int
        Number
    alphabet : str, optional
        Alphabet, by default "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    Returns
    -------
    str
        Number base-36 encoded
    """
    base36 = ""
    sign = ""

    if number < 0:
        sign = "-"
        number = -number

    if 0 <= number < len(alphabet):
        return sign + alphabet[number]

    while number != 0:
        number, i = divmod(number, len(alphabet))
        base36 = alphabet[i] + base36

    return sign + base36


def base36decode(number):
    """Base-36 decoding

    Parameters
    ----------
    number : str
        Number base-36 encoded

    Returns
    -------
    int
        Number
    """
    return int(number, 36)


def datetime64_to_compact_date(date, ref=REFERENCE_EPOCH):
    """Time datetime64-to-compact date conversion

    Parameters
    ----------
    date : np.datetime64
        Time (datetime64)
    ref : np.datetime64, optional
        Reference epoch, by default REFERENCE_EPOCH

    Returns
    -------
    str
        Relative time base-36 encoded
    """
    rel_date = int((date - ref) / np.timedelta64(1, "s"))
    compact_date = base36encode(rel_date)
    return compact_date
