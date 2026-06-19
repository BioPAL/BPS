# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
EO CFI loading module module
"""

import ctypes as ct

eolib = {
    "libproj": "libproj.so.14",
    "libgeotiff": "libgeotiff.so.2",
    "libcommon": "libCfiEECommon.so",
    "libfile": "libCfiFileHandling.so",
    "libdh": "libCfiDataHandling.so",
    "libcfi": "libCfiLib.so",
    "liborb": "libCfiOrbit.so",
    "libpointing": "libCfiPointing.so",
}

EOCFI_LIBS = {id: ct.CDLL(name) for id, name in eolib.items()}
