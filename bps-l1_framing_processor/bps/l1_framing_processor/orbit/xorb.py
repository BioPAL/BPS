# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import ctypes as ct

from bps.l1_framing_processor.orbit.loader import EOCFI_LIBS
from bps.l1_framing_processor.orbit.xl import XL_SAT_BIOMASS, XL_TIME_UTC, xl_get_msg, xl_model_id, xl_time_id

XO_MAX_STR = 512
XO_MAX_COD = 256
XO_NUM_ERR_ORBIT_INIT_FILE = 2
XO_NUM_ERR_OSV_COMPUTE = 1
XO_NUM_ERR_ORBIT_CLOSE = 1
XO_ERR = -1
XO_SAT_JASON_CSA = 132
XO_ORBIT_INIT_AUTO = 0
XO_SEL_TIME = 1
XO_ORBIT_INIT_FILE_ID = 4
XO_OSV_COMPUTE_ID = 8
XO_SEL_FILE = 0  #

XYZ_ARRAY_SIZE = 3


class xo_orbit_id(ct.Structure):
    _fields_ = [("ee_id", ct.c_void_p)]


class xo_anx_extra_info(ct.Structure):
    _fields_ = [("abs_orbit", ct.c_int), ("tanx", ct.c_double), ("tnod", ct.c_double)]


def xo_print_msg(n, msg):
    print(ct.string_at(msg))


def xo_orbit_init_file(
    sat_id=XL_SAT_BIOMASS,
    model_id=None,
    time_id=None,
    orbit_file_mode=XO_ORBIT_INIT_AUTO,
    orbit_file=None,
    time_mode=XO_SEL_FILE,
    time_ref=XL_TIME_UTC,
    time0=0.0,
    time1=0.0,
    orbit0=0,
    orbit1=0,
):
    """Orbit initialisation from a file"""

    # preparing inputs
    sat_id = ct.c_long(sat_id)
    orbit_file_mode = ct.c_long(orbit_file_mode)
    orbit_file = ct.c_char_p(orbit_file.encode("utf-8"))
    time_mode = ct.c_long(time_mode)
    time_ref = ct.c_long(time_ref)

    time0 = ct.c_double(time0)
    time1 = ct.c_double(time1)
    orbit0 = ct.c_long(orbit0)
    orbit1 = ct.c_long(orbit1)

    val_time0 = ct.c_double()
    val_time1 = ct.c_double()
    orbit_id = xo_orbit_id()
    n_files = ct.c_long(1)

    xo_orbit_init_file = getattr(EOCFI_LIBS["liborb"], "xo_orbit_init_file")

    # Define the types of the output and arguments of
    xo_orbit_init_file.restype = ct.c_long
    xo_orbit_init_file.argtypes = [
        ct.POINTER(ct.c_long),
        ct.POINTER(xl_model_id),
        ct.POINTER(xl_time_id),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_char_p),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(xo_orbit_id),
        ct.POINTER(ct.c_long * XO_NUM_ERR_ORBIT_INIT_FILE),
    ]

    ierr = (ct.c_long * XO_NUM_ERR_ORBIT_INIT_FILE)()
    errno = xo_orbit_init_file(
        ct.pointer(sat_id),
        ct.pointer(model_id),
        ct.pointer(time_id),
        ct.pointer(orbit_file_mode),
        ct.pointer(n_files),
        ct.pointer(orbit_file),
        ct.pointer(time_mode),
        ct.pointer(time_ref),
        ct.pointer(time0),
        ct.pointer(time1),
        ct.pointer(orbit0),
        ct.pointer(orbit1),
        ct.pointer(val_time0),
        ct.pointer(val_time1),
        ct.pointer(orbit_id),
        ierr,
    )

    if errno == XO_ERR:
        func_id = ct.c_long(XO_ORBIT_INIT_FILE_ID)
        msg_out = xl_get_msg(func_id, ierr)
        n = msg_out[0]
        msg = msg_out[1]

        xo_print_msg(n, msg)

    return orbit_id, val_time0, val_time1


def xo_osv_compute(orbit_id=None, mode=0, time_ref=XL_TIME_UTC, time_in=0.0):
    mode = ct.c_long(mode)
    time_ref = ct.c_long(time_ref)
    time = ct.c_double(time_in)
    pos_out = (ct.c_double * XYZ_ARRAY_SIZE)()
    vel_out = (ct.c_double * XYZ_ARRAY_SIZE)()
    acc_out = (ct.c_double * XYZ_ARRAY_SIZE)()
    xo_osv_compute = getattr(EOCFI_LIBS["liborb"], "xo_osv_compute")
    xo_osv_compute.restype = ct.c_long
    xo_osv_compute.argtypes = [
        ct.POINTER(xo_orbit_id),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double * XYZ_ARRAY_SIZE),
        ct.POINTER(ct.c_double * XYZ_ARRAY_SIZE),
        ct.POINTER(ct.c_double * XYZ_ARRAY_SIZE),
        ct.POINTER(ct.c_long * XO_NUM_ERR_OSV_COMPUTE),
    ]

    ierr = (ct.c_long * XO_NUM_ERR_OSV_COMPUTE)()
    errno = xo_osv_compute(
        ct.pointer(orbit_id),
        ct.pointer(mode),
        ct.pointer(time_ref),
        ct.pointer(time),
        pos_out,
        vel_out,
        acc_out,
        ierr,
    )
    if errno == XO_ERR:
        func_id = ct.c_long(XO_OSV_COMPUTE_ID)
        msg_out = xl_get_msg(func_id, ierr)
        n = msg_out[0]
        msg = msg_out[1]

        xo_print_msg(n, msg)

    return pos_out, vel_out, acc_out


def xo_orbit_get_anx(orbit_id: xo_orbit_id):
    assert isinstance(orbit_id, xo_orbit_id)

    # outs
    num_rec = ct.c_long()
    abs_orbit = ct.c_int()
    tanx = ct.c_double()
    tnod = ct.c_double()
    # extra_info = xo_anx_extra_info()

    xo_orbit_get_anx = getattr(EOCFI_LIBS["liborb"], "xo_orbit_get_anx")
    xo_orbit_get_anx.restype = ct.c_long
    xo_orbit_get_anx.argtypes = [
        ct.POINTER(xo_orbit_id),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]

    # ct.POINTER(xo_anx_extra_info)]

    xo_orbit_get_anx(
        ct.pointer(orbit_id),
        ct.pointer(num_rec),
        ct.pointer(abs_orbit),
        ct.pointer(tanx),
        ct.pointer(tnod),
    )

    return num_rec, abs_orbit, tanx, tnod
