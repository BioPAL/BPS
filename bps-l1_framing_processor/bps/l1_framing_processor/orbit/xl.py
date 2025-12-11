# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import ctypes as ct

import numpy as np
from bps.l1_framing_processor.orbit.loader import EOCFI_LIBS

XL_MODEL_DEFAULT = 0
XL_NUM_MODEL_TYPES_ENUM = 9
XL_NUM_ERR_MODEL_INIT = 1
XL_MAX_STR = 512
XL_MAX_COD = 256
XL_NUM_ERR_TIME_REF_INIT_FILE = 1
XL_NUM_ERR_POSITION_ON_ORBIT = 1
XL_NUM_ERR_TIME_CLOSE = 1
XL_MODEL_INIT_ID = 42
XL_ERR = -1
XL_TIMEMOD_AUTO = -2
XL_SEL_FILE = 0
XL_TIME_UTC = 1
XL_TIME_REF_INIT_FILE_ID = 13
XL_POSITION_ON_ORBIT = 32
XL_GM2000 = 3  #
XL_SAT_SENTINEL_1A = 110  #
XL_SAT_BIOMASS = 141
XL_EF = 7
XL_GM2000 = 3
XL_CALC_POS = 1
XL_ANGLE_TYPE_TRUE_LAT_EF = 0

XYZ_ARRAY_SIZE = 3
XP_DER_2ND = 2


class xl_model_id(ct.Structure):
    _fields_ = [("ee_id", ct.c_void_p)]


class xl_time_id(ct.Structure):
    _fields_ = [("ee_id", ct.c_void_p)]


# long xl_print_msg(long* n, char msg[XL_MAX_COD][XL_MAX_STR])
def xl_print_msg(n, msg):
    print(ct.string_at(msg))


def xl_get_msg(func_id, ierr):
    msg = (ct.c_char_p * XL_MAX_COD * XL_MAX_STR)()
    n = ct.c_long()
    xl_get_msg = getattr(EOCFI_LIBS["libcfi"], "xl_get_msg")
    xl_get_msg.restype = ct.c_long
    xl_get_msg.argtypes = [
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        (ct.c_char_p * XL_MAX_COD * XL_MAX_STR),
    ]

    xl_get_msg(ct.pointer(func_id), ierr, ct.pointer(n), msg)
    return n, msg


def xl_model_init(mode=0, models=None):
    if models is None:
        models = np.zeros(XL_NUM_MODEL_TYPES_ENUM, dtype=int)

    model_id = xl_model_id()

    mode = ct.c_long(mode)
    if len(models) != XL_NUM_MODEL_TYPES_ENUM:
        raise ValueError

    ModelsArray = ct.c_long * XL_NUM_MODEL_TYPES_ENUM
    models = ModelsArray(*models)

    xl_model_init = getattr(EOCFI_LIBS["libcfi"], "xl_model_init")
    xl_model_init.restype = ct.c_long
    xl_model_init.argtypes = [
        ct.POINTER(ct.c_long),
        ct.POINTER(ct.c_long),
        ct.POINTER(xl_model_id),
        ct.POINTER(ct.c_long * XL_NUM_ERR_MODEL_INIT),
    ]

    ierr = (ct.c_long * XL_NUM_ERR_MODEL_INIT)()
    errno = xl_model_init(ct.pointer(mode), models, ct.pointer(model_id), ierr)
    if errno == XL_ERR:
        func_id = ct.c_long(XL_MODEL_INIT_ID)
        msg_out = xl_get_msg(func_id, ierr)
        n = msg_out[0]
        msg = msg_out[1]

        xl_print_msg(n, msg)

    return model_id


def xl_time_ref_init_file(
    time_mode=XL_TIMEMOD_AUTO,
    time_file="",
    time_init_mode=XL_SEL_FILE,
    time_ref=XL_TIME_UTC,
    time0=0,
    time1=0,
    orbit0=0,
    orbit1=0,
):
    time_mode = ct.c_long(time_mode)
    n_files = ct.c_long(1)
    time_file = ct.c_char_p(time_file.encode("utf-8"))
    time_init_mode = ct.c_long(time_init_mode)
    time_ref = ct.c_long(time_ref)
    time0 = ct.c_double(time0)
    time1 = ct.c_double(time1)
    orbit0 = ct.c_long(orbit0)
    orbit1 = ct.c_long(orbit1)
    val_time0 = ct.c_double()
    val_time1 = ct.c_double()
    time_id = xl_time_id()

    xl_time_ref_init_file = getattr(EOCFI_LIBS["libcfi"], "xl_time_ref_init_file")
    xl_time_ref_init_file.restype = ct.c_long
    xl_time_ref_init_file.argtypes = [
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
        ct.POINTER(xl_time_id),
        ct.POINTER(ct.c_long * XL_NUM_ERR_TIME_REF_INIT_FILE),
    ]

    ierr = (ct.c_long * XL_NUM_ERR_TIME_REF_INIT_FILE)()
    errno = xl_time_ref_init_file(
        ct.pointer(time_mode),
        ct.pointer(n_files),
        ct.pointer(time_file),
        ct.pointer(time_init_mode),
        ct.pointer(time_ref),
        ct.pointer(time0),
        ct.pointer(time1),
        ct.pointer(orbit0),
        ct.pointer(orbit1),
        ct.pointer(val_time0),
        ct.pointer(val_time1),
        ct.pointer(time_id),
        ierr,
    )

    if errno == XL_ERR:
        func_id = ct.c_long(XL_TIME_REF_INIT_FILE_ID)
        msg_out = xl_get_msg(func_id, ierr)
        n = msg_out[0]
        msg = msg_out[1]

        xl_print_msg(n, msg)

    return time_id, val_time0, val_time1


def xl_position_on_orbit(
    model_id,
    time_id,
    angle_type=XL_ANGLE_TYPE_TRUE_LAT_EF,
    time_ref=XL_TIME_UTC,
    time=0.0,
    pos=0.0,
    vel=0.0,
    acc=0.0,
    deriv=XP_DER_2ND,
):
    # inputs
    angle_type = ct.c_long(angle_type)
    time_ref = ct.c_long(time_ref)
    time = ct.c_double(time)
    pos = (ct.c_double * 3)(*pos)
    vel = (ct.c_double * 3)(*vel)
    acc = (ct.c_double * 3)(*acc)
    deriv = ct.c_long(deriv)

    # output
    angle = ct.c_double()
    angle_rate = ct.c_double()
    angle_rate_rate = ct.c_double()
    ierr = (ct.c_long * XL_NUM_ERR_POSITION_ON_ORBIT)()

    # Define the call
    xl_position_on_orbit = getattr(EOCFI_LIBS["libcfi"], "xl_position_on_orbit")
    xl_position_on_orbit.restype = ct.c_long

    xl_position_on_orbit.argtypes = [  # IN
        ct.POINTER(xl_model_id),
        ct.POINTER(xl_time_id),
        ct.POINTER(ct.c_long),  # angle_type
        ct.POINTER(ct.c_long),  # time_ref,
        ct.POINTER(ct.c_double),  # time
        ct.POINTER(ct.c_double * XYZ_ARRAY_SIZE),  # pos
        ct.POINTER(ct.c_double * XYZ_ARRAY_SIZE),  # vel
        ct.POINTER(ct.c_double * XYZ_ARRAY_SIZE),  # acc
        ct.POINTER(ct.c_long),  # deriv
        # out
        ct.POINTER(ct.c_double),  # angle
        ct.POINTER(ct.c_double),  # angle_rate
        ct.POINTER(ct.c_double),  # angle_rate_rate
        ct.POINTER(ct.c_long * XL_NUM_ERR_POSITION_ON_ORBIT),
    ]
    # actual call
    errno = xl_position_on_orbit(
        ct.pointer(model_id),
        ct.pointer(time_id),
        ct.pointer(angle_type),
        ct.pointer(time_ref),
        ct.pointer(time),
        ct.pointer(pos),
        ct.pointer(vel),
        ct.pointer(acc),
        ct.pointer(deriv),
        ct.pointer(angle),
        ct.pointer(angle_rate),
        ct.pointer(angle_rate_rate),
        ierr,
    )

    if errno == XL_ERR:
        func_id = ct.c_long(XL_POSITION_ON_ORBIT)
        msg_out = xl_get_msg(func_id, ierr)
        n = msg_out[0]
        msg = msg_out[1]
        xl_print_msg(n, msg)
    return angle, angle_rate, angle_rate_rate
