# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
EO CFI orbit module
"""

import numpy as np
from bps.l1_framing_processor.orbit.eoorbit import EOOrbit
from bps.l1_framing_processor.orbit.xl import (
    XL_ANGLE_TYPE_TRUE_LAT_EF,
    XL_TIME_UTC,
    XP_DER_2ND,
    xl_model_init,
    xl_position_on_orbit,
    xl_time_ref_init_file,
)
from bps.l1_framing_processor.orbit.xorb import xo_orbit_init_file, xo_osv_compute


class EOCFIOrbit(EOOrbit):
    """EOCFIOrbit class

    Parameters
    ----------
    EOOrbit : EOOrbit
        Generic EO orbit class
    """

    def __init__(self, orbit_file: str):
        """Initialise EOCFIOrbit object

        Parameters
        ----------
        orbit_file : str
            Path to orbit file
        """
        super().__init__(orbit_file)

        self.model_id = xl_model_init()

        time_id, val_time0, val_time1 = xl_time_ref_init_file(time_file=self.orbit_file)
        self.time_id = time_id
        self.start_time = val_time0.value
        self.stop_time = val_time1.value

        orbit_id, val_time0, val_time1 = xo_orbit_init_file(
            model_id=self.model_id, time_id=self.time_id, orbit_file=self.orbit_file
        )
        self.orbit_id = orbit_id

    def get_pos_vel_acc(self, t_az_axis):
        """Get satellite position/velocity/acceleration at given time(s)

        Parameters
        ----------
        t_az_axis : np.array of np.datetime64
            Azimuth time(s)

        Returns
        -------
        np.array (3x)
            Satellite position/velocity/acceleration at given time(s)
        """
        t_az_axis = np.atleast_1d(t_az_axis)
        pos = []
        vel = []
        acc = []
        for t_az in t_az_axis:
            pos_curr, vel_curr, acc_curr = xo_osv_compute(orbit_id=self.orbit_id, time_in=t_az)
            pos.append(pos_curr)
            vel.append(vel_curr)
            acc.append(acc_curr)
        pos = np.asarray(pos)
        vel = np.asarray(vel)
        acc = np.asarray(acc)

        return pos, vel, acc

    def get_ops_angle(self, t_az_axis, pos, vel, acc):
        """Get satellite OPS angle at given time(s)

        Parameters
        ----------
        t_az_axis : np.array of np.datetime64
            Azimuth time(s)
        pos : np.array
            Satellite position at given time(s)
        vel : np.array
            Satellite velocity at given time(s)
        acc : np.array
            Satellite acceleration at given time(s)

        Returns
        -------
        list of float
            Satellite OPS angle at given time(s)
        """
        t_az_axis = np.atleast_1d(t_az_axis)

        angle = []
        for i, t_az in enumerate(t_az_axis):
            arg = {
                "model_id": self.model_id,
                "time_id": self.time_id,
                "angle_type": XL_ANGLE_TYPE_TRUE_LAT_EF,
                "time_ref": XL_TIME_UTC,
                "time": t_az,
                "pos": pos[i, :],
                "vel": vel[i, :],
                "acc": acc[i, :],
                "deriv": XP_DER_2ND,
            }

            angle_curr, _, __ = xl_position_on_orbit(**arg)
            angle.append(angle_curr.value)

        return angle
